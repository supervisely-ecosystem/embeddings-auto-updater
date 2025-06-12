from typing import Any, Dict, List
from urllib.parse import urlparse

import numpy as np
import supervisely as sly
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Batch, CollectionInfo, Distance, VectorParams

import src.globals as g
from src.utils import ImageInfoLite, QdrantFields, TupleFields, timeit, with_retries


def create_client_from_url(url: str) -> AsyncQdrantClient:
    """Create a Qdrant client instance from URL.

    Args:
        url: The Qdrant service URL in format http(s)://<host>[:port]

    Returns:
        AsyncQdrantClient: Configured client instance
    """
    parsed_host = urlparse(url)

    # Validate URL format
    if parsed_host.scheme not in ["http", "https"]:
        raise ValueError(f"Qdrant host should be in format http(s)://<host>[:port], got {url}")

    # Create client with appropriate settings based on URL
    return AsyncQdrantClient(
        url=parsed_host.hostname,
        https=parsed_host.scheme == "https",
        port=parsed_host.port or 6333,  # Default Qdrant port
    )


client = create_client_from_url(g.qdrant_host)


try:
    sly.logger.info(f"Connecting to Qdrant at {g.qdrant_host}...")
    QdrantClient(g.qdrant_host).get_collections()
    sly.logger.info(f"Connected to Qdrant at {g.qdrant_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to Qdrant at {g.qdrant_host}: {e}")


class SearchResultField:
    ITEMS = "items"
    VECTORS = "vectors"
    SCORES = "scores"


@with_retries()
async def delete_collection_items(
    collection_name: str,
    image_infos: List[sly.ImageInfo],
) -> Dict[str, Any]:
    """Delete a collection items with the specified IDs.

    :param collection_name: The name of the collection to delete items from
    :type collection_name: str
    :param image_infos: A list of ImageInfo objects to delete.
    :type image_infos: List[ImageInfo]
    :return: The payloads of the deleted items.
    :rtype: Dict[str, Any]
    """

    msg_prefix = f"[Project ID: {collection_name}]"

    ids = [info.id for info in image_infos]

    sly.logger.debug(f"{msg_prefix} Deleting items from collection...", extra={"ids": ids})
    try:
        await client.delete(collection_name, ids, wait=False)
    except UnexpectedResponse as e:
        sly.logger.debug(
            f"{msg_prefix} Something went wrong, while deleting items from collection.",
            exc_info=e,
            extra={"ids": ids},
        )


@with_retries()
@timeit
async def get_or_create_collection(
    collection_name: str, size: int = 512, distance: Distance = Distance.COSINE
) -> CollectionInfo:
    """Get or create a collection with the specified name.

    :param collection_name: The name of the collection to get or create.
    :type collection_name: str
    :param size: The size of the vectors in the collection, defaults to 512.
    :type size: int, optional
    :param distance: The distance metric to use for the collection, defaults to Distance.COSINE.
    :type distance: Distance, optional
    :return: The CollectionInfo object.
    :rtype: CollectionInfo
    """

    try:
        collection = await client.get_collection(collection_name)
        sly.logger.debug("Collection %s already exists.", collection_name)
    except UnexpectedResponse:
        await client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        sly.logger.debug("Collection %s created.", collection_name)

        # Create necessary indexes for efficient filtering

        await client.create_payload_index(
            collection_name=collection_name,
            field_name=f"{QdrantFields.DATASET_ID}",
            field_schema="keyword",
        )

        sly.logger.debug(
            f"{QdrantFields.DATASET_ID} field indexed for collection {collection_name}"
        )

        collection = await client.get_collection(collection_name)
    return collection


async def collection_exists(collection_name: str) -> bool:
    """Check if a collection with the specified name exists.

    :param collection_name: The name of the collection to check.
    :type collection_name: str
    :return: True if the collection exists, False otherwise.
    :rtype: bool
    """

    try:
        await client.get_collection(collection_name)
        return True
    except UnexpectedResponse:
        return False


@with_retries(retries=5, sleep_time=2)
@timeit
async def upsert(
    collection_name: str,
    vectors: List[np.ndarray],
    image_infos: List[ImageInfoLite],
) -> None:
    """Upsert vectors and payloads to the collection.

    :param collection_name: The name of the collection to upsert to.
    :type collection_name: str
    :param vectors: A list of vectors to upsert.
    :type vectors: List[np.ndarray]
    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]
    """

    ids = [image_info.id for image_info in image_infos]
    payloads = create_payloads(image_infos)
    sly.logger.debug("Upserting %d vectors to collection %s.", len(vectors), collection_name)
    await client.upsert(collection_name, Batch(vectors=vectors, ids=ids, payloads=payloads))

    if sly.is_development():
        # By default qdrant should overwrite vectors with the same ids
        # so this line is only needed to check if vectors were upserted correctly.
        # Do not use this in production since it will slow down the process.
        collecton_info = await client.get_collection(collection_name)
        sly.logger.debug(
            "Collection %s has %d vectors.", collection_name, collecton_info.points_count
        )


@timeit
def create_payloads(image_infos: List[ImageInfoLite]) -> List[Dict[str, Any]]:
    """
    Prepare payloads for ImageInfoLite objects before upserting to Qdrant.
    Converts named tuples to dictionaries and removes fields:
       - ID
       - SCORE

    :param image_infos: A list of ImageInfoLite objects.
    :type image_infos: List[ImageInfoLite]    :
    :return: A list of payloads.
    :rtype: List[Dict[str, Any]]
    """
    ignore_fields = [TupleFields.ID, TupleFields.SCORE]
    payloads = [
        {k: v for k, v in image_info.to_json().items() if k not in ignore_fields}
        for image_info in image_infos
    ]
    return payloads


@with_retries()
async def delete_collection(collection_name: str) -> None:
    """Delete a collection with the specified name.

    :param collection_name: The name of the collection to delete.
    :type collection_name: str
    """
    sly.logger.debug(f"Deleting collection {collection_name}...")
    try:
        await client.delete_collection(collection_name)
        sly.logger.debug(f"Collection {collection_name} deleted.")
    except UnexpectedResponse:
        sly.logger.debug(f"Collection {collection_name} wasn't found while deleting.")
