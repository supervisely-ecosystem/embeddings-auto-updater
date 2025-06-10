import time
from typing import List, Optional, Tuple

import supervisely as sly
from supervisely.sly_logger import logger

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.utils import (
    create_lite_image_infos,
    fix_vectors,
    get_all_projects,
    get_project_info,
    get_team_info,
    image_get_list_async,
    parse_timestamp,
    set_image_embeddings_updated_at,
    set_project_embeddings_updated_at,
    timeit,
)


@timeit
async def process_images(
    api: sly.Api,
    project_id: int,
    to_create: List[sly.ImageInfo],
    to_delete: List[sly.ImageInfo],
    return_vectors: bool = False,
    check_collection_exists: bool = True,
) -> Tuple[List[sly.ImageInfo], List[List[float]]]:
    """Process images from the specified project. Download images, save them to HDF5,
    get vectors from the images and upsert them to Qdrant.

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to process images from.
    :type project_id: int
    :param to_create: List of image infos to create in Qdrant.
    :type to_create: List[sly.ImageInfo]
    :param to_delete: List of image infos to delete from Qdrant.
    :type to_delete: List[sly.ImageInfo]
    :param return_vectors: If True, return vectors of the created images.
    :type return_vectors: bool
    :param check_collection_exists: If True, check if the Qdrant collection exists.
    :type check_collection_exists: bool
    :return: Tuple of two lists: list of created image infos and list of vectors.
    :rtype: Tuple[List[sly.ImageInfo], List[List[float]]]
    """

    collection_msg = f"[Collection: {project_id}]"

    to_create = await create_lite_image_infos(
        cas_size=g.IMAGE_SIZE_FOR_CAS,
        image_infos=to_create,
    )

    if len(to_create) == 0 and len(to_delete) == 0:
        logger.debug(f"{collection_msg} All images are up-to-date.")
        return to_create
    # if await qdrant.collection_exists(project_id):
    # Get diff of image infos, check if they are already in the Qdrant collection

    if check_collection_exists:
        await qdrant.get_or_create_collection(project_id)

    current_progress = 0
    total_progress = len(to_create)

    vectors = []
    if len(to_create) > 0:
        logger.debug(f"{collection_msg} Images to be vectorized: {total_progress}.")
        for image_batch in sly.batched(to_create):
            # Get vectors from images.
            vectors_batch = await cas.get_vectors(
                [image_info.cas_url for image_info in image_batch]
            )
            vectors_batch = fix_vectors(vectors_batch)
            logger.debug(f"{collection_msg} Got {len(vectors_batch)} vectors for images.")

            # Upsert vectors to Qdrant.
            await qdrant.upsert(project_id, vectors_batch, image_batch)
            current_progress += len(image_batch)
            logger.debug(
                f"{collection_msg} Upserted {len(vectors_batch)} vectors to Qdrant. [{current_progress}/{total_progress}]",
            )
            await set_image_embeddings_updated_at(api, image_batch)

            if return_vectors:
                vectors.extend(vectors_batch)

        logger.debug(f"{collection_msg} All {total_progress} images have been vectorized.")

    for image_batch in sly.batched(to_delete):
        # Delete images from the Qdrant.
        await qdrant.delete_collection_items(collection_name=project_id, image_infos=image_batch)
        await set_image_embeddings_updated_at(api, image_batch, [None * len(image_batch)])
        logger.debug(f"{collection_msg} Deleted {len(image_batch)} images from Qdrant.")

    return to_create, vectors


@timeit
async def update_embeddings(
    api: sly.Api,
    project_id: int,
    force: bool = False,
    project_info: Optional[sly.ProjectInfo] = None,
):
    collection_msg = f"[Collection: {project_id}] "

    if project_info is None:
        project_info = await get_project_info(api, project_id)

    if force:
        logger.info(f"{collection_msg} Force enabled, recreating embeddings for all images.")
        await qdrant.delete_collection(project_id)
        # do not need to create collection here, it will be created in process_images
        images_to_create = await image_get_list_async(api, project_id)
        images_to_delete = []
    elif project_info.embeddings_updated_at is None:
        # do not need to check or create collection here, it will be created in process_images
        logger.info(
            f"{collection_msg} Embeddings are not updated yet, creating embeddings for all images."
        )
        images_to_create = await image_get_list_async(api, project_id)
        images_to_delete = []
    elif parse_timestamp(project_info.embeddings_updated_at) < parse_timestamp(
        project_info.updated_at
    ):
        logger.info(
            f"{collection_msg} Embeddings are outdated, will check for images that need to be updated."
        )
        images_to_create = await image_get_list_async(api, project_id, wo_embeddings=True)
        if project_info.embeddings_updated_at is not None:
            images_to_delete = await image_get_list_async(
                api, project_id, deleted_after=project_info.embeddings_updated_at
            )
        else:
            images_to_delete = []

    else:
        logger.debug("Embeddings for project %d are up-to-date.", project_info.id)
        return
    image_infos = await process_images(api, project_id, images_to_create, images_to_delete)
    if len(image_infos) > 0:
        await set_image_embeddings_updated_at(api, image_infos)
        await set_project_embeddings_updated_at(api, project_id)


@timeit
async def auto_update_embeddings(
    api: sly.Api, project_id: int, project_info: Optional[sly.ProjectInfo] = None
):
    """
    Update embeddings for the specified project if needed.
    """
    project_message = f"[Project ID: {project_id}]"

    if project_info is None:
        project_info = await get_project_info(api, project_id)

    if project_info is None:
        logger.warning(f"{project_message} Info not found. Skipping embeddings update.")
        return

    team_info: sly.TeamInfo = await get_team_info(api, project_info.team_id)
    if team_info.usage is not None and team_info.usage.plan == "free":
        logger.info(
            f"{project_message} Embeddings update is not available on free plan.",
            extra={
                "project_id": project_id,
                "project_name": project_info.name,
                "team_id": team_info.id,
                "team_name": team_info.name,
            },
        )
        api.project.disable_embeddings(project_id)
        logger.info(
            f"{project_message} Embeddings are disabled for project due to free plan.",
            project_info.name,
            project_id,
        )
        return

    # Check if embeddings activated for the project
    log_extra = {
        "team_id": project_info.team_id,
        "workspace_id": project_info.workspace_id,
        "project_name": project_info.name,
        "project_id": project_id,
        "items_count": project_info.items_count,
        "updated_at": project_info.updated_at,
        "embeddings_enabled": project_info.embeddings_enabled,
        "embeddings_updated_at": project_info.embeddings_updated_at,
    }
    if not project_info.embeddings_enabled:
        logger.debug(f"{project_message} Embeddings are not activated.", extra=log_extra)
        return
    logger.debug(f"{project_message} Auto update embeddings started.", extra=log_extra)
    t = time.monotonic()
    await update_embeddings(api, project_id, force=False, project_info=project_info)
    t = time.monotonic() - t
    logger.debug(
        f"{project_message} Auto update finished. Time: {t} seconds.",
        extra={**log_extra, "time": t},
    )


@timeit
async def auto_update_all_embeddings():
    """Update embeddings for all available projects"""
    logger.info("Auto update all embeddings task started.")

    project_infos: List[sly.ProjectInfo] = await get_all_projects(g.api)
    for project_info in project_infos:
        await auto_update_embeddings(
            g.api,
            project_info.id,
            project_info=project_info,
        )
    logger.info("Auto update all embeddings task finished.")
