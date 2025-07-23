import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial, wraps
from time import perf_counter
from typing import Callable, Dict, List, Optional, Union

import httpx
import supervisely as sly
from supervisely._utils import batched, resize_image_url
from supervisely.api.module_api import ApiField
from supervisely.api.task_api import TaskApi


class TupleFields:
    """Fields of the named tuples used in the project."""

    ID = "id"
    HASH = "hash"
    LINK = "link"
    DATASET_ID = "dataset_id"
    PROJECT_ID = "project_id"
    FULL_URL = "full_url"
    CAS_URL = "cas_url"
    HDF5_URL = "hdf5_url"
    UPDATED_AT = "updated_at"
    UNIT_SIZE = "unitSize"
    URL = "url"
    THUMBNAIL = "thumbnail"
    ATLAS_ID = "atlasId"
    ATLAS_INDEX = "atlasIndex"
    VECTOR = "vector"
    IMAGES = "images"
    SCORE = "score"


class QdrantFields:
    """Fields for the queries to the Qdrant API."""

    KMEANS = "kmeans"
    NUM_CLUSTERS = "num_clusters"
    OPTION = "option"
    RANDOM = "random"
    CENTROIDS = "centroids"

    # Payload Fields
    DATASET_ID = "dataset_id"
    IMAGE_ID = "image_id"
    ID = "id"


class EventFields:
    """Fields of the event in request objects."""

    PROJECT_ID = "project_id"
    DATASET_ID = "dataset_id"
    TEAM_ID = "team_id"
    IMAGE_IDS = "image_ids"
    FORCE = "force"
    PROMPT = "prompt"
    LIMIT = "limit"
    METHOD = "method"
    REDUCTION_DIMENSIONS = "reduction_dimensions"
    SAMPLING_METHOD = "sampling_method"
    SAMPLE_SIZE = "sample_size"
    CLUSTERING_METHOD = "clustering_method"
    NUM_CLUSTERS = "num_clusters"
    SAVE = "save"
    RETURN_VECTORS = "return_vectors"
    THRESHOLD = "threshold"

    ATLAS = "atlas"
    POINTCLOUD = "pointcloud"

    # Search by fields
    BY_PROJECT_ID = "by_project_id"
    BY_DATASET_ID = "by_dataset_id"
    BY_IMAGE_IDS = "by_image_ids"

    # Event types
    SEARCH = "search"
    DIVERSE = "diverse"
    CLUSTERING = "clustering"
    EMBEDDINGS = "embeddings"


class SamplingMethods:
    """Sampling methods for the images."""

    RANDOM = "random"
    CENTROIDS = "centroids"


class ClusteringMethods:
    """Clustering methods for the images."""

    KMEANS = "kmeans"
    DBSCAN = "dbscan"


class ResponseFields:
    """Fields of the response file."""

    COLLECTION_ID = "collection_id"
    MESSAGE = "message"
    STATUS = "status"
    VECTORS = "vectors"
    IMAGE_IDS = "image_ids"
    BACKGROUND_TASK_ID = "background_task_id"
    RESULT = "result"
    IS_RUNNING = "is_running"


class ResponseStatus:
    """Status of the response."""

    SUCCESS = "success"
    COMPLETED = "completed"
    ERROR = "error"
    IN_PROGRESS = "in_progress"
    NOT_FOUND = "not_found"
    NO_TASK = "no_task"
    CANCELLED = "cancelled"
    FAILED = "failed"
    RUNNING = "running"


class CustomDataFields:
    """Fields of the custom data."""

    EMBEDDINGS_UPDATE_STARTED_AT = "embeddings_update_started_at"


@dataclass
class ImageInfoLite:
    id: int
    dataset_id: int
    full_url: str
    cas_url: str
    updated_at: str  # or datetime.datetime if you parse it
    score: float = None

    def to_json(self):
        return {
            TupleFields.ID: self.id,
            TupleFields.DATASET_ID: self.dataset_id,
            TupleFields.FULL_URL: self.full_url,
            TupleFields.CAS_URL: self.cas_url,
            TupleFields.UPDATED_AT: self.updated_at,
            TupleFields.SCORE: self.score,
        }
        # Alternative: return asdict(self)  # if field names match keys

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            id=data[TupleFields.ID],
            dataset_id=data[TupleFields.DATASET_ID],
            full_url=data[TupleFields.FULL_URL],
            cas_url=data[TupleFields.CAS_URL],
            updated_at=data[TupleFields.UPDATED_AT],
            score=data.get(TupleFields.SCORE, None),
        )


def timeit(func: Callable) -> Callable:
    """Decorator to measure the execution time of the function.
    Works with both async and sync functions.

    :param func: Function to measure the execution time of.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    """

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = await func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            _log_execution_time(func.__name__, execution_time)
            return result

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            _log_execution_time(func.__name__, execution_time)
            return result

        return sync_wrapper


def _log_execution_time(function_name: str, execution_time: float) -> None:
    """Log the execution time of the function.

    :param function_name: Name of the function.
    :type function_name: str
    :param execution_time: Execution time of the function.
    :type execution_time: float
    """
    sly.logger.debug("%.4f sec | %s", execution_time, function_name)


def to_thread(func: Callable) -> Callable:
    """Decorator to run the function in a separate thread.
    Can be used for slow synchronous functions inside of the asynchronous code
    to avoid blocking the event loop.

    :param func: Function to run in a separate thread.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    """

    # For Python 3.9+.
    # @wraps(func)
    # def wrapper(*args, **kwargs):
    #     return asyncio.to_thread(func, *args, **kwargs)

    # For Python 3.7+.
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        func_with_args = partial(func, *args, **kwargs)
        return loop.run_in_executor(None, func_with_args)

    return wrapper


def with_retries(retries: int = 3, sleep_time: int = 1, on_failure: Callable = None) -> Callable:
    """Decorator to retry the function in case of an exception.
    Works only with async functions. Custom function can be executed on failure.
    NOTE: The on_failure function should be idempotent and synchronous.

    :param retries: Number of retries.
    :type retries: int
    :param sleep_time: Time to sleep between retries.
    :type sleep_time: int
    :param on_failure: Function to execute on failure, if None, raise an exception.
    :type on_failure: Callable, optional
    :raises Exception: If the function fails after all retries.
    :return: Decorator.
    :rtype: Callable
    """

    def retry_decorator(func):
        @wraps(func)
        async def async_function_with_retries(*args, **kwargs):
            for _ in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    sly.logger.debug(
                        "Failed to execute %s, retrying. Error: %s", func.__name__, str(e)
                    )
                    await asyncio.sleep(sleep_time)
            if on_failure is not None:
                return on_failure()
            else:
                raise RuntimeError(f"Failed to execute {func.__name__} after {retries} retries.")

        return async_function_with_retries

    return retry_decorator


@to_thread
@timeit
def get_project_info(api: sly.Api, project_id: int) -> sly.ProjectInfo:
    """Returns project info by ID.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param project_id: ID of the project to get info.
    :type project_id: int
    :return: Project info.
    :rtype: sly.ProjectInfo
    """
    return api.project.get_info_by_id(project_id)


@to_thread
@timeit
def get_team_info(api: sly.Api, team_id: int) -> sly.TeamInfo:
    """Returns team info by ID.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param team_id: ID of the team to get info.
    :type team_id: int
    :return: Team info.
    :rtype: sly.TeamInfo
    """
    return api.team.get_info_by_id(team_id)


def _get_project_info_by_name(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    return api.project.get_info_by_name(workspace_id, project_name)


@to_thread
@timeit
def get_project_info_by_name(api: sly.Api, workspace_id: int, project_name: str) -> sly.ProjectInfo:
    """Returns project info by name.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param workspace_id: ID of the workspace to get project from.
    :type workspace_id: int
    :param project_name: Name of the project to get info.
    :type project_name: str
    :return: Project info.
    :rtype: sly.ProjectInfo
    """
    return _get_project_info_by_name(api, workspace_id, project_name)


@to_thread
@timeit
def set_image_embeddings_updated_at(
    api: sly.Api,
    image_infos: List[Union[sly.ImageInfo, ImageInfoLite]],
    timestamps: Optional[List[str]] = None,
):
    """Sets the embeddings updated at timestamp for the images."""
    ids = [image_info.id for image_info in image_infos]
    ids = list(set(ids))
    api.image.set_embeddings_updated_at(ids, timestamps)


@to_thread
@timeit
def set_project_embeddings_updated_at(
    api: sly.Api,
    project_id: int,
    timestamp: str = None,
):
    """Sets the embeddings updated at timestamp for the project."""
    api.project.set_embeddings_updated_at(project_id, timestamp)


@to_thread
@timeit
def get_project_embeddings_updated_at(api: sly.Api, project_id: int) -> Optional[str]:
    """Gets the embeddings updated at timestamp for the project."""
    project_info = api.project.get_info_by_id(project_id)
    return project_info.embeddings_updated_at


@to_thread
@timeit
def set_embeddings_in_progress(api: sly.Api, project_id: int, in_progress: bool):
    """Sets the embeddings in progress flag for the project."""
    api.project.set_embeddings_in_progress(id=project_id, in_progress=in_progress)


@to_thread
@timeit
def get_team_file_info(api: sly.Api, team_id: int, path: str):
    return api.file.get_info_by_path(team_id, path)


@timeit
async def create_lite_image_infos(
    cas_size: int,
    image_infos: List[sly.ImageInfo],
) -> List[ImageInfoLite]:
    """Returns lite version of image infos to cut off unnecessary data.

    :param cas_size: Size of the image for CLIP, it will be added to URL.
    :type cas_size: int
    :param image_infos: List of image infos to get lite version from.
    :type image_infos: List[sly.ImageInfo]
    :return: List of lite version of image infos.
    :rtype: List[ImageInfoLite]
    """
    if not image_infos or len(image_infos) == 0:
        return []
    if isinstance(image_infos[0], ImageInfoLite):
        return image_infos
    return [
        ImageInfoLite(
            id=image_info.id,
            dataset_id=image_info.dataset_id,
            full_url=image_info.full_storage_url,
            cas_url=resize_image_url(
                image_info.full_storage_url,
                method="fit",
                width=cas_size,
                height=cas_size,
            ),
            updated_at=image_info.updated_at,
        )
        for image_info in image_infos
    ]


def parse_timestamp(timestamp: str, timestamp_format: str = "%Y-%m-%dT%H:%M:%S.%fZ") -> datetime:
    """
    Parse timestamp string to datetime object.
    Timestamp format: "2021-01-22T19:37:50.158Z".
    """
    return datetime.strptime(timestamp, timestamp_format)


async def send_request(
    api: sly.Api,
    task_id: int,
    method: str,
    data: Dict,
    context: Optional[Dict] = None,
    skip_response: bool = False,
    timeout: Optional[int] = 60,
    outside_request: bool = True,
    retries: int = 10,
    raise_error: bool = False,
):
    """send_request"""
    if type(data) is not dict:
        raise TypeError("data argument has to be a dict")
    if context is None:
        context = {}
    context["outside_request"] = outside_request
    resp = await api.post_async(
        "tasks.request.direct",
        {
            ApiField.TASK_ID: task_id,
            ApiField.COMMAND: method,
            ApiField.CONTEXT: context,
            ApiField.STATE: data,
            "skipResponse": skip_response,
            "timeout": timeout,
        },
        retries=retries,
        raise_error=raise_error,
    )
    return resp.json()


@timeit
def fix_vectors(vectors_batch):
    for i, vector in enumerate(vectors_batch):
        for j, value in enumerate(vector):
            if not isinstance(value, float):
                sly.logger.debug(
                    "Value %s is not of type float: %s. Converting to float.", value, type(value)
                )
                vectors_batch[i][j] = float(value)
    return vectors_batch


async def run_safe(func, *args, **kwargs):
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        sly.logger.error("Error in function %s: %s", func.__name__, e, exc_info=True)
        return None


def get_filter_images_wo_embeddings() -> Dict:
    """Create filter to get images that dont have embeddings.

    :return: Dictionary representing the filter.
    :rtype: Dict
    """
    return {
        ApiField.FIELD: ApiField.EMBEDDINGS_UPDATED_AT,
        ApiField.OPERATOR: "eq",
        ApiField.VALUE: None,
    }


def get_filter_images_with_embeddings() -> Dict:
    """Create filter to get images that dont have embeddings.

    :return: Dictionary representing the filter.
    :rtype: Dict
    """
    return {
        ApiField.FIELD: ApiField.EMBEDDINGS_UPDATED_AT,
        ApiField.OPERATOR: "not",
        ApiField.VALUE: None,
    }


def get_filter_deleted_after(timestamp: str) -> Dict:
    """Create filter to get images deleted after a specific date.

    :return: Dictionary representing the filter.
    :rtype: Dict
    """
    return {
        ApiField.FIELD: ApiField.UPDATED_AT,
        ApiField.OPERATOR: "gt",
        ApiField.VALUE: timestamp,
    }


@timeit
async def image_get_list_async(
    api: sly.Api,
    project_id: int,
    dataset_id: int = None,
    image_ids: List[int] = None,
    per_page: int = 1000,
    wo_embeddings: Optional[bool] = False,
    deleted_after: Optional[str] = None,
) -> List[sly.ImageInfo]:
    """
    Get list of images from the project or dataset.
     - If `image_ids` is provided, it will return only those images.
     - If `dataset_id` is provided, it will return images from that dataset.
     - If neither `dataset_id` nor `image_ids` is provided, it will return all images from the project.
     - If `wo_embeddings` is True, it will return only images without embeddings.
     - If `deleted_after` is provided, it will return only images that were updated after that date.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param project_id: ID of the project to get images from.
    :type project_id: int
    :param dataset_id: ID of the dataset to get images from. If None, will get images from the whole project.
    :type dataset_id: int, optional
    :param image_ids: List of image IDs to get images from. If None, will get all images.
    :type image_ids: List[int], optional
    :param per_page: Number of images to return per page. Default is 1000.
    :type per_page: int
    :param wo_embeddings: If True, will return only images without embeddings. Default is False.
    :type wo_embeddings: bool, optional
    :param deleted_after: If provided, will return only images that were updated after this date.
    :type deleted_after: str, optional
    :return: List of images from the project or dataset.
    :rtype: List[sly.ImageInfo]
    :raises ValueError: If both `wo_embeddings` and `deleted_after` are set to True.
    """
    method = "images.list"
    base_data = {
        ApiField.PROJECT_ID: project_id,
        ApiField.FORCE_METADATA_FOR_LINKS: False,
        ApiField.PER_PAGE: per_page,
    }

    if dataset_id is not None:
        base_data[ApiField.DATASET_ID] = dataset_id

    if wo_embeddings and deleted_after:
        raise ValueError("Both created_after and deleted_after cannot be set at the same time.")
    if wo_embeddings:
        base_data[ApiField.FILTER] = [get_filter_images_wo_embeddings()]
    if deleted_after is not None:
        if ApiField.FILTER not in base_data:
            base_data[ApiField.FILTER] = []
        base_data[ApiField.FILTER].append(get_filter_deleted_after(deleted_after))
        base_data[ApiField.FILTER].append(get_filter_images_with_embeddings())
        base_data[ApiField.SHOW_DISABLED] = True

    semaphore = api.get_default_semaphore()
    all_items = []
    tasks = []

    async def _get_all_pages(ids_filter: List[Dict]):
        page_data = base_data.copy()
        if ids_filter:
            if ApiField.FILTER not in page_data:
                page_data[ApiField.FILTER] = []
            page_data[ApiField.FILTER].extend(ids_filter)

        page_data[ApiField.PAGE] = 1
        first_response = await api.post_async(method, page_data)
        first_response_json = first_response.json()

        total_pages = first_response_json.get("pagesCount", 1)
        batch_items = []

        entities = first_response_json.get("entities", [])
        for item in entities:
            image_info = api.image._convert_json_info(item)
            batch_items.append(image_info)

        if total_pages > 1:

            async def fetch_page(page_num):
                page_data_copy = page_data.copy()
                page_data_copy[ApiField.PAGE] = page_num

                async with semaphore:
                    response = await api.post_async(method, page_data_copy)
                    response_json = response.json()

                    page_items = []
                    entities = response_json.get("entities", [])
                    for item in entities:
                        image_info = api.image._convert_json_info(item)
                        page_items.append(image_info)

                    return page_items

            # Create tasks for all remaining pages
            tasks = []
            for page_num in range(2, total_pages + 1):
                tasks.append(asyncio.create_task(fetch_page(page_num)))

            page_results = await asyncio.gather(*tasks)

            for page_items in page_results:
                batch_items.extend(page_items)

        return batch_items

    if image_ids is None:
        # If no image IDs specified, get all images
        tasks.append(asyncio.create_task(_get_all_pages([])))
    else:
        # Process image IDs in batches of 50
        for batch in batched(image_ids):
            ids_filter = [
                {ApiField.FIELD: ApiField.ID, ApiField.OPERATOR: "in", ApiField.VALUE: batch}
            ]
            tasks.append(asyncio.create_task(_get_all_pages(ids_filter)))
            await asyncio.sleep(0.02)  # Small delay to avoid overwhelming the server

    # Wait for all tasks to complete
    batch_results = await asyncio.gather(*tasks)

    # Combine results from all batches
    for batch_items in batch_results:
        all_items.extend(batch_items)

    return all_items


async def embeddings_up_to_date(
    api: sly.Api, project_id: int, project_info: Optional[sly.ProjectInfo] = None
):
    if project_info is None:
        project_info = await get_project_info(api, project_id)
    if project_info is None:
        return None
    if project_info.embeddings_updated_at is None:
        return False
    images_to_create = await image_get_list_async(api, project_id, wo_embeddings=True)
    if len(images_to_create) > 0:
        return False
    return True


async def get_list_all_pages_async(
    api: sly.Api,
    method,
    data,
    convert_json_info_cb,
    progress_cb=None,
    limit: int = None,
    return_first_response: bool = False,
    semaphore: Optional[List[asyncio.Semaphore]] = None,
):
    """
    Get list of all or limited quantity entities from the Supervisely server.
    """
    from copy import deepcopy

    def _add_sort_param(data):
        """_add_sort_param"""
        results = deepcopy(data)
        results[ApiField.SORT] = ApiField.ID
        results[ApiField.SORT_ORDER] = "asc"  # @TODO: move to enum
        return results

    convert_func = convert_json_info_cb

    if ApiField.SORT not in data:
        data = _add_sort_param(data)
    if semaphore is None:
        semaphore = api.get_default_semaphore()

    # Get first page to determine pagination details
    first_page_data = {**data, "page": 1}
    first_response = await api.post_async(method, first_page_data)
    first_response_json = first_response.json()

    total = first_response_json["total"]
    per_page = first_response_json["perPage"]
    pages_count = first_response_json["pagesCount"]

    results = first_response_json["entities"]
    if progress_cb is not None:
        progress_cb(len(results))

    # If only one page or limit is already exceeded with first page
    if (pages_count == 1 and len(results) == total) or (
        limit is not None and len(results) >= limit
    ):
        if limit is not None:
            results = results[:limit]
        if return_first_response:
            return [convert_func(item) for item in results], first_response_json
        return [convert_func(item) for item in results]

    # Process remaining pages concurrently
    async def fetch_page(page_num):
        async with semaphore:
            page_data = {**data, "page": page_num, "per_page": per_page}
            response = await api.post_async(method, page_data)
            response_json = response.json()
            page_items = response_json.get("entities", [])
            if progress_cb is not None:
                progress_cb(len(page_items))
            return page_items

    # Create tasks for all remaining pages
    tasks = []
    for page_num in range(2, pages_count + 1):
        tasks.append(asyncio.create_task(fetch_page(page_num)))

    # Wait for all tasks to complete
    for task in asyncio.as_completed(tasks):
        page_items = await task
        results.extend(page_items)
        if limit is not None and len(results) >= limit:
            break

    if len(results) != total and limit is None:
        raise RuntimeError(f"Method {method!r}: error during pagination, some items are missed")

    if limit is not None:
        results = results[:limit]

    return [convert_func(item) for item in results]


@timeit
async def get_all_projects(
    api: sly.Api,
    project_ids: Optional[List[int]] = None,
    filters: Optional[List[Dict]] = None,
) -> List[sly.ProjectInfo]:
    """
    Get all projects from the Supervisely server that have a flag for automatic embeddings update.

    Fields that will be returned:
        - id
        - size
        - workspace_id
        - type
        - created_at
        - updated_at
        - name
        - team_id
        - embeddings_enabled
        - embeddings_in_progress
        - embeddings_updated_at

    """
    method = "projects.list.all"
    convert_json_info = api.project._convert_json_info
    fields = [
        ApiField.EMBEDDINGS_ENABLED,
        ApiField.EMBEDDINGS_IN_PROGRESS,
        ApiField.EMBEDDINGS_UPDATED_AT,
    ]
    data = {
        ApiField.SKIP_EXPORTED: True,
        ApiField.EXTRA_FIELDS: fields,
        ApiField.FILTER: [
            {
                ApiField.FIELD: ApiField.EMBEDDINGS_ENABLED,
                ApiField.OPERATOR: "=",
                ApiField.VALUE: True,
            }
        ],
    }
    if filters is not None:
        data[ApiField.FILTER].extend(filters)
    tasks = []
    if project_ids is not None:
        for batch in batched(project_ids):
            data[ApiField.FILTER] = [
                {
                    ApiField.FIELD: ApiField.ID,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: batch,
                },
                {
                    ApiField.FIELD: ApiField.EMBEDDINGS_ENABLED,
                    ApiField.OPERATOR: "=",
                    ApiField.VALUE: True,
                },
            ]
            tasks.append(
                get_list_all_pages_async(
                    api,
                    method,
                    data=data,
                    convert_json_info_cb=convert_json_info,
                    progress_cb=None,
                    limit=None,
                    return_first_response=False,
                )
            )
    else:
        tasks.append(
            get_list_all_pages_async(
                api,
                method,
                data=data,
                convert_json_info_cb=convert_json_info,
                progress_cb=None,
                limit=None,
                return_first_response=False,
            )
        )
    results = []
    for task in asyncio.as_completed(tasks):
        results.extend(await task)
    return results


@to_thread
@timeit
def set_update_flag(api: sly.Api, project_id: int):
    custom_data = api.project.get_custom_data(project_id)
    custom_data[CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT] = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    api.project.update_custom_data(project_id, custom_data, silent=True)


@to_thread
@timeit
def get_update_flag(api: sly.Api, project_id: int) -> Optional[dict]:
    custom_data = api.project.get_custom_data(project_id)
    info = {
        CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT: None,
    }
    if CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT in custom_data:
        info[CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT] = custom_data[
            CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT
        ]
    return info


@to_thread
@timeit
def clear_update_flag(api: sly.Api, project_id: int):
    custom_data = api.project.get_custom_data(project_id)
    if custom_data is None or custom_data == {}:
        return
    if CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT in custom_data:
        del custom_data[CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT]
        api.project.update_custom_data(project_id, custom_data, silent=True)


@to_thread
@timeit
def is_task_running(api: sly.Api, task_id: int) -> Optional[bool]:
    """Check if the task is currently running by its ID.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param task_id: ID of the task to check status.
    :type task_id: int
    :return: Dictionary with task status or None if task not found.
    :rtype: Optional[Dict]
    """

    try:
        status = api.task.is_running(task_id)
    except Exception as e:
        sly.logger.error("Error checking task status: %s", e, exc_info=True)
        status = None
    return status


def get_app_host(api: sly.Api, slug: str) -> str:
    """Get the app host URL from the Supervisely API.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :return: The app host URL.
    :rtype: str
    """
    session_token = api.app.get_session_token(slug)
    sly.logger.debug("Session token for CLIP slug %s: %s", slug, session_token)
    host = api.server_address.rstrip("/") + "/net/" + session_token
    sly.logger.debug("App host URL for CLIP: %s", host)
    return host


@to_thread
@timeit
def get_project_inprogress_status(endpoint: str, id: int) -> dict:
    """Get the project in-progress status by ID by sending a request to the service endpoint.

    :param endpoint: Endpoint URL of the service.
    :type endpoint: str
    :param id: ID of the project.
    :type id: int
    :return: Dictionary with project in-progress status.
    :rtype: dict
    """

    response = httpx.post(endpoint, json={"project_id": id})
    return response.json()


@to_thread
@timeit
def check_generator_is_ready(endpoint: str, timeout: int = 10) -> bool:
    """Check if the generator service is ready by sending a request to the service endpoint.

    :param endpoint: Endpoint URL of the generator service.
    :type endpoint: str
    :param timeout: Timeout for the request in seconds. Default is 10 seconds.
    :type timeout: int
    :return: True if the generator service is ready, False otherwise.
    :rtype: bool
    """
    try:
        response = httpx.get(
            f"{endpoint.rstrip('/')}/is_ready", timeout=timeout, follow_redirects=True
        )
        response.raise_for_status()
        status = response.json().get("status", "")
        sly.logger.debug("Received response from Generator service: %s", status)
        if status == "ready":
            sly.logger.debug("Generator service is ready.")
            return True
    except httpx.ConnectError as e:
        sly.logger.warning("Cannot connect to generator service at %s: %s", endpoint, e)
    except httpx.TimeoutException as e:
        sly.logger.warning("Timeout connecting to generator service at %s: %s", endpoint, e)
    except httpx.HTTPStatusError as e:
        sly.logger.warning("HTTP error from generator service: %s", e)
    except Exception as e:
        sly.logger.error(
            "Unexpected error checking generator service readiness: %s", e, exc_info=True
        )

    sly.logger.debug("Generator service is not ready or unreachable.")
    return False
