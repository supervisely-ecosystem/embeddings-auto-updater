import random
from typing import List, Optional, Tuple

import supervisely as sly
from supervisely.sly_logger import logger

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.autorestart import PROJECT_KEY, AutoRestartInfo
from src.utils import (
    ApiField,
    CustomDataFields,
    ResponseFields,
    ResponseStatus,
    clear_update_flag,
    create_lite_image_infos,
    datetime,
    fix_vectors,
    get_all_projects,
    get_project_info,
    get_project_inprogress_status,
    get_team_info,
    get_update_flag,
    image_get_list_async,
    parse_timestamp,
    set_embeddings_in_progress,
    set_image_embeddings_updated_at,
    set_project_embeddings_updated_at,
    set_update_flag,
    stop_running_in_progress_task,
    timeit,
    timezone,
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

    msg_prefix = f"[Project: {project_id}]"
    vectors = []

    if len(to_create) == 0 and len(to_delete) == 0:
        logger.debug(f"{msg_prefix} Nothing to update.")
        return to_create, vectors

    to_create = await create_lite_image_infos(
        cas_size=g.IMAGE_SIZE_FOR_CLIP,
        image_infos=to_create,
    )

    # if await qdrant.collection_exists(project_id):
    # Get diff of image infos, check if they are already in the Qdrant collection

    if check_collection_exists:
        await qdrant.get_or_create_collection(project_id)

    current_progress = 0
    total_progress = len(to_create)

    if len(to_create) > 0:
        logger.debug(f"{msg_prefix} Images to be vectorized: {total_progress}.")
        for image_batch in sly.batched(to_create):
            # Get vectors from images.
            vectors_batch = await cas.get_vectors(
                [image_info.cas_url for image_info in image_batch]
            )
            vectors_batch = fix_vectors(vectors_batch)
            logger.debug(f"{msg_prefix} Got {len(vectors_batch)} vectors for images.")

            # Upsert vectors to Qdrant.
            await qdrant.upsert(project_id, vectors_batch, image_batch)
            current_progress += len(image_batch)
            logger.debug(
                f"{msg_prefix} Upserted {len(vectors_batch)} vectors to Qdrant. [{current_progress}/{total_progress}]",
            )
            await set_image_embeddings_updated_at(api, image_batch)

            if return_vectors:
                vectors.extend(vectors_batch)

        logger.debug(f"{msg_prefix} All {total_progress} images have been vectorized.")

    if len(to_delete) > 0:
        logger.debug(f"{msg_prefix} Vectors for images to be deleted: {len(to_delete)}.")
        for image_batch in sly.batched(to_delete):
            # Delete images from the Qdrant.
            await qdrant.delete_collection_items(
                collection_name=project_id, image_infos=image_batch
            )
            await set_image_embeddings_updated_at(api, image_batch, [None] * len(image_batch))
            logger.debug(f"{msg_prefix} Deleted {len(image_batch)} images from Qdrant.")
    logger.info(f"{msg_prefix} Embeddings Created: {len(to_create)}, Deleted: {len(to_delete)}.")
    return to_create, vectors


@timeit
async def update_embeddings(
    api: sly.Api,
    project_id: int,
    force: bool = False,
    project_info: Optional[sly.ProjectInfo] = None,
    skip_in_progress_check: bool = False,
):
    msg_prefix = f"[Project: {project_id}]"

    if project_info is None:
        project_info = await get_project_info(api, project_id)

    if project_info.embeddings_in_progress is True and not skip_in_progress_check:
        logger.info(f"{msg_prefix} Embeddings update is already in progress. Skipping.")
        return
    # if project_info.embeddings_updated_at is not None and parse_timestamp(
    #     project_info.embeddings_updated_at
    # ) > parse_timestamp(project_info.updated_at):
    #     logger.info(f"{msg_prefix} Is not updated since last embeddings update. Skipping.")
    #     return
    await set_embeddings_in_progress(api, project_id, True)
    await set_update_flag(api, project_id)
    g.current_task = project_id
    await AutoRestartInfo.set_autorestart_params(project_id)
    try:
        if force:
            logger.info(f"{msg_prefix} Force enabled, recreating embeddings for all images.")
            await qdrant.delete_collection(project_id)
            # do not need to create collection here, it will be created in process_images
            images_to_create = await image_get_list_async(api, project_id)
            images_to_delete = []
        elif project_info.embeddings_updated_at is None:
            # do not need to check or create collection here, it will be created in process_images
            logger.info(
                f"{msg_prefix} Embeddings have not been created yet. Will be created for all required images."
            )
            images_to_create = await image_get_list_async(api, project_id, wo_embeddings=True)
            images_to_delete = []
        else:
            logger.info(f"{msg_prefix} Checking for images that need embeddings to be created.")
            images_to_create = await image_get_list_async(api, project_id, wo_embeddings=True)
            images_to_delete = await image_get_list_async(
                api, project_id, deleted_after=project_info.embeddings_updated_at
            )

        if len(images_to_create) == 0 and len(images_to_delete) == 0:
            logger.info(f"{msg_prefix} Nothing to update.")
            await set_embeddings_in_progress(api, project_id, False)
            await clear_update_flag(api, project_id)
            return

        await process_images(api, project_id, images_to_create, images_to_delete)
        if len(images_to_create) > 0 or len(images_to_delete) > 0:
            await set_project_embeddings_updated_at(api, project_id)
    except Exception as e:
        logger.error(
            f"{msg_prefix} Error during embeddings update: {e}",
            exc_info=True,
            extra={"project_id": project_id},
        )
    finally:
        g.current_task = None
        await set_embeddings_in_progress(api, project_id, False)
        await clear_update_flag(api, project_id)


@timeit
async def auto_update_embeddings(
    api: sly.Api, project_id: int, project_info: Optional[sly.ProjectInfo] = None
):
    """
    Update embeddings for the specified project if needed.
    """
    msg_prefix = f"[Project: {project_id}]"

    if project_info is None:
        project_info = await get_project_info(api, project_id)

    if project_info is None:
        logger.warning(f"{msg_prefix} Info not found. Skipping.")
        return

    team_info: sly.TeamInfo = await get_team_info(api, project_info.team_id)
    if team_info.usage is not None and team_info.usage.plan == "free":
        logger.info(
            f"{msg_prefix} Embeddings update is not available on 'free' plan.",
            extra={
                "project_id": project_id,
                "project_name": project_info.name,
                "team_id": team_info.id,
                "team_name": team_info.name,
            },
        )
        api.project.disable_embeddings(project_id)
        logger.info(
            f"{msg_prefix} AI Search are disabled due to plan limitations.",
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
        logger.info(f"{msg_prefix} AI Search is not activated. Skipping.", extra=log_extra)
        return
    logger.info(f"{msg_prefix} Auto update embeddings started.", extra=log_extra)
    await update_embeddings(api, project_id, force=False, project_info=project_info)
    logger.info(f"{msg_prefix} Auto update embeddings finished.")


@timeit
async def auto_update_all_embeddings():
    """Update embeddings for all available projects"""
    logger.info("[All Projects] Auto update task started.")

    try:
        project_infos: List[sly.ProjectInfo] = await get_all_projects(g.api)
        random.shuffle(project_infos)
        for project_info in project_infos:
            try:
                await auto_update_embeddings(g.api, project_info.id)
            except Exception as e:
                logger.error(
                    f"[Project: {project_info.id}] Error updating embeddings : {e}",
                    exc_info=True,
                )
        await AutoRestartInfo.clear_autorestart_params()
    except Exception as e:
        logger.error(f"Error in auto_update_all_embeddings: {e}", exc_info=True)

    logger.info("[All Projects] Auto update task finished.")


@timeit
async def check_in_progress_projects():
    """
    Check if there are any projects that are stuck in progress.
    """
    filters = [
        {
            ApiField.FIELD: ApiField.EMBEDDINGS_IN_PROGRESS,
            ApiField.OPERATOR: "=",
            ApiField.VALUE: True,
        }
    ]

    logger.info("[All Projects] Check in progress task started.")
    project_infos: List[sly.ProjectInfo] = await get_all_projects(g.api, filters=filters)
    for project_info in project_infos:
        should_clear_in_progress = False
        msg_prefix = f"[Project: {project_info.id}]"

        if project_info.id == g.current_task:
            logger.debug(f"{msg_prefix} Is in progress now with auto update task. Skipping...")
            continue

        info = await get_update_flag(g.api, project_info.id)
        if info[CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT] is not None:
            now = parse_timestamp(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            update = parse_timestamp(info[CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT])
            hours_diff = (now - update).total_seconds() / 3600
            logger.debug(
                f"{msg_prefix} Update started at {info[CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT]}, "
                f"hours difference: {hours_diff:.2f}"
            )
            if hours_diff >= g.update_frame:
                try:
                    response = await get_project_inprogress_status(project_info.id)
                    cur_status = response.get(ResponseFields.STATUS)
                    is_running = response.get(ResponseFields.IS_RUNNING, False)
                    message = response.get(ResponseFields.MESSAGE, "")

                    # Clear in progress flag for all statuses except RUNNING
                    # Possible statuses from endpoint: NO_TASK, CANCELLED, COMPLETED, FAILED, RUNNING
                    if (
                        cur_status
                        in [
                            ResponseStatus.NO_TASK,
                            ResponseStatus.CANCELLED,
                            ResponseStatus.COMPLETED,
                            ResponseStatus.FAILED,
                        ]
                        or not is_running
                    ):
                        logger.info(
                            f"{msg_prefix} Embeddings creation task status: {cur_status}",
                            extra={
                                "is_running": is_running,
                                "message": message
                            }
                        )
                        should_clear_in_progress = True
                    elif cur_status == ResponseStatus.RUNNING or is_running:
                        logger.debug(
                            f"{msg_prefix} Embeddings creation task status: {cur_status}",
                            extra={
                                "is_running": is_running,
                                "message": message
                            }
                        )
                    else:
                        # Fallback: if status is unknown, log warning but clear the flag to avoid stuck projects
                        logger.warning(
                            f"{msg_prefix} Unknown task status: {cur_status}",
                            extra={
                                "is_running": is_running,
                                "message": message
                            }
                        )
                        should_clear_in_progress = True
                except Exception as e:
                    logger.error(
                        f"{msg_prefix} Error checking task status via endpoint: {e}. ",
                        exc_info=True,
                    )
                    should_clear_in_progress = True

        if should_clear_in_progress:
            try:
                logger.info(f"{msg_prefix} Stopping task and resetting in progress state.")
                await stop_running_in_progress_task(project_info.id)
                await clear_update_flag(g.api, project_info.id)
                await set_embeddings_in_progress(g.api, project_info.id, False)
            except Exception as e:
                logger.error(
                    f"{msg_prefix} Error stopping in progress task: {e}.",
                    exc_info=True,
                )

    logger.info("[All Projects] Check in progress task finished.")


@timeit
async def continue_project_processing(project_id: int):
    """
    Check if the project processing should continue based on the update flag.
    :param project_id: Project ID to check.
    :return: True if the project processing should continue, False otherwise.
    """
    msg_prefix = f"[Project: {project_id}]"

    info = await get_project_info(g.api, project_id)
    if info is None:
        return

    await clear_update_flag(g.api, project_id)
    await set_embeddings_in_progress(g.api, project_id, False)
    logger.info(f"{msg_prefix} Continue updating embeddings after App restart.")
    await update_embeddings(g.api, project_id, project_info=info, skip_in_progress_check=True)
    logger.info(f"{msg_prefix} Update embeddings finished.")


async def safe_check_autorestart():
    """Safely checks for autorestart information and processes it if available.
    This function is designed to handle exceptions gracefully and log any issues that arise during the check.
    It ensures that the autorestart check does not disrupt the normal operation of the application.
    """
    try:
        autorestart = AutoRestartInfo.check_autorestart()
        sly.logger.debug("Autorestart info checked")
        if autorestart is not None:
            project_id = autorestart.deploy_params.get(PROJECT_KEY, None)
            if project_id is not None:
                try:
                    sly.logger.info(
                        "Autorestart detected, applying deploy params: ",
                        extra=autorestart.deploy_params,
                    )
                    await continue_project_processing(project_id)
                except Exception as e:
                    sly.logger.warning(
                        "Autorestart failed. Runnning app in normal mode.", exc_info=True
                    )
    except Exception as e:
        sly.logger.error("Error in autorestart check", exc_info=True)
