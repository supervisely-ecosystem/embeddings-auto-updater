import os

import supervisely as sly
from dotenv import load_dotenv

from src.utils import get_app_host

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

internal_address = os.getenv("SUPERVISELY_API_SERVER_ADDRESS", None)
sly.logger.debug("Internal Supervisely API server address: %s", internal_address)
api_token = os.getenv("API_TOKEN", None)
sly.logger.debug("API token from environment: %s", api_token)
if (internal_address is not None and internal_address != "") and (api_token is None or api_token == ""):
    temp_api = sly.Api(ignore_task_id=True)
    response = temp_api.post(
        "instance.admin-info", data={"accessToken": "SUPERVISELY_API_ACCESS_TOKEN"}
    )
    token = response.json()["id"]
    sly.logger.debug("Using Supervisely API token: %s", token)
elif api_token is not None and api_token != "":
    token = api_token
    sly.logger.debug("Using Supervisely API token from environment: %s", token)
else:
    token = None
    sly.logger.debug("No internal Supervisely API server address found, using public API")

api = sly.Api(ignore_task_id=True, token=token)
sly.logger.debug("Connected to Supervisely API: %s", api.server_address)
api.file.load_dotenv_from_teamfiles(override=True)
clip_slug = "supervisely-ecosystem/deploy-clip-as-service"

# region envvars
generator_host = os.getenv("modal.state.generatorHost") or os.getenv("GENERATOR_HOST")
qdrant_host = os.getenv("modal.state.qdrantHost") or os.getenv("QDRANT_HOST")
clip_host = os.getenv("modal.state.clipHost", None) or os.getenv("CLIP_HOST", None)

sly.logger.debug("CLIP host from environment: %s", clip_host)
if clip_host is None or "":
    clip_host = get_app_host(api, clip_slug)

try:
    clip_host = int(clip_host)
    task_info = api.task.get_info_by_id(clip_host)
    try:
        clip_host = api.server_address + task_info["settings"]["message"]["appInfo"]["baseUrl"]
    except KeyError:
        sly.logger.warning("Cannot get CLIP URL from task settings")
        raise RuntimeError("Cannot connect to CLIP Service")
except ValueError:
    if clip_host[:4] not in ["http", "ws:/", "grpc"]:
        clip_host = "grpc://" + clip_host

update_interval = int(os.getenv("modal.state.updateInterval") or os.getenv("UPDATE_INTERVAL"))
update_frame = int(os.getenv("modal.state.updateFrame") or os.getenv("UPDATE_FRAME"))
# endregion

if not qdrant_host:
    raise ValueError("QDRANT_HOST is not set in the environment variables")
if not clip_host:
    raise ValueError("CLIP_HOST is not set in the environment variables")
if not generator_host:
    raise ValueError("GENERATOR_HOST is not set in the environment variables")

sly.logger.info("Qdrant host: %s", qdrant_host)
sly.logger.info("CLIP host: %s", clip_host)
sly.logger.info("Embeddings Generator host: %s", generator_host)

# region constants
IMAGE_SIZE_FOR_CLIP = 224
UPDATE_EMBEDDINGS_INTERVAL = update_interval  # minutes, default is 10
CHECK_INPROGRESS_INTERVAL = update_frame  # hours, default is 12
CHECK_INPROGRESS_STATUS_ENDPOINT = generator_host.rstrip("/") + "/check_background_task_status"
# endregion

sly.logger.debug("Image size for CLIP: %s", IMAGE_SIZE_FOR_CLIP)
sly.logger.debug("Update interval in minutes: %s", UPDATE_EMBEDDINGS_INTERVAL)
sly.logger.debug("Update frame in hours: %s", update_frame)

current_task = None  # project_id which is currently processed creating embeddings
