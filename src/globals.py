import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

internal_address = os.getenv("SUPERVISELY_API_SERVER_ADDRESS", None)
sly.logger.debug("Internal Supervisely API server address: %s", internal_address)
if internal_address == "":
    internal_address = None
    del os.environ["SUPERVISELY_API_SERVER_ADDRESS"]
    sly.logger.debug("Removed empty SUPERVISELY_API_SERVER_ADDRESS from environment")

api_token = os.getenv("API_TOKEN", None)
sly.logger.debug("API token from environment: %s", api_token)
if api_token == "":
    api_token = None
    del os.environ["API_TOKEN"]
    sly.logger.debug("Removed empty API_TOKEN from environment")

access_token = os.getenv("SUPERVISELY_API_ACCESS_TOKEN", None)
sly.logger.debug("Access token from environment: %s", access_token)
if access_token == "":
    access_token = None

if access_token is None and internal_address is not None and api_token is None:
    message = (
        f"You are trying to connect to the internal Supervisely API server: {internal_address}. "
        "Access token or API token is required to connect to the internal Supervisely API server, "
        "but it is not provided in the environment variables."
    )
    sly.logger.error(message)
    raise RuntimeError(message)

if internal_address is not None and api_token is None:
    temp_api = sly.Api(ignore_task_id=True)
    response = temp_api.post("instance.admin-info", data={"accessToken": access_token})
    token = response.json()["apiToken"]
    sly.logger.debug("Using Supervisely API token: %s", token)
elif api_token is not None:
    token = api_token
    sly.logger.debug("Using Supervisely API token from environment")
else:
    token = None
    sly.logger.debug("No internal Supervisely API server address found, using public API")

api = sly.Api(ignore_task_id=True, token=token)
sly.logger.debug("Connected to Supervisely API: %s", api.server_address)

# region envvars
generator_host = os.getenv("GENERATOR_HOST")
qdrant_host = os.getenv("QDRANT_HOST")
clip_host = os.getenv("CLIP_HOST", None)
net_server_address = os.getenv("SUPERVISELY_NET_SERVER_ADDRESS", None)

update_interval = os.getenv("UPDATE_INTERVAL")
update_interval = (
    int(update_interval) if update_interval and update_interval != "" else 10
)  # default value in minutes
update_frame = os.getenv("UPDATE_FRAME")
update_frame = (
    float(update_frame) if update_frame and update_frame != "" else 12
)  # default value in hours

# endregion

if not qdrant_host:
    raise ValueError("QDRANT_HOST is not set in the environment variables")

if not generator_host:
    raise ValueError("GENERATOR_HOST is not set in the environment variables")

sly.logger.info("Qdrant host: %s", qdrant_host)
sly.logger.info("Supervisely network server address: %s", net_server_address)
sly.logger.info("Embeddings Generator host: %s", generator_host)
if clip_host is not None and clip_host != "" and net_server_address is not None:
    sly.logger.info(
        "CLIP host is set and will be used instead of Supervisely network server address"
    )

# region constants
IMAGE_SIZE_FOR_CLIP = 224
UPDATE_EMBEDDINGS_INTERVAL = update_interval  # minutes, default is 10
CHECK_INPROGRESS_INTERVAL = int(update_frame * 60)  # minutes, default is 12 hours
CHECK_INPROGRESS_STATUS_ENDPOINT = generator_host.rstrip("/") + "/task_status"
CANCEL_INPROGRESS_TASK_ENDPOINT = generator_host.rstrip("/") + "/cancel_embeddings"
# endregion

sly.logger.debug("Image size for CLIP: %s", IMAGE_SIZE_FOR_CLIP)
sly.logger.debug("Update interval in minutes: %s", UPDATE_EMBEDDINGS_INTERVAL)
sly.logger.debug("Update frame in hours: %s", update_frame)

current_task = None  # project_id which is currently processed creating embeddings
current_task_handle = None  # asyncio.Task handle for current embeddings task
