import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api.from_env(ignore_task_id=True)
sly.logger.debug("Connected to Supervisely API: %s", api.server_address)
api.file.load_dotenv_from_teamfiles(override=True)

# region envvars
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
sly.logger.debug("Team ID: %s, Workspace ID: %s", team_id, workspace_id)

# The first option (modal.state) comes from the Modal window, the second one (QDRANT_HOST)
# comes from the .env file.
qdrant_host = os.getenv("modal.state.qdrantHost") or os.getenv("QDRANT_HOST")
cas_host = os.getenv("modal.state.casHost") or os.getenv("CAS_HOST")

update_interval = int(os.getenv("modal.state.update_interval") or os.getenv("UPDATE_INTERVAL"))
try:
    cas_host = int(cas_host)
    task_info = api.task.get_info_by_id(cas_host)
    try:
        cas_host = api.server_address + task_info["settings"]["message"]["appInfo"]["baseUrl"]
    except KeyError:
        sly.logger.warning("Cannot get CAS URL from task settings")
        raise RuntimeError("Cannot connect to CLIP Service")
except ValueError:
    if cas_host[:4] not in ["http", "ws:/", "grpc"]:
        cas_host = "grpc://" + cas_host
# endregion

if not qdrant_host:
    raise ValueError("QDRANT_HOST is not set in the environment variables")
if not cas_host:
    raise ValueError("CAS_HOST is not set in the environment variables")

sly.logger.info("Qdrant host: %s", qdrant_host)
sly.logger.info("CAS host: %s", cas_host)

# region constants
IMAGE_SIZE_FOR_CAS = 224
UPDATE_EMBEDDINGS_INTERVAL = update_interval  # minutes, default is 10
# endregion

sly.logger.debug("Image size for CAS: %s", IMAGE_SIZE_FOR_CAS)
sly.logger.debug("Update embeddings interval: %s", UPDATE_EMBEDDINGS_INTERVAL)

background_tasks = {}
