from dataclasses import dataclass
from typing import List

from supervisely import Api, logger
from supervisely.api.module_api import ApiField

from src.utils import to_thread


@dataclass
class AutoRestartInfo:
    deploy_params: dict

    class Fields:
        AUTO_RESTART_INFO = "autoRestartInfo"
        DEPLOY_PARAMS = "deployParams"

    def generate_fields(self) -> List[dict]:
        return [
            {
                ApiField.FIELD: self.Fields.AUTO_RESTART_INFO,
                ApiField.PAYLOAD: {self.Fields.DEPLOY_PARAMS: self.deploy_params},
            }
        ]

    @classmethod
    def from_response(cls, data: dict):
        autorestart_info = data.get(cls.Fields.AUTO_RESTART_INFO, None)
        if autorestart_info is None:
            return None
        return cls(deploy_params=autorestart_info.get(cls.Fields.DEPLOY_PARAMS, None))

    def is_changed(self, deploy_params: dict) -> bool:
        return self.deploy_params != deploy_params

    @staticmethod
    def check_autorestart(api: Api) -> "AutoRestartInfo":
        autorestart = None
        try:
            if api.task_id is not None:
                logger.debug("Checking autorestart info...")
                response = api.task.get_fields(
                    api.task_id, [AutoRestartInfo.Fields.AUTO_RESTART_INFO]
                )
                autorestart = AutoRestartInfo.from_response(response)
                if autorestart is not None:
                    logger.debug("Autorestart info found.")
                else:
                    logger.debug("Autorestart info is not set.")
        except Exception:
            logger.error("Autorestart info is not available.", exc_info=True)
        return autorestart

    @staticmethod
    @to_thread
    def set_autorestart_params(api: Api, project_id: int):
        try:
            deploy_params = {"project_id": project_id}
            logger.debug("Setting autorestart params...", extra=deploy_params)
            api.task.set_fields(
                api.task_id,
                [
                    {
                        ApiField.FIELD: AutoRestartInfo.Fields.AUTO_RESTART_INFO,
                        ApiField.PAYLOAD: {AutoRestartInfo.Fields.DEPLOY_PARAMS: deploy_params},
                    }
                ],
            )
            logger.info("Autorestart params set successfully.")
        except Exception as e:
            logger.error(f"Failed to set autorestart params: {e}", exc_info=True)

    @staticmethod
    @to_thread
    def clear_autorestart_params(api: Api):
        try:
            logger.debug("Clearing autorestart params...")
            api.task.set_fields(
                api.task_id,
                [
                    {
                        ApiField.FIELD: AutoRestartInfo.Fields.AUTO_RESTART_INFO,
                        ApiField.PAYLOAD: None,
                    }
                ],
            )
            logger.info("Autorestart params cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear autorestart params: {e}", exc_info=True)
