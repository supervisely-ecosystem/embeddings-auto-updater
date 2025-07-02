import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from supervisely import logger

from src.utils import to_thread

FILE_PATH: str = "/state/current_project.json"
PROJECT_KEY = "project_id"


@dataclass
class AutoRestartInfo:
    deploy_params: dict

    @classmethod
    def from_file(cls, file_path: str = FILE_PATH) -> Optional["AutoRestartInfo"]:
        """Load project info from local file"""
        try:
            if not os.path.exists(file_path):
                logger.debug(f"Autorestart file {file_path} does not exist.")
                return None

            with open(file_path, "r") as f:
                data = json.load(f)

            project_id = data.get(PROJECT_KEY)
            if project_id is None:
                logger.debug("Project ID not found in autorestart file.")
                return None

            deploy_params = {PROJECT_KEY: project_id}
            logger.debug("Autorestart info loaded from file.", extra=deploy_params)
            return cls(deploy_params=deploy_params)

        except Exception as e:
            logger.error(f"Failed to load autorestart info from file: {e}", exc_info=True)
            return None

    def is_changed(self, deploy_params: dict) -> bool:
        return self.deploy_params != deploy_params

    @staticmethod
    def check_autorestart(file_path: str = FILE_PATH) -> Optional["AutoRestartInfo"]:
        """Check autorestart info from local file"""
        logger.debug("Checking autorestart info from file...")
        autorestart = AutoRestartInfo.from_file(file_path)
        if autorestart is not None:
            logger.debug("Autorestart info found in file.")
        else:
            logger.debug("Autorestart info is not set in file.")
        return autorestart

    @staticmethod
    @to_thread
    def set_autorestart_params(project_id: int, file_path: str = FILE_PATH):
        """Set autorestart params to local file"""
        try:
            deploy_params = {PROJECT_KEY: project_id}
            logger.debug("Setting autorestart params to file...", extra=deploy_params)

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Write project info to file
            with open(file_path, "w") as f:
                json.dump({PROJECT_KEY: project_id}, f, indent=2)

            logger.info("Autorestart params set successfully to file.")
        except Exception as e:
            logger.error(f"Failed to set autorestart params to file: {e}", exc_info=True)

    @staticmethod
    @to_thread
    def clear_autorestart_params(file_path: str = FILE_PATH):
        """Clear autorestart params from local file"""
        try:
            logger.debug("Clearing autorestart params from file...")

            if os.path.exists(file_path):
                with open(file_path, "w") as f:
                    json.dump({}, f, indent=2)
                logger.info("Autorestart params cleared successfully from file.")
            else:
                logger.debug("Autorestart file does not exist, nothing to clear.")

        except Exception as e:
            logger.error(f"Failed to clear autorestart params from file: {e}", exc_info=True)
