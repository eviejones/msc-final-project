import os
import re
import zipfile

import geopandas as gpd
import pandas as pd
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset

from utils.constants import FORCE_DOWNLOAD
from utils.logger import get_logger

logger = get_logger("HDX Client")

class HdxClient:
    def __init__(self):
        self.data = Dataset
        if not Configuration._configuration:
            Configuration.create(
                hdx_site="prod", user_agent="msc-project", hdx_read_only=True
            )

    def _format_file_name(self, file_name: str) -> str:
        """Formats a file name to match the naming convention used across ingest clients."""
        name = file_name.lower().replace("-", "")
        name = re.sub(r"\s+", "_", name.strip())
        return f"hdx_{name}"

    def get_data(self, dataset_name, file_name, file_type):
        download_dir = "data/hdx"
        file_name = self._format_file_name(file_name)
        file_path = os.path.join(download_dir, f"{file_name}.{file_type}")

        if not FORCE_DOWNLOAD and os.path.exists(file_path):
            logger.info(f"Reading local file {file_path}")
            return pd.read_csv(file_path)

        dataset = self.data.read_from_hdx(dataset_name)
        if not dataset:
            logger.error("Dataset not found.")
            return None

        logger.info(f"Fetching data from HDX for dataset: {dataset_name}")
        resources = dataset.get_resources()
        target_resource = next(
            (r for r in resources if r["format"].lower() == file_type.lower()), None
        )

        if target_resource:
            os.makedirs(download_dir, exist_ok=True)
            if os.path.exists(file_path): # Delete existing file if forcee_download
                os.remove(file_path)
            _, path = target_resource.download(folder=download_dir)
            logger.info(f"Downloaded file to: {path}")
            if path != file_path:
                os.replace(path, file_path)
                logger.info(f"Renamed downloaded file to: {file_path}")
            return pd.read_csv(file_path)

        logger.error("No suitable target resource found.")
        return None
