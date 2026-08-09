import os
import zipfile

import geopandas as gpd
import pandas as pd
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset

from utils.logger import get_logger

logger = get_logger("Train model")


class HdxClient:
    def __init__(self):
        self.data = Dataset
        if not Configuration._configuration:
            Configuration.create(
                hdx_site="prod", user_agent="msc-project", hdx_read_only=True
            )

    def get_data(self, dataset_name, file_name, file_type, force_download=False):
        download_dir = "../data/hdx"
        file_path = os.path.join(download_dir, f"{file_name}.{file_type}")

        if not force_download and os.path.exists(file_path):
            logger.info(f"Reading local file {file_path}")
            return pd.read_csv(file_path)

        dataset = self.data.read_from_hdx(dataset_name)
        if not dataset:
            logger.error("Dataset not found.")
            return None

        resources = dataset.get_resources()
        target_resource = next(
            (r for r in resources if r["format"].lower() == file_type.lower()), None
        )

        if target_resource:
            os.makedirs(download_dir, exist_ok=True)
            _, path = target_resource.download(folder=download_dir)
            logger.info(f"Downloaded file to: {path}")
            return pd.read_csv(path)

        logger.error("No suitable target resource found.")
        return None
