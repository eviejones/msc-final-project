import logging
import os
import zipfile
import pandas as pd
import geopandas as gpd
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WFP Ingest")
logger.setLevel(logging.INFO)

class HdxClient:
    def __init__(self):
        self.data = Dataset
        if not Configuration._configuration:
            Configuration.create(
                hdx_site="prod", user_agent="msc-project", hdx_read_only=True
            )

    def _load_file(self, file_path):
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        return gpd.read_file(file_path)

    def get_data(self, dataset_name, file_name, file_type, download=True):
        download_dir = "../data/hdx"
        file_path = os.path.join(download_dir, f"{file_name}.{file_type}")

        # Read local file if download is false
        if not download:
            logger.info(f"download=False: Reading local file {file_path}")
            if os.path.exists(file_path):
                return (
                    pd.read_csv(file_path)
                    if file_type.lower() == "csv"
                    else gpd.read_file(file_path)
                )
            logger.error(f"Local file not found at {file_path}")
            return None

        # Download data from hdx if download is true
        dataset = self.data.read_from_hdx(dataset_name)
        if not dataset:
            return None

        resources = dataset.get_resources()
        target_resource = next(
            (r for r in resources if r["format"].lower() == file_type.lower()), None
        )

        if target_resource:
            os.makedirs(download_dir, exist_ok=True)
            _, path = target_resource.download(folder=download_dir)

            # Unzip if zipped
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(download_dir)
                    path = file_path

            return (
                pd.read_csv(path)
                if file_type.lower() == "csv"
                else gpd.read_file(path)
            )

        return None