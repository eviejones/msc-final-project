import logging
import os
import zipfile
import pandas as pd
import geopandas as gpd
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Hdx Ingest")
logger.setLevel(logging.INFO)


class HdxClient:
    def __init__(self):
        self.data = Dataset
        if not Configuration._configuration:
            Configuration.create(
                hdx_site="prod", user_agent="msc-project", hdx_read_only=True
            )

    def get_admin_boundaries(self, dataset_name, file_name, file_type, download=True):
        download_dir = "../data/hdx"

        if not download:
            file_path = next(
                (
                    os.path.join(download_dir, f)
                    for f in os.listdir(download_dir)
                    if file_name.lower() in f.lower()
                    and f.endswith(f".{file_type.lower()}")
                ),
                None,
            )

            if file_path:
                logger.info(f"download=False: Reading local file {file_path}")
                return gpd.read_file(file_path)

            logger.error(f"Local file matching '{file_name}.{file_type}' not found.")
            return None

        dataset = self.data.read_from_hdx(dataset_name)
        if dataset is None:
            logger.error("Dataset not found.")
            return None

        resources = dataset.get_resources()
        target_resource = next(
            (r for r in resources if file_type.lower() in r["format"].lower()), None
        )

        if target_resource:
            os.makedirs(download_dir, exist_ok=True)
            _, path = target_resource.download(folder=download_dir)
            logger.info(f"Downloaded file to: {path}")

            if zipfile.is_zipfile(path):
                logger.info("Extracting zip archive...")
                with zipfile.ZipFile(path, "r") as zip_ref:
                    extract_dir = os.path.dirname(path)
                    zip_ref.extractall(extract_dir)

                    target_file = next(
                        (
                            os.path.join(extract_dir, f)
                            for f in zip_ref.namelist()
                            if file_name.lower() in f.lower()
                            and f.endswith(f".{file_type.lower()}")
                        ),
                        None,
                    )

                    if target_file:
                        path = target_file
                    else:
                        logger.error(
                            f"Could not find '{file_name}' with extension '.{file_type}' inside the zip."
                        )
                        return None

            logger.info(f"Successfully loaded '{file_name}' into dataframe")
            return gpd.read_file(path)

        logger.error("No suitable vector/boundary format found in resources.")
        return None

    def get_data(self, dataset_name, file_name, file_type, download=True):
        download_dir = "../data/hdx"
        file_path = os.path.join(download_dir, f"{file_name}.{file_type}")

        # Read local file if download is false
        if not download:
            logger.info(f"download=False: Reading local file {file_path}")
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            logger.error(f"Local file not found at {file_path}")
            return None

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
