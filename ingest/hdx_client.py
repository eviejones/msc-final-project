from hdx.utilities.easy_logging import setup_logging
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset
import geopandas as gpd
import zipfile
import os

setup_logging()

class HdxClient:
    def __init__(self):
        self.data = Dataset
        if not Configuration._configuration:
            Configuration.create(hdx_site='prod', user_agent='msc-project', hdx_read_only=True)

    def get_data(self, dataset_name, file_name, file_type):
        dataset = self.data.read_from_hdx(dataset_name)
        if dataset is None:
            print("Dataset not found.")
            return None

        resources = dataset.get_resources()

        target_resource = None
        for resource in resources:
            if resource['format'].lower() == file_type.lower():
                target_resource = resource
                break

        if target_resource:
            download_dir = "../data/hdx"
            os.makedirs(download_dir, exist_ok=True)

            # Download the file to your local environment
            url, path = target_resource.download(folder=download_dir)
            print(f"Downloaded file to: {path}")

            # Unzip and match the specific file_name (e.g., 'adm1')
            if zipfile.is_zipfile(path):
                print("Extracting zip archive...")
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    extract_dir = os.path.dirname(path)
                    zip_ref.extractall(extract_dir)

                    extracted_files = zip_ref.namelist()

                    # Look for a file containing both the target file_name and a valid spatial extension
                    target_file = next(
                        (os.path.join(extract_dir, f) for f in extracted_files
                         if file_name.lower() in f.lower() and f.endswith(('.geojson', '.shp', '.gpkg'))),
                        None
                    )

                    if target_file:
                        path = target_file
                    else:
                        print(f"Could not find a file matching '{file_name}' with a supported spatial format inside the zip.")
                        return None

            # Load the specific file into a GeoPandas GeoDataFrame
            gdf = gpd.read_file(path) # TODO could also be normal df

            print(f"Successfully loaded '{file_name}' into variable dataframe")
            return gdf
        else:
            print("No suitable vector/boundary format found in resources.")
            return None
