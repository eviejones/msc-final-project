import pycountry

from ingest.hdx_client import HdxClient
from utils.logger import get_logger

logger = get_logger("Name Mapping")

SUDAN_STATE_MAPPING = {
    "Al Gezira": "Al Jazirah",
    "Nile": "River Nile",
    "Eastern Darfur": "East Darfur",
    "Abyei PCA": "Abyei",
    "Aj Jazirah": "Al Jazirah",
}


def clean_state_names(value):
    return SUDAN_STATE_MAPPING.get(value, value) # TODO make this work for other countries


def get_iso3(country: str) -> str:
    """Resolves a country name to its three letter iso code."""
    return pycountry.countries.lookup(country).alpha_3

def pcode_mapping(country: str, admin_level: int = 1) -> dict:
    """Fetches the admin boundaries for a country to map the pcode names from the rainfall data set to clean region names."""
    logger.info("Creating a PCODE map for rainfall data. This may take a few seconds...")
    iso3 = get_iso3(country).lower()

    hdx = HdxClient()
    boundaries = hdx.get_admin_boundaries(
        dataset_name=f"cod-ab-{iso3}",
        file_name=f"{iso3}_admin{admin_level}",
        file_type="geojson",
    )
    if boundaries is None:
        raise ValueError(f"Could not fetch admin{admin_level} boundaries for {country}")

    pcode_col = 'adm1_pcode' # Manually setting this to admin level 1 for now 
    name_col =  'adm1_name'
    return dict(zip(boundaries[pcode_col], boundaries[name_col]))
