import difflib

import pandas as pd
import pycountry

from ingest.hdx_client import HdxClient
from utils.constants import COUNTRY
from utils.logger import get_logger

logger = get_logger("Name mapping")

FUZZY_MATCH_CUTOFF = 0.8

# As Sudan has been the focus of this project, the region name differences have been manually added. For other countries, fuzzy matching can be used.
REGION_NAME_OVERRIDES = {
    "Sudan": {
        "Al Gezira": "Al Jazirah",
        "Nile": "River Nile",
        "Eastern Darfur": "East Darfur",
        "Abyei PCA": "Abyei",
        "Aj Jazirah": "Al Jazirah",
    },
}


def clean_region_names(value: str, canonical_names=None) -> str:
    """
    Maps a raw admin1 name to its canonical form for the current COUNTRY.

    Checks the manual override table for COUNTRY first. If there's no override and a list of
    canonical names is supplied, falls back to fuzzy string matching against it.
    """
    if pd.isna(value):
        return value

    overrides = REGION_NAME_OVERRIDES.get(COUNTRY, {})
    if value in overrides:
        mapped = overrides[value]
        logger.info(f"Renamed region '{value}' to '{mapped}' (manual override)")
        return mapped

    if canonical_names is None or value in canonical_names:
        return value

    close_matches = difflib.get_close_matches(
        value, canonical_names, n=1, cutoff=FUZZY_MATCH_CUTOFF
    )
    if close_matches:
        mapped = close_matches[0]
        logger.info(f"Renamed region '{value}' to '{mapped}' (fuzzy match)")
        return mapped

    logger.warning(f"No mapping found for region '{value}'; leaving unchanged")
    return value


def build_region_name_map(values, canonical_names=None) -> dict:
    """Resolves each distinct raw region name once, so renames are logged once."""
    return {
        value: clean_region_names(value, canonical_names)
        for value in pd.unique(values)
        if pd.notna(value)
    }


def get_iso3(country: str) -> str:
    """Resolves a country name to its three letter iso code."""
    return pycountry.countries.lookup(country).alpha_3


def pcode_mapping(country: str, admin_level: int = 1) -> dict:
    """Fetches the admin boundaries for a country to map the pcode names from the rainfall data set to clean region names."""
    logger.info(
        "Creating a PCODE map for rainfall data. This may take a few seconds..."
    )
    iso3 = get_iso3(country).lower()

    hdx = HdxClient()
    boundaries = hdx.get_admin_boundaries(
        dataset_name=f"cod-ab-{iso3}",
        file_name=f"{iso3}_admin{admin_level}",
        file_type="geojson",
    )
    if boundaries is None:
        raise ValueError(f"Could not fetch admin{admin_level} boundaries for {country}")

    pcode_col = "adm1_pcode"  # Manually setting this to admin level 1 for now
    name_col = "adm1_name"
    return dict(zip(boundaries[pcode_col], boundaries[name_col]))
