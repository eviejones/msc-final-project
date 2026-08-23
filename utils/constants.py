"""The main constants required for setting constants used throughout the project."""

from pathlib import Path

import pandas as pd

# ---- Non-country specific constants
FORCE_DOWNLOAD = False  # Whether to force download of data, even if it already exists locally. Do not do this unless you want it to overwrite existing saved data.

REPORTS_DIR = "evaluation/model_reports"  # Folder to save the reports in


# ---- Country specific constants

# Sudan

COUNTRY = "Sudan"  # Country to collect data for

PRIMARY_COMMODITIES = [
    "Sorghum",
    "Millet",
    "Wheat flour",
]  # List of primary food commodities to focus on for analysis, this would be dependent on a country

TRAIN_START_DATE = pd.to_datetime("2018-01-01")  # Must be in pandas datatime format
TRAIN_END_DATE = pd.to_datetime("2022-12-31")

ONSET_START_DATE = pd.to_datetime("2023-01-01")
ONSET_END_DATE = pd.to_datetime("2023-12-31")

ACTIVE_START_DATE = pd.to_datetime("2024-01-01")
ACTIVE_END_DATE = pd.to_datetime("2025-12-31")

# Ethiopia

# COUNTRY = "Ethiopia"

# PRIMARY_COMMODITIES = [
#     "Sorghum",
#     "Teff",
#     "Maize",
# ]  # List of primary food commodities to focus on for analysis, this would be dependent on a country

# TRAIN_START_DATE = pd.to_datetime("2015-07-01")
# TRAIN_END_DATE   = pd.to_datetime("2020-06-30")
# ONSET_START_DATE = pd.to_datetime("2020-07-01")
# ONSET_END_DATE   = pd.to_datetime("2021-06-30")
# ACTIVE_START_DATE = pd.to_datetime("2021-07-01")
# ACTIVE_END_DATE  = pd.to_datetime("2022-06-30")
