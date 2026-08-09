import pandas as pd

from ingest.hdx_client import HdxClient
from utils.constants import COUNTRY
from utils.dates import TRAIN_START_DATE, WARMUP_START_DATE_1_MONTH, get_padded_index
from utils.logger import get_logger
from utils.name_mapping import clean_state_names, get_iso3, pcode_mapping

logger = get_logger("Rainfall Processing")


def read_rainfall() -> pd.DataFrame:
    """Reads and standardises subnational rainfall data from the HDX client.

    This function retrieves the rainfall dataset, filters it to administrative
    level 1, and maps PCODEs to clean region names. 

    Returns:
        pd.DataFrame: A formatted DataFrame containing the combined rainfall data
        with standardised regions and formatted date periods.
    """
    hdx = HdxClient()

    iso = get_iso3(COUNTRY).lower()
    pcodes = pcode_mapping(COUNTRY)

    rainfall = hdx.get_data(
        dataset_name=f"{iso}-rainfall-subnational",
        file_name=f"{iso}-rainfall-subnat-full",
        file_type="csv",
    )
    rainfall_admin1 = rainfall[rainfall["adm_level"] == 1].copy()
    rainfall_admin1["region"] = (
        rainfall_admin1["PCODE"].map(pcodes).apply(clean_state_names)
    )
    logger.info("Rainfall PCODEs mapped to regions.")

    rainfall_filtered = rainfall_admin1.dropna(subset=["region"])
    rainfall_filtered["date"] = pd.to_datetime(rainfall_filtered["date"])
    rainfall_filtered["year_month"] = rainfall_filtered["date"].dt.to_period("M")

    return rainfall_filtered



def process_rainfall(
    df_rainfall: pd.DataFrame, all_regions, all_months
) -> pd.DataFrame:
    """Processes the raw rainfall dataset to calculate monthly regional anomalies.

    This function filters the dataset within the global start and end dates,
    calculates the median 'r3q' value for each region and month, ensures a
    complete time series index without missing months, and calculates a
    lagged 3-month rainfall anomaly.

    Args:
        df_rainfall (pd.DataFrame): The raw rainfall DataFrame returned by `read_rainfall`.
        all_regions: Array of unique regions for the Cartesian spine.
        all_months: Array of target months for the Cartesian spine.

    Returns:
        pd.DataFrame: A processed DataFrame indexed by region and year_month,
        containing the shifted rainfall anomalies.
    """
    df_grouped = (
        df_rainfall.groupby(["region", "year_month"])["r3q"].median().reset_index()
    )

    df_padded, final_regions, padded_months = get_padded_index(
        df_grouped, all_regions, all_months, WARMUP_START_DATE_1_MONTH
    )

    full_padded_index = pd.MultiIndex.from_product(
        [final_regions, padded_months], names=["region", "year_month"]
    )

    df_expanded = (
        df_padded.set_index(["region", "year_month"])
        .reindex(full_padded_index)
        .reset_index()
        .sort_values(["region", "year_month"])
    )

    df_expanded = df_expanded.rename(columns={"r3q": "rainfall_3m_anomaly"})

    df_expanded["rainfall_3m_anomaly"] = df_expanded.groupby("region")[
        "rainfall_3m_anomaly"
    ].shift(1)

    # Remove additional month so first month is not NaN
    df_expanded = df_expanded[df_expanded["year_month"] >= pd.Period(TRAIN_START_DATE, freq="M")].copy()

    return df_expanded


def get_clean_data(
    all_regions=None,
    all_months=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Orchestrates the reading and processing of rainfall data.

    It ties together the fetching and processing pipelines, replacing any missing
    values in the final anomaly calculations with zero, and extracts a list
    of predictor column names for downstream modelling.

    Args:
        all_regions: Array of unique regions full index.
        all_months: Array of target months for the full index.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing:
            - The final, cleaned, and processed DataFrame.
            - A list of predictor column names (e.g., ["rainfall_3m_anomaly"]).
    """
    rainfall_df = read_rainfall()
    processed_df = process_rainfall(rainfall_df, all_regions, all_months)
    processed_df["rainfall_3m_anomaly"] = processed_df["rainfall_3m_anomaly"]

    predictor_cols = ["rainfall_3m_anomaly"]
    return processed_df, predictor_cols
