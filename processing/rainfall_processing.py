import logging

import pandas as pd

from ingest.hdx_client import HdxClient
from utils.dates import *
from utils.name_mapping import SUDAN_PCODE_MAPPING, clean_state_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Rainfall Processing")
logger.setLevel(logging.INFO)


# TODO: Figure this out because I don't get Abyei
def read_rainfall(download: bool = True, remove_abyei: bool = True) -> pd.DataFrame:
    """Reads and standardises subnational rainfall data from the HDX client.

    This function retrieves the main Sudan rainfall dataset, filters it to administrative
    level 1, and maps PCODEs to clean region names. Optionally, it retrieves South Sudan
    data to extract information for the Abyei region and appends it to the dataset.

    Args:
        download (bool): Whether to download the fresh dataset from HDX. Defaults to True.
        remove_abyei (bool): If True, excludes the Abyei region from the final dataset.
        If False, fetches South Sudan data to include Abyei. Defaults to True.

    Returns:
        pd.DataFrame: A formatted DataFrame containing the combined rainfall data
        with standardised regions and formatted date periods.
    """
    hdx = HdxClient()

    rainfall_sudan = hdx.get_data(
        dataset_name="sdn-rainfall-subnational",
        file_name="sdn-rainfall-subnat-full",
        file_type="csv",
        download=download,
    )
    rainfall_sudan_admin1 = rainfall_sudan[rainfall_sudan["adm_level"] == 1].copy()
    rainfall_sudan_admin1["region"] = (
        rainfall_sudan_admin1["PCODE"].map(SUDAN_PCODE_MAPPING).apply(clean_state_names)
    )

    if remove_abyei:
        rainfall_combined = rainfall_sudan_admin1
    else:
        # Read South Sudan data and pull where region == "Abyei"
        rainfall_ss = hdx.get_data(
            dataset_name="ssd-rainfall-subnational",
            file_name="ssd-rainfall-subnat-full",
            file_type="csv",
            download=download,
        )
        rainfall_ss_admin1 = rainfall_ss[rainfall_ss["adm_level"] == 1].copy()

        rainfall_ss_admin1["region"] = (
            rainfall_ss_admin1["PCODE"]
            .map(SUDAN_PCODE_MAPPING)
            .apply(clean_state_names)
        )

        rainfall_abyei = rainfall_ss_admin1[
            rainfall_ss_admin1["region"] == "Abyei"
        ].copy()

        rainfall_combined = pd.concat(
            [rainfall_sudan_admin1, rainfall_abyei], ignore_index=True
        )
        print("South Sudan PCODE sample:", rainfall_ss_admin1["PCODE"].head().tolist())
        print("South Sudan columns:", rainfall_ss_admin1.columns.tolist())

    rainfall_combined = rainfall_combined.dropna(subset=["region"])
    rainfall_combined["date"] = pd.to_datetime(rainfall_combined["date"])
    rainfall_combined["year_month"] = rainfall_combined["date"].dt.to_period("M")

    return rainfall_combined


def process_rainfall(df_rainfall: pd.DataFrame) -> pd.DataFrame:
    """Processes the raw rainfall dataset to calculate monthly regional anomalies.

    This function filters the dataset within the global start and end dates,
    calculates the median 'r3q' value for each region and month, ensures a
    complete time series index without missing months, and calculates a
    lagged 3-month rainfall anomaly.

    Args:
        df_rainfall (pd.DataFrame): The raw rainfall DataFrame returned by `read_rainfall`.

    Returns:
        pd.DataFrame: A processed DataFrame indexed by region and year_month,
        containing the shifted rainfall anomalies.
    """
    df = df_rainfall[
        (df_rainfall["year_month"] >= start_date)
        & (df_rainfall["year_month"] <= end_date)
    ].copy()

    df_grouped = df.groupby(["region", "year_month"])["r3q"].median().reset_index()

    all_regions = df_grouped["region"].unique()
    all_months = pd.period_range(
        df_grouped["year_month"].min(), df_grouped["year_month"].max(), freq="M"
    )
    full_index = pd.MultiIndex.from_product(
        [all_regions, all_months],
        names=["region", "year_month"],
    )
    df_expanded = (
        df_grouped.set_index(["region", "year_month"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["region", "year_month"])
    )

    df_expanded = df_expanded.rename(columns={"r3q": "rainfall_3m_anomaly"})

    df_expanded["rainfall_3m_anomaly"] = df_expanded.groupby("region")[
        "rainfall_3m_anomaly"
    ].shift(1)

    return df_expanded


def get_clean_data(
    download: bool = True, remove_abyei: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Orchestrates the reading and processing of rainfall data.

    It ties together the fetching and processing pipelines, replacing any missing
    values in the final anomaly calculations with zero, and extracts a list
    of predictor column names for downstream modelling.

    Args:
        download (bool): Whether to download the fresh dataset from HDX. Defaults to True.
        remove_abyei (bool): Whether to exclude the Abyei region. Defaults to True.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing:
            - The final, cleaned, and processed DataFrame.
            - A list of predictor column names (e.g., ["rainfall_3m_anomaly"]).
    """
    rainfall_df = read_rainfall(download, remove_abyei)
    processed_df = process_rainfall(rainfall_df)
    processed_df["rainfall_3m_anomaly"] = processed_df["rainfall_3m_anomaly"].fillna(0)

    predictor_cols = ["rainfall_3m_anomaly"]
    return processed_df, predictor_cols
