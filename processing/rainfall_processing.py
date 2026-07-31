import pandas as pd
from ingest.hdx_client import HdxClient
from utils.dates import *
from utils.name_mapping import clean_state_names, SUDAN_PCODE_MAPPING

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Rainfall Processing")
logger.setLevel(logging.INFO)


def read_rainfall(download=True, remove_abyei=True):
    hdx = HdxClient()
    rainfall_sudan = hdx.get_data(
        dataset_name="sdn-rainfall-subnational",
        file_name="sdn-rainfall-subnat-full",
        file_type="csv",
        download=download
    )
    rainfall_sudan_admin1 = rainfall_sudan[rainfall_sudan["adm_level"] == 1].copy()

    if remove_abyei:
        rainfall_combined = rainfall_sudan_admin1
    else:
        rainfall_ss = hdx.get_data(
            dataset_name="ssd-rainfall-subnational",
            file_name="ssd-rainfall-subnat-full",
            file_type="csv",
            download=download
        )
        rainfall_ss_admin1 = rainfall_ss[rainfall_ss["adm_level"] == 1].copy()

        rainfall_abyei = rainfall_ss_admin1[
            rainfall_ss_admin1["region"].str.lower().str.contains("abyei", na=False) |
            rainfall_ss_admin1["PCODE"].str.contains("SS...", na=False)
            ].copy()

        rainfall_abyei["region"] = "Abyei"

        rainfall_combined = pd.concat([rainfall_sudan_admin1, rainfall_abyei], ignore_index=True)

    rainfall_combined["region"] = rainfall_combined["PCODE"].map(SUDAN_PCODE_MAPPING).fillna(
        rainfall_combined["region"])

    rainfall_combined = rainfall_combined.dropna(subset=["region"])

    rainfall_combined["date"] = pd.to_datetime(rainfall_combined["date"])
    rainfall_combined["year_month"] = rainfall_combined["date"].dt.to_period("M")

    return rainfall_combined

def process_rainfall(df_rainfall: pd.DataFrame) -> pd.DataFrame:
    df = df_rainfall[
        (df_rainfall["year_month"] >= start_date) & (df_rainfall["year_month"] <= end_date)
        ].copy()

    df_grouped = (
        df.groupby(["region", "year_month"])["r3q"]
        .median()
        .reset_index()
    )

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

    df_expanded["rainfall_3m_anomaly"] = df_expanded.groupby("region")["rainfall_3m_anomaly"].shift(1)

    return df_expanded


def get_clean_data(download=True, remove_abyei=True):
    rainfall_df = read_rainfall(download, remove_abyei)
    processed_df = process_rainfall(rainfall_df)

    # Fill any NaNs created by the .shift(1) or missing data with 0 (or a neutral anomaly value)
    processed_df["rainfall_3m_anomaly"] = processed_df["rainfall_3m_anomaly"].fillna(0)

    predictor_cols = ["rainfall_3m_anomaly"]
    return processed_df, predictor_cols
