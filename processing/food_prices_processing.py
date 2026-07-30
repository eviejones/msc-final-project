import pandas as pd
from ingest.hdx_client import HdxClient
from utils.dates import *
from utils.name_mapping import *

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WFP Processing")
logger.setLevel(logging.INFO)


def read_food_prices(download):
    hdx = HdxClient()

    # 1. Load Sudan
    df_sudan = hdx.get_data(
        dataset_name="wfp-food-prices-for-sudan",
        file_name="Sudan - Food Prices",
        file_type="csv",
        download=download,
    )

    # 2. Load South Sudan
    df_south_sudan = hdx.get_data(
        dataset_name="wfp-food-prices-for-south-sudan",
        file_name="South Sudan - Food Prices",
        file_type="csv",
        download=download,
    )

    # 3. FILTER ONLY ABYEI FROM SOUTH SUDAN DATA
    df_abyei = df_south_sudan[df_south_sudan["market"] == "Abyei"].copy()
    df_abyei["admin1"] = "Abyei"

    # 4. Concatenate Sudan + Abyei (Excludes all other South Sudan regions)
    df_combined = pd.concat([df_sudan, df_abyei], ignore_index=True)

    primary_commodities = ["Sorghum", "Millet", "Wheat flour"]

    df_filtered = df_combined[
        (df_combined["commodity"].isin(primary_commodities))
        & (df_combined["pricetype"] == "Retail")
        & (df_combined["priceflag"].isin(["actual", "aggregate"]))
    ].copy()

    df_filtered_cols = df_filtered[
        ["date", "admin1", "admin2", "market", "commodity", "unit", "usdprice"]
    ].copy()

    def clean_state_names(value):
        return SUDAN_STATE_MAPPING.get(value, value)

    renamed_df = df_filtered_cols.copy()
    renamed_df["admin1"] = df_filtered_cols["admin1"].apply(clean_state_names)
    renamed_df["date"] = pd.to_datetime(renamed_df["date"])
    renamed_df["year_month"] = renamed_df["date"].dt.to_period("M")
    renamed_df["unit_weight_in_kg"] = (
        renamed_df["unit"]
        .str.extract(r"([\d.]+)", expand=False)
        .astype(float)
        .fillna(1.0)
    )
    renamed_df["usdprice"] = pd.to_numeric(renamed_df["usdprice"], errors="coerce")
    renamed_df["usdprice_per_kg"] = (
        renamed_df["usdprice"] / renamed_df["unit_weight_in_kg"]
    )
    return renamed_df


def process_and_pivot_food_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    df = df_prices[
        (df_prices["year_month"] >= start_date) & (df_prices["year_month"] <= end_date)
    ].copy()

    df_grouped = (
        df.groupby(["admin1", "year_month", "commodity"])["usdprice_per_kg"]
        .median()
        .reset_index()
    )

    all_regions = df_grouped["admin1"].unique()
    all_months = pd.period_range(
        df_grouped["year_month"].min(), df_grouped["year_month"].max(), freq="M"
    )
    all_commodities = df_grouped["commodity"].unique()

    full_index = pd.MultiIndex.from_product(
        [all_regions, all_months, all_commodities],
        names=["admin1", "year_month", "commodity"],
    )

    df_expanded = (
        df_grouped.set_index(["admin1", "year_month", "commodity"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["admin1", "commodity", "year_month"])
    )

    df_expanded["usdprice_per_kg"] = df_expanded.groupby(["admin1", "commodity"])[
        "usdprice_per_kg"
    ].transform(lambda x: x.ffill().bfill())

    df_pivoted = df_expanded.pivot(
        index=["admin1", "year_month"],
        columns="commodity",
        values="usdprice_per_kg",
    ).reset_index()

    df_pivoted.columns.name = None
    df_pivoted = df_pivoted.rename(
        columns={
            "Millet": "price_millet",
            "Sorghum": "price_sorghum",
            "Wheat flour": "price_wheat_flour",
        }
    )

    return df_pivoted


def get_clean_data(download=True):
    food_prices_df = read_food_prices(download)
    pivoted_df = process_and_pivot_food_prices(food_prices_df)
    pivoted_df = pivoted_df.rename(columns={"admin1": "region"})

    predictor_cols = [
        col for col in pivoted_df.columns if col not in ["region", "year_month"]
    ]
    pivoted_df[predictor_cols] = pivoted_df[predictor_cols].fillna(0)
    return pivoted_df, predictor_cols
