import logging

import pandas as pd

from ingest.hdx_client import HdxClient
from utils.dates import *
from utils.name_mapping import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WFP Processing")
logger.setLevel(logging.INFO)


def read_food_prices(download: bool = True, remove_abyei: bool = True) -> pd.DataFrame:
    """Fetches, filters, and cleans World Food Programme (WFP) food price data for Sudan
    from the HDX platform.

    This function retrieves raw CSV data, optionally includes the Abyei region from
    the South Sudan dataset, and filters for primary commodities (Sorghum, Millet,
    Wheat flour) sold at retail prices. It standardizes dates, admin regions, and
    calculates a unified USD price per kilogram.

    Args:
        download (bool, optional): If True, downloads fresh datasets from HDX.
            If False, relies on local cached files. Defaults to True.
        remove_abyei (bool, optional): If True, excludes the Abyei region. If False,
            fetches the South Sudan dataset, extracts Abyei market data, and appends
            it to the Sudan dataset. Defaults to True.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing historical retail prices for
            specific commodities, with standardized state names, temporal periods,
            and calculated 'usdprice_per_kg'.
    """
    hdx = HdxClient()

    df_sudan = hdx.get_data(
        dataset_name="wfp-food-prices-for-sudan",
        file_name="Sudan - Food Prices",
        file_type="csv",
        download=download,
    )

    if remove_abyei:
        df_combined = df_sudan
    else:  # Load Abyei from South Sudan
        df_south_sudan = hdx.get_data(
            dataset_name="wfp-food-prices-for-south-sudan",
            file_name="South Sudan - Food Prices",
            file_type="csv",
            download=download,
        )

        df_abyei = df_south_sudan[df_south_sudan["market"] == "Abyei"].copy()
        df_abyei["admin1"] = "Abyei"

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
    """Aggregates and pivots food price data to create time-series features.

    This function filters the dataset by global start/end dates, calculates the
    median monthly price per region and commodity, and ensures a continuous monthly
    timeline by reindexing. Missing values are filled using forward-fill followed
    by backward-fill. The data is then pivoted so each commodity is a column, and
    values are shifted by 1 month to create lag features for predictive modeling.

    Args:
        df_prices (pd.DataFrame): The cleaned food prices DataFrame, typically
            the output from `read_food_prices`.

    Returns:
        pd.DataFrame: A continuous time-series DataFrame indexed by region ('admin1')
            and 'year_month', featuring 1-month lagged median prices for millet,
            sorghum, and wheat flour.
    """
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
    price_cols = ["price_millet", "price_sorghum", "price_wheat_flour"]
    df_pivoted[price_cols] = df_pivoted.groupby("admin1")[price_cols].shift(1)
    return df_pivoted


def get_clean_data(download: bool = True, remove_abyei: bool = True):
    """An orchestrator function that runs the full food price data pipeline.

    This function calls the read and process functions in sequence, renames
    the geography column to a standard 'region' name, and replaces any
    remaining NaN values in the feature columns with 0.

    Args:
        download (bool, optional): Indicates whether to download fresh data
            from HDX. Defaults to True.
        remove_abyei (bool, optional): Indicates whether to exclude Abyei market
            data from South Sudan. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The final machine-learning-ready dataset.
            - list[str]: A list of column names identifying the predictor
              variables (the lagged price features).
    """
    food_prices_df = read_food_prices(download, remove_abyei)
    pivoted_df = process_and_pivot_food_prices(food_prices_df)
    pivoted_df = pivoted_df.rename(columns={"admin1": "region"})

    predictor_cols = [
        col for col in pivoted_df.columns if col not in ["region", "year_month"]
    ]
    pivoted_df[predictor_cols] = pivoted_df[predictor_cols].fillna(0)
    return pivoted_df, predictor_cols
