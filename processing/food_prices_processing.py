import pandas as pd

from ingest.hdx_client import HdxClient
from utils.constants import COUNTRY, PRIMARY_COMMODITIES, TRAIN_START_DATE
from utils.dates import WARMUP_START_DATE_1_MONTH, get_padded_index
from utils.logger import get_logger
from utils.name_mapping import clean_state_names

logger = get_logger("WFP Processing")


def read_food_prices() -> pd.DataFrame:
    """Fetches, filters, and cleans World Food Programme (WFP) food price data for the selected country from the HDX platform.

    This function retrieves raw CSV data, optionally includes the Abyei region from
    the South Sudan dataset, and filters for primary commodities (from constants) sold at retail prices. It standardizes dates, admin regions, and
    calculates a unified USD price per kilogram.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing historical retail prices for
            specific commodities, with standardized state names, temporal periods,
            and calculated 'usdprice_per_kg'.
    """
    hdx = HdxClient()

    country_lower = COUNTRY.lower().replace(" ", "-")

    df = hdx.get_data(
        dataset_name=f"wfp-food-prices-for-{country_lower}",
        file_name=f"{COUNTRY} - Food Prices",
        file_type="csv",
    )

    df_filtered = df[
        (df["commodity"].isin(PRIMARY_COMMODITIES))
        & (df["pricetype"] == "Retail")
        & (df["priceflag"].isin(["actual", "aggregate"]))
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
    renamed_df = renamed_df.rename(columns={"admin1": "region"})

    return renamed_df


def process_and_pivot_food_prices(
    df_prices: pd.DataFrame, all_regions, all_months
) -> pd.DataFrame:
    """Aggregates and pivots food price data to create time-series features.

    This function filters the dataset starting one month prior to the global
    start date (for the burn-in buffer), calculates the median monthly price,
    ensures a continuous monthly timeline, and forward-fills missing gaps.
    The data is then pivoted, shifted by 1 month to create lag features,
    and sliced to start exactly at the target training date.

    Args:
        df_prices (pd.DataFrame): The cleaned food prices DataFrame.
        all_regions: Array-like of all unique regions for the Cartesian spine.
        all_months: Array-like of all target months for the Cartesian spine.

    Returns:
        pd.DataFrame: A continuous time-series DataFrame indexed by region
            and 'year_month', featuring 1-month lagged median prices.
    """
    df_grouped = (
        df_prices.groupby(["region", "year_month", "commodity"])["usdprice_per_kg"]
        .median()
        .reset_index()
    )

    df_padded, final_regions, padded_months = get_padded_index(
        df_grouped, all_regions, all_months, WARMUP_START_DATE_1_MONTH
    )
    all_commodities = df_grouped["commodity"].unique()

    full_padded_index = pd.MultiIndex.from_product(
        [final_regions, padded_months, all_commodities],
        names=["region", "year_month", "commodity"],
    )

    df_expanded = (
        df_padded.set_index(["region", "year_month", "commodity"])
        .reindex(full_padded_index)
        .reset_index()
        .sort_values(["region", "commodity", "year_month"])
    )

    df_expanded["usdprice_per_kg"] = df_expanded.groupby(["region", "commodity"])[
        "usdprice_per_kg"
    ].transform(
        lambda x: x.ffill()
    )  # Forward fill on food data as we can assume that prices remain the same

    df_pivoted = df_expanded.pivot(
        index=["region", "year_month"],
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
    df_pivoted[price_cols] = df_pivoted.groupby("region")[price_cols].shift(1)
    df_pivoted = df_pivoted[
        df_pivoted["year_month"] >= pd.Period(TRAIN_START_DATE, freq="M")
    ].copy()

    return df_pivoted


def get_clean_data(
    all_regions=None,
    all_months=None,
):
    """An orchestrator function that runs the full food price data pipeline.

    This function calls the read and process functions in sequence, and extracts
    a list of predictor column names for downstream modelling. Any leading missing
    values are intentionally left as NaN to be handled natively by XGBoost's
    sparsity-aware split finding.

    Args:
        all_regions (array-like, optional): Array of all unique regions for the Cartesian spine.
        all_months (array-like, optional): Array of all target months for the Cartesian spine.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The final machine-learning-ready dataset.
            - list[str]: A list of column names identifying the predictor
                variables (the lagged price features).
    """
    food_prices_df = read_food_prices()
    pivoted_df = process_and_pivot_food_prices(food_prices_df, all_regions, all_months)

    predictor_cols = [
        col for col in pivoted_df.columns if col not in ["region", "year_month"]
    ]

    return pivoted_df, predictor_cols
