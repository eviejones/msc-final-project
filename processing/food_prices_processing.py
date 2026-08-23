import numpy as np
import pandas as pd

from ingest.hdx_client import HdxClient
from utils.constants import COUNTRY, PRIMARY_COMMODITIES, TRAIN_START_DATE
from utils.dates import WARMUP_START_DATE_1_MONTH, get_padded_index
from utils.logger import get_logger
from utils.name_mapping import build_state_name_map

logger = get_logger("WFP Processing")


def read_food_prices(all_regions: np.ndarray | None = None) -> pd.DataFrame:
    """Fetches, filters, and cleans World Food Programme (WFP) food price data for the selected country from the HDX platform.

    This function retrieves raw CSV data and filters for primary commodities
    (from constants) sold at retail prices. It standardises dates, admin
    regions, and calculates a unified USD price per kilogram.

    Args:
        all_regions (np.ndarray | None, optional): Canonical region names
            (e.g. from the ACLED baseline) to fuzzy-match WFP region names
            against. Defaults to None.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing historical retail prices for
            specific commodities, with standardised state names, temporal periods,
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
    name_map = build_state_name_map(df_filtered_cols["admin1"], all_regions)
    renamed_df["admin1"] = df_filtered_cols["admin1"].map(name_map)
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
    df_prices: pd.DataFrame,
    all_regions: np.ndarray | None,
    all_months: pd.PeriodIndex | None,
    price_recency: bool = False,
) -> pd.DataFrame:
    """Aggregates and pivots food price data to create time-series features.

    This function filters the dataset starting one month prior to the global
    start date (for the burn-in buffer), calculates the median monthly price,
    ensures a continuous monthly timeline, and forward-fills missing gaps.
    The data is then pivoted, shifted by 1 month to create lag features,
    and sliced to start exactly at the target training date.

    Args:
        df_prices (pd.DataFrame): The cleaned food prices DataFrame.
        all_regions (np.ndarray | None): Array of all unique regions for the
            Cartesian spine. If None, derived from `df_prices`.
        all_months (pd.PeriodIndex | None): Array of all target months for the
            Cartesian spine. If None, derived from `df_prices`.
        price_recency (bool, optional): If True, adds a
            'months_since_reading_{commodity}' column alongside each price
            column, counting the number of months since the last genuine
            (non-forward-filled) reading. Leading gaps (no reading ever
            recorded) are left as NaN in both price and staleness columns,
            consistent with how leading gaps are handled elsewhere, for
            XGBoost's native missing-value handling. Defaults to False.

    Returns:
        pd.DataFrame: A continuous time-series DataFrame indexed by region
            and 'year_month', featuring 1-month lagged median prices (and,
            optionally, lagged staleness indicators).
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

    if price_recency:
        df_expanded["is_actual_reading"] = df_expanded["usdprice_per_kg"].notna()

    df_expanded["usdprice_per_kg"] = df_expanded.groupby(["region", "commodity"])[
        "usdprice_per_kg"
    ].transform(
        lambda x: x.ffill()
    )  # Forward fill on food data as we can assume that prices remain the same

    if price_recency:
        def months_since_last_reading(s: pd.Series) -> pd.Series:
            """Calculates the number of months since last readin."""
            group_id = s.cumsum()
            counts = s.groupby(group_id).cumcount()
            counts = counts.where(group_id > 0, other=np.nan)
            return counts

        df_expanded["months_since_reading"] = df_expanded.groupby(
            ["region", "commodity"]
        )["is_actual_reading"].transform(months_since_last_reading)

    value_cols = ["usdprice_per_kg"]
    if price_recency:
        value_cols.append("months_since_reading")

    df_pivoted = df_expanded.pivot(
        index=["region", "year_month"],
        columns="commodity",
        values=value_cols,
    )
    df_pivoted.columns = [
        f"{'price' if value == 'usdprice_per_kg' else 'months_since_reading'}"
        f"_{commodity.lower().replace(' ', '_')}"
        for value, commodity in df_pivoted.columns
    ]
    df_pivoted = df_pivoted.reset_index()

    price_cols = [c for c in df_pivoted.columns if c.startswith("price_")]
    staleness_cols = [
        c for c in df_pivoted.columns if c.startswith("months_since_reading_")
    ]

    df_pivoted[price_cols] = df_pivoted.groupby("region")[price_cols].shift(1)
    if staleness_cols:
        df_pivoted[staleness_cols] = df_pivoted.groupby("region")[staleness_cols].shift(1)

    df_pivoted = df_pivoted[
        df_pivoted["year_month"] >= pd.Period(TRAIN_START_DATE, freq="M")
    ].copy()

    return df_pivoted


def get_clean_data(
    all_regions: np.ndarray | None = None,
    all_months: pd.PeriodIndex | None = None,
    price_recency: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """An orchestrator function that runs the full food price data pipeline.

    This function calls the read and process functions in sequence, and extracts
    a list of predictor column names for downstream modelling. Any leading missing
    values are intentionally left as NaN to be handled natively by XGBoost's
    sparsity-aware split finding.

    Args:
        all_regions (np.ndarray | None, optional): Array of all unique regions
            for the Cartesian spine. Defaults to None.
        all_months (pd.PeriodIndex | None, optional): Array of all target
            months for the Cartesian spine. Defaults to None.
        price_recency (bool, optional): If True, adds a
            months-since-last-reading feature per commodity alongside price.
            Defaults to False, preserving the original pipeline's behaviour.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing:
            - pd.DataFrame: The final machine-learning-ready dataset.
            - list[str]: A list of column names identifying the predictor
                variables (the lagged price features, and staleness features
                if enabled).
    """
    food_prices_df = read_food_prices(all_regions)
    pivoted_df = process_and_pivot_food_prices(
        food_prices_df, all_regions, all_months, price_recency
    )

    predictor_cols = [
        col for col in pivoted_df.columns if col not in ["region", "year_month"]
    ]

    return pivoted_df, predictor_cols
