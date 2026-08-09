import numpy as np
import pandas as pd

from ingest.acled_client import AcledClient
from utils.constants import COUNTRY
from utils.dates import END_DATE, WARMUP_START_DATE_6_MONTHS
from utils.logger import get_logger

logger = get_logger("Events processing")


def create_regional_monthly_baseline(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """
    Outputs dataframe with regional monthly conflict events and marked escalations.

    1. Groups data by region and month, counting conflict events
    2. Expands data to include all regions and months
    3. Creates six monthly rolling average and standard deviation
    4. Adds a mew column that indicates whether there was an escalation k deviations about the mean.

    Args:
        df (pd.DataFrame): Full ACLED dataframe including six months build up
        k (float): The number of standard deviations above the mean for it to be considered an escalation in conflict.

    Returns:
        pd.DataFrame: Grouped regional dataframe, with conflict escalation marked (Y)
    """
    df = df.copy()

    df_grouped = (
        df.groupby(["admin1", "year_month"])["conflict"]
        .sum()
        .reset_index(name="conflict_event_count")
    )

    # Build full dataset of all regions and months
    all_regions = df["admin1"].unique()
    all_months = pd.period_range(
        df["year_month"].min(), df["year_month"].max(), freq="M"
    )
    full_index = pd.MultiIndex.from_product(
        [all_regions, all_months], names=["admin1", "year_month"]
    )

    df_grouped = (
        df_grouped.set_index(["admin1", "year_month"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["admin1", "year_month"])
    )

    df_grouped = df_grouped.sort_values(by=["admin1", "year_month"])

    # Calculate rolling statistics ending at the previous month (t-1)
    df_grouped["rolling_mean_6m"] = df_grouped.groupby("admin1")[
        "conflict_event_count"
    ].transform(lambda x: x.rolling(window=6, min_periods=6).mean().shift(1))

    df_grouped["rolling_std_6m"] = df_grouped.groupby("admin1")[
        "conflict_event_count"
    ].transform(lambda x: x.rolling(window=6, min_periods=6).std().shift(1))

    df_grouped["escalation_threshold"] = df_grouped["rolling_mean_6m"] + (
        k * df_grouped["rolling_std_6m"]
    )

    # Define the binary target variable (is current conflict > historical threshold?)
    df_grouped["target_escalation"] = np.where(
        df_grouped["conflict_event_count"] > df_grouped["escalation_threshold"], 1, 0
    )
    logger.info(f"Escalation target set at {k} standard deviations above the mean.")

    return df_grouped


def pre_process_data(
    df: pd.DataFrame, k: float, event_col: str = "sub_event_type"
) -> tuple[pd.DataFrame, list[str]]:
    """
    Processes data so it is suitable to feed into the model.

    1. Groups data by region and year_month, marks conflict events
    2. Combines previously grouped data with fatalities and baseline data
    3. Creates list of columns used for prediction.

    Args:
        df (pd.DataFrames): Full data from ACLED.
        k (float): The number of standard deviations above the mean for it to be considered an escalation in conflict.
        event_col (str): The event column to group on, either sub_event_type or event_type.
    Returns:
        pd.DataFrame: The data ready for passing to the model.
        list[str]: List of columns used for prediction.
    """
    if event_col not in ["event_type", "sub_event_type"]:
        raise ValueError(
            "Event column must be either 'sub_event_type' or 'event_type'."
        )
    df = df.copy()
    if not isinstance(df["year_month"].dtype, pd.PeriodDtype):
        df["year_month"] = pd.to_datetime(df["year_month"]).dt.to_period("M")


    pivot_df = pd.pivot_table(
        df,
        values="event_id_cnty",
        index=["admin1", "year_month"],
        columns=[event_col],
        aggfunc="count",
        fill_value=0,
    ).reset_index()

    logger.info(f"Data grouped by {event_col}")

    pivot_df.columns = (
        pivot_df.columns.str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    baseline_df = create_regional_monthly_baseline(df, k)

    fatalities_df = (
        df.groupby(["admin1", "year_month"])["fatalities"].sum().reset_index()
    )

    combined_df = baseline_df.merge(
        pivot_df, on=["admin1", "year_month"], how="left"
    ).merge(fatalities_df, on=["admin1", "year_month"], how="left")

    # Define what type of column each is
    event_cols = pivot_df.columns.drop(["admin1", "year_month"]).tolist()
    current_event_cols = event_cols + ["fatalities"]
    lagged_event_cols = ["rolling_mean_6m", "rolling_std_6m", "escalation_threshold"]
    predictor_cols = current_event_cols + lagged_event_cols

    combined_df[current_event_cols] = (
        combined_df[current_event_cols]
        .fillna(0)
        .groupby(combined_df["admin1"])[current_event_cols]
        .shift(1)
    )

    combined_df[predictor_cols] = combined_df[predictor_cols].fillna(0)

    combined_df = (
        combined_df.rename(columns={"admin1": "region"})
        .sort_values(by=["year_month", "region"])
        .reset_index(drop=True)
    )

    return combined_df, predictor_cols


def get_clean_data(k: float = 0.5, event_col: str = "event_type"):
    """Gets the clean ACLED event data, both raw and processed with target variables.

    Args:
        k (float): The number of standard deviations above the mean to mark as an escalation. Defaults to 0.5.
        event_col (str): The event column to group on, either sub_event_type or event_type. Defaults to "event_type".
    """

    acled = AcledClient()
    all_data = acled.get_data(
        country=COUNTRY, start_date=WARMUP_START_DATE_6_MONTHS, end_date=END_DATE
    )

    all_data = all_data[
        ~all_data["admin1"].str.lower().str.contains("abyei", na=False)
    ]  # Remove Abyei events as it is included in ACLED data
    processed_df, predictor_cols = pre_process_data(all_data, k, event_col)
    return processed_df, predictor_cols, all_data
