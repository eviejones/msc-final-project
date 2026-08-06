import logging

import numpy as np
import pandas as pd

from ingest.acled_client import AcledClient
from utils.dates import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ACLED processing")
logger.setLevel(logging.INFO)


def mark_conflict_events(df: pd.DataFrame) -> pd.DataFrame:
    """Takes the ACLED dataframe and marks event as conflict (1) or not conflict (0).

     Conflict events are used to create the target Y.

     Args:
         df (pd.DataFrame): The full ACLED dataframe
    Returns:
        pd.DataFrame: The dataframe with am additional column 'conflict' with binary markers.
    """

    acled_subevent_mapping = {
        # BATTLES (Conflict)
        "Armed clash": 1,
        "Government regains territory": 1,
        "Non-state actor overtakes territory": 1,
        # EXPLOSIONS / REMOTE VIOLENCE (Conflict)
        "Air/drone strike": 1,
        "Chemical weapon": 1,
        "Remote explosive/landmine/IED": 1,
        "Shelling/artillery/missile attack": 1,
        "Suicide bomb": 1,
        "Grenade": 1,
        # VIOLENCE AGAINST CIVILIANS (Conflict)
        "Abduction/forced disappearance": 1,
        "Attack": 1,
        "Sexual violence": 1,
        # RIOTS (Conflict)
        "Mob violence": 1,
        "Violent demonstration": 1,
        # PROTESTS (Non-conflict)
        "Excessive force against protesters": 0,
        "Peaceful protest": 0,
        "Protest with intervention": 0,
        # STRATEGIC DEVELOPMENTS (Non-conflict)
        "Agreement": 0,
        "Arrests": 0,
        "Change to group/activity": 0,
        "Disrupted weapons use": 0,
        "Headquarters or base established": 0,
        "Looting/property destruction": 0,
        "Non-violent transfer of territory": 0,
        "Other": 0,
    }
    df["conflict"] = df["sub_event_type"].apply(lambda x: acled_subevent_mapping[x])
    return df  # TODO add validation


def create_regional_monthly_baseline(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """Outputs dataframe with regional monthly conflict events and marked escalations.

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
    """Processes data so it is suitable to feed into the model.

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
    df["year_month"] = pd.to_datetime(df["year_month"]).dt.to_period("M")

    df = mark_conflict_events(df)

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


def get_clean_data(
    download=True, remove_abyei=True, k: float = 0.5, event_col: str = "event_type"
):
    if download:
        acled = AcledClient()
        all_data = acled.get_data(countries, start_date, end_date)
    else:
        all_data = pd.read_csv("../data/all_data.csv")

    if remove_abyei:
        all_data = all_data[
            ~all_data["admin1"].str.lower().str.contains("abyei", na=False)
        ]
    processed_df, predictor_cols = pre_process_data(all_data, k, event_col)
    return processed_df, predictor_cols, all_data
