"""Functions to enable data preparation."""

import pandas as pd

from utils.dates import validate_date_ranges
from utils.logger import get_logger

logger = get_logger("Data preparation")


def calculate_conflict_ratio(df: pd.DataFrame) -> dict:
    """
    Calculates the number of regions where there was a monthly escalation.

    Args:
        df (pd.DataFrame): Processed data.

    Returns:
        dict: Contains the number of target esclations and the ratio.
    """
    count_0 = (df["target_escalation"] == 0).sum()
    count_1 = (df["target_escalation"] == 1).sum()
    ratio = count_0 / count_1

    return {"non-escalation": count_0, "escalation": count_1, "ratio": ratio}


def split_data(
    df: pd.DataFrame,
    predictor_cols: list[str],
    start_date: str,
    end_date: str,
    target_col: str = "target_escalation",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Splits data based on specified dates, returns array of y values and dataframe of all features.

    Args:
        df (pd.DataFrame): Processed data.
        predictor_cols (list[str]): List of columns used for prediction.
        target_col (str): Name of target (Y) column.
        start_date (str): Date to start splitting. In format YYYY-MM-DD.
        end_date (str): Date to end splitting. In format YYYY-MM-DD.

    Returns:
        pd.DataFrame: Split dataframe.
        pd.Series: All target Y values.
        pd.DataFrame: All features dataframe.
    """
    start_period = pd.Period(start_date, freq="M")
    end_period = pd.Period(end_date, freq="M")

    split_df = df[
        (df["year_month"] >= start_period) & (df["year_month"] <= end_period)
    ].copy()

    y = split_df[target_col].copy()
    X = split_df[predictor_cols].copy()

    return split_df, y, X


def validate_data_inputs(
    data_sources: list[str] | None = None, k: float = 0.5, event_col: str = "event_type"
) -> None:
    """Validates the input parameters for combining datasets.

    Checks that the provided data sources, threshold multiplier (k), and
    event column are of the correct type and contain permitted values.
    Also triggers a validation of the global date ranges.

    Args:
        data_sources (list[str] | None, optional): A list of additional data
            sources to merge. Valid options are "food", "rain", and "text".
            Defaults to None.
        k (float, optional): The standard deviation multiplier. Defaults to 0.5.
        event_col (str, optional): The target column for events. Must be either
            "event_type" or "sub_event_type". Defaults to "event_type".

    Raises:
        TypeError: If data_sources is not a list/None, or if k is not numeric.
        ValueError: If unsupported data sources or event columns are provided.
    """
    # Validate dates
    validate_date_ranges()

    # Validate data_sources
    if data_sources is not None:
        if not isinstance(data_sources, list):
            raise TypeError(
                f"data_sources must be a list or None, got {type(data_sources).__name__}"
            )

        valid_sources = {"food", "rain", "text"}
        sources_lower = [source.lower() for source in data_sources]
        invalid_sources = [src for src in sources_lower if src not in valid_sources]

        if invalid_sources:
            raise ValueError(
                f"Invalid data_sources provided: {invalid_sources}. "
                f"Allowed sources are: {list(valid_sources)}"
            )

    # Validate k
    if not isinstance(k, (int, float)):
        raise TypeError(
            f"k must be a numeric value (float or int), got {type(k).__name__}"
        )

    # Validate event_col
    valid_event_cols = {"event_type", "sub_event_type"}
    if event_col not in valid_event_cols:
        raise ValueError(
            f"Invalid event_col: '{event_col}'. Allowed values are: {valid_event_cols}"
        )