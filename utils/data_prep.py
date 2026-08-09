import pandas as pd

import processing.acled_events_processing as acled
import processing.acled_text_processing as notes
import processing.food_prices_processing as food
import processing.rainfall_processing as rain

from utils.logger import get_logger
from utils.dates import train_start_date, end_date

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
    split_df = df[
        (df["year_month"] >= start_date) & (df["year_month"] <= end_date)
    ].copy()

    y = split_df[target_col].copy()
    X = split_df[predictor_cols].copy()

    return split_df, y, X


def get_clean_combined_data(
    data_sources: list[str] | None = None,
    download: bool = False,
    remove_abyei: bool = True,
    k: float = 0.5,
    event_col: str = "event_type",
) -> tuple[pd.DataFrame, list[str]]:
    """Fetches and merges clean data from specified sources.

    This function always fetches ACLED data as the foundational dataset.
    It conditionally merges additional datasets (like food and rain) if
    they are specified in the data_sources list.

    Args:
        data_sources (list[str] | None, optional): A list of additional data sources
            to merge. Valid options include "food" and "rain" (case-insensitive).
            Defaults to None.
        download (bool, optional): If True, forces a fresh download of the data
            instead of using cached versions. Defaults to False.
        remove_abyei (bool, optional): If True, filters out data for the Abyei
            region. Defaults to True.
        k (float): The number of standard deviations above the mean to set the
            target threshold. Defaults to 0.5.
        event_col (str): Whether to use event_type or sub_event_type column. Defaults to event_type.

    Returns:
        combined_df (pd.DataFrame): The merged dataset.
        predictor_cols (list[str]): A complete list of predictor column
        names from all merged datasets.
    """
    # ---- Validate inputs
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
    if k > 2:
        raise ValueError(f"k must be less than or equal to 2, got {k}")

    # Validate event_col
    valid_event_cols = {"event_type", "sub_event_type"}
    if event_col not in valid_event_cols:
        raise ValueError(
            f"Invalid event_col: '{event_col}'. Allowed values are: {valid_event_cols}"
        )

    # ---- Fetch data (always fetch ACLED as the base dataset)
    processed_acled_df, acled_predictor_cols, raw_acled_df = acled.get_clean_data(k=k,           
            event_col=event_col)
    combined_df = processed_acled_df
    predictor_cols = acled_predictor_cols
    logger.info("ACLED data processed.")

    all_regions = combined_df["region"].unique()
    all_months = pd.period_range(train_start_date, end_date, freq="M")

    if data_sources is not None:
        sources_lower = [source.lower() for source in data_sources]

        if "food" in sources_lower:
            processed_food_df, food_predictor_cols = food.get_clean_data(
                all_regions=all_regions,
                all_months=all_months,
            )
            combined_df = combined_df.merge(
                processed_food_df, on=["region", "year_month"], how="left"
            )
            predictor_cols = predictor_cols + food_predictor_cols
            logger.info("Food prices data processed.")
        if "rain" in sources_lower:
            processed_rain_df, rain_predictor_cols = rain.get_clean_data(
                force_download=download,
                remove_abyei=remove_abyei,
                all_regions=all_regions,
                all_months=all_months,
            )
            combined_df = combined_df.merge(
                processed_rain_df, on=["region", "year_month"], how="left"
            )
            predictor_cols = predictor_cols + rain_predictor_cols
            logger.info("Rainfall data processed.")
        if "text" in sources_lower:
            if download:
                processed_notes_df, notes_prediction_cols = notes.get_clean_data(
                    calculate=True,
                    df=raw_acled_df,
                    all_regions=all_regions,
                    all_months=all_months,  # Calculates embeddings, this could take a while
                )
            else:
                processed_notes_df, notes_prediction_cols = notes.get_clean_data(
                    False, None, all_regions, all_months
                )  # Read local embeddings
            combined_df = combined_df.merge(
                processed_notes_df, on=["region", "year_month"], how="left"
            )
            emb_fill_cols = [c for c in notes_prediction_cols if c.startswith("emb_")]
            combined_df[emb_fill_cols] = combined_df[emb_fill_cols].fillna(0.0)
            combined_df["has_acled_event"] = (
                combined_df["has_acled_event"].fillna(0).astype(int)
            )
            predictor_cols = predictor_cols + notes_prediction_cols
            logger.info("Notes data processed.")
    return combined_df, predictor_cols
