import pandas as pd

import processing.acled_events_processing as acled
import processing.acled_text_processing as notes
import processing.food_prices_processing as food
import processing.rainfall_processing as rain
from utils.data_prep import validate_data_inputs
from utils.dates import (
    END_DATE,
    TRAIN_START_DATE,
    validate_data_coverage,
)
from utils.logger import get_logger

logger = get_logger("Data preparation")

def get_clean_combined_data(
    data_sources: list[str] | None = None,
    k: float = 1.75,
    event_col: str = "sub_event_type",
    conflict_only_embeddings: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetches and merges clean data from specified sources.

    This function always fetches ACLED data as the foundational dataset.
    It conditionally merges additional datasets (like food and rain) if
    they are specified in the data_sources list.

    Args:
        data_sources (list[str] | None, optional): A list of additional data sources
            to merge. Valid options include "food" and "rain" (case-insensitive).
            Defaults to None.
        k (float): The number of standard deviations above the mean to set the
            target threshold. Defaults to 0.5.
        event_col (str): Whether to use event_type or sub_event_type column. Defaults to event_type.
        conflict_only_embeddings (bool): If True and "text" is in data_sources, text
            embeddings are computed only from events where conflict == 1, rather than
            all events. Cached separately so both variants can be compared. Does not
            affect any other predictor columns. Defaults to False.

    Returns:
        combined_df (pd.DataFrame): The merged dataset.
        predictor_cols (list[str]): A complete list of predictor column
        names from all merged datasets.
    """
    # ---- Validation
    validate_data_inputs(data_sources, k, event_col)

    # ---- Fetch data (always fetch ACLED as the base dataset)
    processed_acled_df, acled_predictor_cols, raw_acled_df = acled.get_clean_data(
        k=k, event_col=event_col
    )
    combined_df = processed_acled_df
    predictor_cols = acled_predictor_cols
    logger.info("ACLED data processed.")

    processed_datasets = {"ACLED events": processed_acled_df}

    # ---- Set region/month based on ACLED and testing and training period

    all_regions = combined_df["region"].unique()
    all_months = pd.period_range(TRAIN_START_DATE, END_DATE, freq="M")

    # ---- Get data for each of the additional sources

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
            processed_datasets["Food prices"] = processed_food_df
            logger.info("Food prices data processed.")
        if "rain" in sources_lower:
            processed_rain_df, rain_predictor_cols = rain.get_clean_data(
                all_regions=all_regions,
                all_months=all_months,
            )
            combined_df = combined_df.merge(
                processed_rain_df, on=["region", "year_month"], how="left"
            )
            predictor_cols = predictor_cols + rain_predictor_cols
            processed_datasets["Rainfall"] = processed_rain_df
            logger.info("Rainfall data processed.")
        if "text" in sources_lower:
            processed_notes_df, notes_prediction_cols = notes.get_clean_data(
                df=raw_acled_df,
                all_regions=all_regions,
                all_months=all_months,
                conflict_only=conflict_only_embeddings,
            )
            combined_df = combined_df.merge(
                processed_notes_df, on=["region", "year_month"], how="left"
            )
            predictor_cols = predictor_cols + notes_prediction_cols
            processed_datasets["Text embeddings"] = processed_notes_df
            logger.info("Notes data processed.")

    # ---- Validate that every processed source covers TRAIN_START_DATE - END_DATE
    validate_data_coverage(processed_datasets)

    return combined_df, predictor_cols
