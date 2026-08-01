import pandas as pd
import processing.acled_processing as acled
import processing.food_prices_processing as food
import processing.rainfall_processing as rain
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Data preparation")
logger.setLevel(logging.INFO)


def calculate_conflict_ratio(df: pd.DataFrame) -> dict:
    """Calculates the number of regions where there was a monthly escalation.

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
    params: dict,
    data_sources: list[str] | None = None,
    download: bool = False,
    remove_abyei: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetches and merges clean data from specified sources.

    This function always fetches ACLED data as the foundational dataset.
    It conditionally merges additional datasets (like food and rain) if
    they are specified in the data_sources list.

    Args:
        params (dict): Configuration parameters required for fetching ACLED data.
        data_sources (list[str] | None, optional): A list of additional data sources
            to merge. Valid options include "food" and "rain" (case-insensitive).
            Defaults to None.
        download (bool, optional): If True, forces a fresh download of the data
            instead of using cached versions. Defaults to False.
        remove_abyei (bool, optional): If True, filters out data for the Abyei
            region. Defaults to True.

    Returns:
            combined_df (pd.DataFrame): The merged dataset.
            predictor_cols (list[str]): A complete list of predictor column
            names from all merged datasets.
    """
    # Always fetch ACLED as the base dataset
    processed_acled_df, acled_predictor_cols = acled.get_clean_data(
        params, download, remove_abyei
    )
    combined_df = processed_acled_df
    predictor_cols = acled_predictor_cols
    logger.info("ACLED data processed.")

    if data_sources is not None:
        sources_lower = [source.lower() for source in data_sources]

        if "food" in sources_lower:
            processed_food_df, food_predictor_cols = food.get_clean_data(
                download, remove_abyei
            )
            combined_df = combined_df.merge(
                processed_food_df, on=["region", "year_month"], how="inner"
            )
            predictor_cols = predictor_cols + food_predictor_cols
            logger.info("Food prices data processed.")
        if "rain" in sources_lower:
            processed_rain_df, rain_predictor_cols = rain.get_clean_data(download)
            combined_df = combined_df.merge(
                processed_rain_df, on=["region", "year_month"], how="inner"
            )
            predictor_cols = predictor_cols + rain_predictor_cols
            logger.info("Rainfall data processed.")

    return combined_df, predictor_cols
