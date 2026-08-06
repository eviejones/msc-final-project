import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from utils.logger import get_logger

get_logger("Cross validation")


def grouped_timeseries_cv_ids(dates: pd.Series, n_splits: int = 4):
    """Generates train and test indices for time series cross-validation.

    Args:
        dates (pd.Series): Column of all dates.
        n_splits (int, optional): Number of splits. Defaults to 4.

    Yields:
        tuple[np.ndarray, np.ndarray]: Tuple containing the indices for the current testing and training splits.
    """
    dates = pd.Series(dates).reset_index(drop=True)
    unique_months = np.sort(dates.unique())

    timeseries_cv = TimeSeriesSplit(n_splits=n_splits)

    for train_month_pos, test_month_pos in timeseries_cv.split(unique_months):
        train_months = unique_months[train_month_pos]
        test_months = unique_months[test_month_pos]

        train_idx = dates[dates.isin(train_months)].index.to_numpy()
        test_idx = dates[dates.isin(test_months)].index.to_numpy()
        yield train_idx, test_idx


def verify_cv_splits(df, cv_splits, date_column="year_month"):
    logger.info("Cross-validation testing splits:")
    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        train_dates = df.iloc[train_idx][date_column].unique()
        test_dates = df.iloc[test_idx][date_column].unique()

        train_dates = sorted(train_dates)
        test_dates = sorted(test_dates)
        print(f"--- Fold {fold + 1} ---")
        print(
            f"Train window: {train_dates[0]} to {train_dates[-1]} ({len(train_idx)} rows)"
        )
        print(
            f"Test window:  {test_dates[0]} to {test_dates[-1]} ({len(test_idx)} rows)"
        )

        overlap = set(train_dates).intersection(set(test_dates))
        if overlap:
            print(f"Overlapping months: {overlap}")

        if train_dates[-1] >= test_dates[0]:
            print("Training window overlaps or exceeds the test window!")

        print("-" * 30)


def timeseries_cross_val_predict(
    best_model, X_train: pd.DataFrame, y_train: pd.Series, cv: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns arrays of training data actual Y and predicted probabilities.

    Loops through each fold of training data to produce a full list of actual Y
    and out-of-fold predicted probabilities.

    Args:
        best_model: The estimator model object to be evaluated.
        X_train (pd.DataFrame): Training data predictor features.
        y_train (pd.Series): Training data target labels.
        cv (list[int]): List of cross-validation train/test split index arrays.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the actual target values
        and the corresponding predicted probabilities.
    """
    oof_y_true = []
    oof_y_proba = []

    for train_idx, test_idx in cv:
        fold_model = clone(best_model)
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_test_fold, y_test_fold = X_train.iloc[test_idx], y_train.iloc[test_idx]

        # Fit on the past, predict on the future
        fold_model.fit(X_train_fold, y_train_fold)
        preds = fold_model.predict_proba(X_test_fold)[:, 1]

        oof_y_true.extend(y_test_fold)
        oof_y_proba.extend(preds)
    return np.array(oof_y_true), np.array(oof_y_proba)
