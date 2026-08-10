from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import RandomizedSearchCV

from processing.acled_text_processing import apply_pca_train_only
from utils.constants import (
    ACTIVE_END_DATE,
    ACTIVE_START_DATE,
    ONSET_END_DATE,
    ONSET_START_DATE,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
)
from utils.cross_validation import (
    grouped_timeseries_cv_ids,
    timeseries_cross_val_predict,
    verify_cv_splits,
)
from utils.data_prep import calculate_conflict_ratio, split_data
from utils.logger import get_logger

logger = get_logger("Train model")

# Ref: https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
def compute_shap_importance(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    predictor_cols: list[str],
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Computes SHAP values for each feature from a fitted XGBoost model.

    Args:
        model (xgb.XGBClassifier): A fitted XGBoost classifier.
        X (pd.DataFrame): Feature matrix to explain (e.g. X_train). For large
            datasets, consider passing a random sample (e.g. X.sample(2000,
            random_state=7)) to keep runtime reasonable.
        predictor_cols (list[str]): Column names corresponding to X's columns,
            in order. Pass the final_predictor_cols from train_evaluate_model
            (post-PCA names if use_pca=True).
        top_n (int, optional): Number of top features to return. Defaults to 30.

    Returns:
        pd.DataFrame: Columns ["feature", "mean_abs_shap"], sorted descending,
            truncated to top_n rows.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": predictor_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df.head(top_n)


def train_evaluate_model(
    processed_df: pd.DataFrame,
    predictor_cols: list[str],
    params: dict[str, Any],
    best_params: bool = False,
    use_pca: bool = True,
    compute_shap: bool = False,
    shap_sample_size: int | None = 2000,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame | None]:
    """
    Trains and evaluates an XGBoost classifier on time-series split data, optionally using PCA.

    The dataset is split into training, onset, and active sets based on predefined date ranges.
    It evaluates models using time-series cross-validation on the training set to find the
    optimal classification threshold (maximizing the F1 score). It then applies this threshold
    to the onset and active test sets and returns detailed classification metrics.

    Args:
        processed_df (pd.DataFrame): The full preprocessed dataset containing features,
            targets, and a 'year_month' column.
        predictor_cols (list[str]): A list of column names in processed_df to use as features.
        params (dict[str, Any]): A dictionary containing hyperparameters for XGBoost or the
            RandomizedSearchCV grid. Must include 'k', 'n_splits', and 'event_col' keys.
        best_params (bool, optional): If True, skips RandomizedSearchCV and fits the XGBoost
            model directly using the provided `params`. Defaults to False.
        use_pca (bool, optional): If True, applies PCA to the embedding features
            (fitting only on the training set to prevent leakage). Defaults to True.
        compute_shap (bool, optional): If True, computes SHAP feature importance on
            the trained model using the training set (or a sample of it, see
            shap_sample_size). Adds runtime, so leave False for sweep/search runs
            and only enable for final chosen configs. Defaults to False.
        shap_sample_size (int | None, optional): If set and compute_shap is True,
            SHAP is computed on a random sample of this many training rows instead
            of the full training set, to keep runtime manageable. Set to None to
            use the full training set. Defaults to 2000.

    Returns:
        tuple[dict[str, Any], dict[str, Any], pd.DataFrame | None]: A tuple containing:
            - results (dict): A dictionary of evaluation metrics (AUPRC, Precision, Recall, F1)
                for both onset and active datasets, as well as the optimal threshold.
            - fitted_best_params (dict): A dictionary of the best hyperparameters used for the
                model.
            - shap_importance (pd.DataFrame | None): Top features by mean absolute SHAP
                value if compute_shap=True, otherwise None.

    Raises:
        TypeError: If arguments are of incorrect types.
        ValueError: If required columns or parameter keys are missing.
    """
    # ---- Input Validation
    if not isinstance(processed_df, pd.DataFrame):
        raise TypeError(
            f"processed_df must be a pandas DataFrame, got {type(processed_df).__name__}"
        )

    if not isinstance(predictor_cols, list) or not all(
        isinstance(c, str) for c in predictor_cols
    ):
        raise TypeError("predictor_cols must be a list of strings")

    if not predictor_cols:
        raise ValueError("predictor_cols cannot be empty")

    missing_cols = [col for col in predictor_cols if col not in processed_df.columns]
    if missing_cols:
        raise ValueError(
            f"The following predictor columns are missing from processed_df: {missing_cols}"
        )

    if "year_month" not in processed_df.columns:
        raise ValueError(
            "processed_df must contain a 'year_month' column for time-series splitting"
        )

    if not isinstance(params, dict):
        raise TypeError(f"params must be a dictionary, got {type(params).__name__}")

    required_params = {"k", "n_splits", "event_col"}
    missing_params = required_params - params.keys()
    if missing_params:
        raise ValueError(
            f"params dictionary is missing required keys: {missing_params}"
        )

    if not isinstance(best_params, bool):
        raise TypeError(
            f"best_params must be a boolean, got {type(best_params).__name__}"
        )

    if not isinstance(use_pca, bool):
        raise TypeError(f"use_pca must be a boolean, got {type(use_pca).__name__}")

    # ----Data Processing & Splitting

    # Still includes emb
    train_df, y_train, _ = split_data(
        processed_df,
        predictor_cols,
        TRAIN_START_DATE,
        TRAIN_END_DATE,
    )
    onset_df, y_onset, _ = split_data(
        processed_df,
        predictor_cols,
        ONSET_START_DATE,
        ONSET_END_DATE,
    )
    active_df, y_active, _ = split_data(
        processed_df,
        predictor_cols,
        ACTIVE_START_DATE,
        ACTIVE_END_DATE,
    )

    if use_pca:
        X_train, X_onset, X_active, pca, final_predictor_cols = apply_pca_train_only(
            train_df, onset_df, active_df, predictor_cols
        )
    else:
        X_train = train_df[predictor_cols].copy()
        X_onset = onset_df[predictor_cols].copy()
        X_active = active_df[predictor_cols].copy()
        final_predictor_cols = predictor_cols

    X_train.columns = X_train.columns.astype(object)
    X_onset.columns = X_onset.columns.astype(object)
    X_active.columns = X_active.columns.astype(object)

    ratios = calculate_conflict_ratio(train_df)
    scale_weight = ratios["non-escalation"] / ratios["escalation"]

    n_splits = params.get("n_splits", 4)
    grouped_timeseries_cv = list(
        grouped_timeseries_cv_ids(train_df["year_month"], n_splits=n_splits)
    )
    verify_cv_splits(train_df, grouped_timeseries_cv)

    param_grid = {
        k: v
        for k, v in params.items()
        if k not in ["k", "event_col", "n_splits", "remove_abyei", "use_pca"]
    }

    # ---- Model Training
    if best_params:
        best_model = xgb.XGBClassifier(
            scale_pos_weight=scale_weight,
            eval_metric="aucpr",
            random_state=7,
            **param_grid,
        )
        best_model.fit(X_train, y_train)
        fitted_best_params = params
    else:
        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_weight,
            eval_metric="aucpr",
            random_state=7,
        )

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=150,
            cv=grouped_timeseries_cv,
            scoring="average_precision",
            n_jobs=-1,
            random_state=23,
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        fitted_best_params = random_search.best_params_

    shap_importance = None
    if compute_shap:
        if shap_sample_size is not None and len(X_train) > shap_sample_size:
            X_shap = X_train.sample(shap_sample_size, random_state=7)
        else:
            X_shap = X_train
        shap_importance = compute_shap_importance(
            best_model, X_shap, final_predictor_cols
        )

    oof_y_true, oof_y_proba = timeseries_cross_val_predict(
        best_model, X_train, y_train, grouped_timeseries_cv
    )

    # ---- Optimisation
    precisions, recalls, thresholds = precision_recall_curve(oof_y_true, oof_y_proba)
    f1_scores = (2 * precisions * recalls / (precisions + recalls + 1e-10))[:-1]
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    # Evaluate on onset test set
    y_pred_proba_onset = best_model.predict_proba(X_onset)[:, 1]
    y_pred_custom_onset = (y_pred_proba_onset >= optimal_threshold).astype(int)

    # Evaluate on active test set
    y_pred_proba_active = best_model.predict_proba(X_active)[:, 1]
    y_pred_custom_active = (y_pred_proba_active >= optimal_threshold).astype(int)

    onset_report = classification_report(
        y_onset, y_pred_custom_onset, output_dict=True, zero_division=0
    )
    active_report = classification_report(
        y_active, y_pred_custom_active, output_dict=True, zero_division=0
    )

    class_key = "1" if "1" in onset_report else 1

    results = {
        "optimal_threshold": f"{optimal_threshold:.4f}",
        "n_predictors": len(final_predictor_cols),
        # Onset Metrics
        "onset_aupr": f"{average_precision_score(y_onset, y_pred_proba_onset):.4f}",
        "onset_precision_class1": f"{onset_report[class_key]['precision']:.4f}",
        "onset_recall_class1": f"{onset_report[class_key]['recall']:.4f}",
        "onset_f1_class1": f"{onset_report[class_key]['f1-score']:.4f}",
        # Active Metrics
        "active_aupr": f"{average_precision_score(y_active, y_pred_proba_active):.4f}",
        "active_precision_class1": f"{active_report[class_key]['precision']:.4f}",
        "active_recall_class1": f"{active_report[class_key]['recall']:.4f}",
        "active_f1_class1": f"{active_report[class_key]['f1-score']:.4f}",
    }

    fitted_best_params["k"] = params["k"]
    fitted_best_params["n_splits"] = params["n_splits"]
    fitted_best_params["event_col"] = params["event_col"]

    return results, fitted_best_params, shap_importance