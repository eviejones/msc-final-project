from utils.data_prep import split_data, calculate_conflict_ratio
from utils.dates import *
from utils.cross_validation import (
    grouped_timeseries_cv_ids,
    verify_cv_splits,
    timeseries_cross_val_predict,
)
import numpy as np
import xgboost as xgb
import logging

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

from sklearn.model_selection import RandomizedSearchCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Train baseline")
logger.setLevel(logging.INFO)


def train_evaluate_model(processed_df, predictor_cols, params, best_params=False):

    # Split data
    train_df, y_train, X_train = split_data(
        processed_df,
        predictor_cols,
        train_start_date,
        train_end_date,
    )
    onset_df, y_onset, X_onset = split_data(
        processed_df,
        predictor_cols,
        onset_start_date,
        onset_end_date,
    )
    active_df, y_active, X_active = split_data(
        processed_df,
        predictor_cols,
        active_start_date,
        active_end_date,
    )

    X_train.columns = X_train.columns.astype(object)
    X_onset.columns = X_onset.columns.astype(object)
    X_active.columns = X_active.columns.astype(object)

    ratios = calculate_conflict_ratio(train_df)
    scale_weight = ratios["non-escalation"] / ratios["escalation"]

    # Extract CV splits before cleaning param_grid
    n_splits = params.get("n_splits", 4)
    grouped_timeseries_cv = list(
        grouped_timeseries_cv_ids(train_df["year_month"], n_splits=n_splits)
    )
    verify_cv_splits(train_df, grouped_timeseries_cv)

    # Clean model hyperparameters by removing non-XGBoost pipeline keys
    param_grid = {
        k: v
        for k, v in params.items()
        if k not in ["k", "event_col", "n_splits", "remove_abyei"]
    }

    if best_params:
        # Fit model directly with exact fixed parameters (no CV search needed)
        best_model = xgb.XGBClassifier(
            scale_pos_weight=scale_weight,
            eval_metric="aucpr",
            random_state=7,
            **param_grid,
        )
        best_model.fit(X_train, y_train)
        fitted_best_params = params
    else:
        # Run hyperparameter search
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

    # Take out-of-fold predictions on training data
    oof_y_true, oof_y_proba = timeseries_cross_val_predict(
        best_model, X_train, y_train, grouped_timeseries_cv
    )

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
        # Onset Metrics
        "onset_aupr": (f"{average_precision_score(y_onset, y_pred_proba_onset):.4f}"),
        "onset_precision_class1": f"{onset_report[class_key]['precision']:.4f}",
        "onset_recall_class1": f"{onset_report[class_key]['recall']:.4f}",
        "onset_f1_class1": f"{onset_report[class_key]['f1-score']:.4f}",
        # Active Metrics
        "active_aupr": (
            f"{average_precision_score(y_active, y_pred_proba_active):.4f}"
        ),
        "active_precision_class1": (f"{active_report[class_key]['precision']:.4f}"),
        "active_recall_class1": f"{active_report[class_key]['recall']:.4f}",
        "active_f1_class1": f"{active_report[class_key]['f1-score']:.4f}",
    }
    fitted_best_params["k"] = params["k"]
    fitted_best_params["n_splits"] = params["n_splits"]
    fitted_best_params["event_col"] = params["event_col"]
    return results, fitted_best_params
