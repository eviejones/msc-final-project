"""Runs the model for the set configuration and saves detailed reports in the evaluation folder.

The set configuration is decided in 03_results.ipynb and is:

"k": 1.75,
"event_col": "sub_event_type",
"include_rain": False,
"n_splits": 5,
"seed": 999

Each variant (corpus-type, PCA or not) is run on the best hyperparameters found during training (01_run_test_models.ipynb)."""

import json
from typing import Any

import pandas as pd

from models.train_models import train_evaluate_model
from processing.all_data_processing import get_clean_combined_data
from utils.constants import COUNTRY
from utils.logger import get_logger
from utils.reporting import save_model_report

logger = get_logger("Run best models")


def _get_best_params_from_results(
    all_results: pd.DataFrame, config: dict[str, Any]
) -> dict[str, int | float]:
    """
    Extracts the XGBoost parameters from the results DataFrame.

    Filters the provided DataFrame using the specified configuration to find a
    singular matching row, and extracts the corresponding XGBoost hyperparameters.
    Parameters such as maximum depth are cast to integers where appropriate to
    ensure expected behaviour.

    Args:
        all_results (pd.DataFrame): The DataFrame containing the hyperparameter
            optimisation results.
        config (dict[str, Any]): A dictionary mapping column names to target
            values for filtering the results.

    Returns:
        dict[str, int | float]: A dictionary of the extracted XGBoost
            parameters.

    Raises:
        ValueError: If the configuration matches zero rows or more than one row
            in the results DataFrame.
    """
    mask = pd.Series(True, index=all_results.index)
    for col, val in config.items():
        if col in all_results.columns:
            mask &= all_results[col] == val

    matches = all_results[mask]
    if len(matches) != 1:
        raise ValueError(
            f"Number of matching rows found: {len(matches)} for config {config}"
        )
    row = matches.iloc[0]

    param_names = [  # XGBoost params
        "max_depth",
        "min_child_weight",
        "max_delta_step",
        "gamma",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "colsample_bylevel",
    ]
    params = {p: row[f"param_{p}"].item() for p in param_names}
    for p in ["max_depth", "min_child_weight", "max_delta_step"]:
        params[p] = int(params[p])
    params["seed"] = int(config["seed"])
    return params


def _summarise(label: str, subset: pd.DataFrame) -> dict[str, int | float]:
    """
    Calculates and summarises evaluation metrics for a subset of predictions.

    Computes the recall and precision for a specified subset of data containing
    true labels and model predictions. Prints a formatted summary string and
    returns a dictionary of the calculated metrics.

    Args:
        label (str): A descriptive name for the subset, used in the printed output.
        subset (pd.DataFrame): The data containing the actual and predicted
            values. Must contain 'y_true' and 'y_pred' numeric columns.

    Returns:
        dict[str, int | float]: A dictionary containing the number of rows
            ('n_rows'), total true positives ('n_true_pos'), correctly predicted
            positives ('n_caught'), recall ('recall'), and precision ('precision').
    """
    n_true_pos = subset["y_true"].sum()
    n_caught = subset[(subset["y_true"] == 1) & (subset["y_pred"] == 1)].shape[0]
    recall = n_caught / n_true_pos if n_true_pos else float("nan")
    n_pred_pos = subset["y_pred"].sum()
    precision = n_caught / n_pred_pos if n_pred_pos else float("nan")
    print(
        f"{label}: {len(subset)} rows, {n_true_pos} true escalations, "
        f"{n_caught} caught -> recall={recall:.3f}, precision={precision:.3f}"
    )
    return {
        "n_rows": len(subset),
        "n_true_pos": int(n_true_pos),
        "n_caught": n_caught,
        "recall": recall,
        "precision": precision,
    }


def run_model(
    config: dict[str, Any], params: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Executes the modelling pipeline for a given configuration and parameter set.

    Constructs the appropriate data sources based on the configuration flags,
    retrieves the cleaned and combined dataset, and trains the model. It bypasses
    randomised search to directly evaluate the model using the provided parameters,
    returning performance metrics, SHAP feature importance, and predictions.

    Args:
        config (dict[str, Any]): A dictionary containing pipeline configuration
            variables (e.g., 'include_food', 'k', 'event_col', 'use_pca').
        params (dict[str, Any]): A dictionary of hyperparameters for the model.

    Returns:
        tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]: A tuple
            containing:
            - results: A dictionary of evaluation metrics.
            - best_params: A dictionary of the final hyperparameters used.
            - shap_importance: A DataFrame containing SHAP feature importances.
            - onset_predictions: A DataFrame of predictions for conflict onset.
    """
    data_sources = [
        src
        for src, include in zip(
            ["food", "rain", "text"],
            [config["include_food"], config["include_rain"], config["include_text"]],
        )
        if include
    ]

    model_data, predictor_cols = get_clean_combined_data(
        data_sources=data_sources,
        k=config["k"],
        event_col=config["event_col"],
        conflict_only_embeddings=config["conflict_only_embeddings"],
    )

    final_params = {
        **params,
        "k": config["k"],
        "event_col": config["event_col"],
        "n_splits": config["n_splits"],
        "use_pca": config["use_pca"],
    }

    results, best_params, shap_importance, onset_predictions = train_evaluate_model(
        model_data,
        predictor_cols,
        final_params,
        best_params=True,  # skip RandomizedSearchCV
        use_pca=config["use_pca"],
        compute_shap=True,
        shap_sample_size=2000,
        return_onset_predictions=True,
    )
    return results, best_params, shap_importance, onset_predictions


def model_report(label, config, params):
    results, best_params, shap_importance, onset_predictions = run_model(config, params)
    print("=" * 40)
    print(f"MODEL REPORT: {label}")
    print("=" * 40)

    print("\n--- Results ---")
    print(json.dumps(results, indent=2))

    print("\n--- Best Params ---")
    print(json.dumps(best_params, indent=2))

    print("\n--- SHAP Importance (Top 10) ---")
    print(
        shap_importance.head(10).to_string(index=False)
    )  # ONly to string for better formatting
    print("-" * 40, "\n")

    onset_predictions["year_month"] = onset_predictions["year_month"].astype(str)
    war_outbreak = "2023-04"
    print("Pre and post war:")
    pre_war = onset_predictions[onset_predictions["year_month"] < war_outbreak]
    post_war = onset_predictions[onset_predictions["year_month"] >= war_outbreak]

    pre_war_summary = _summarise("Pre-war  (Jan-Mar 2023)", pre_war)
    post_war_summary = _summarise("Post-war (Apr-Dec 2023)", post_war)

    key_regions = [
        "Khartoum",
        "North Darfur",
        "South Darfur",
        "West Darfur",
        "Central Darfur",
        "East Darfur",
        "West Kordofan",
        "South Kordofan",
    ]

    print("\n-----Key war-affected regions\n")
    key_region_rows = onset_predictions[onset_predictions["region"].isin(key_regions)]
    key_regions_summary = _summarise("Key regions (all onset months)", key_region_rows)

    khartoum_rows = onset_predictions[onset_predictions["region"] == "Khartoum"]
    khartoum_summary = _summarise("  Khartoum", khartoum_rows)

    for region in key_regions:
        if region == "Khartoum":
            continue
        region_rows = onset_predictions[onset_predictions["region"] == region]
        if region_rows["y_true"].sum() > 0:
            _summarise(f"  {region}", region_rows)

    comparison_row = {
        "model": label,
        "event_col": config["event_col"],
        "use_pca": config["use_pca"],
        "onset_aupr": float(results["onset_aupr"]),
        "active_aupr": float(results["active_aupr"]),
        "pre_war_recall": pre_war_summary["recall"],
        "post_war_recall": post_war_summary["recall"],
        "khartoum_recall": khartoum_summary["recall"],
        "key_regions_recall": key_regions_summary["recall"],
    }

    return results, best_params, shap_importance, onset_predictions, comparison_row


def run_best_models(set_confg):
    """Executes the reporting pipeline for all Part 2 (see methodology notebook) final model configurations."""

    all_results = pd.read_csv(f"evaluation/{COUNTRY.lower()}_results.csv")

    # ---- Model A - Structural only
    model_a_config = {
        **set_confg,
        "include_text": False,
        "conflict_only_embeddings": False,
        "use_pca": False,
    }
    model_a_params = _get_best_params_from_results(all_results, model_a_config)
    results_a, best_params_a, shap_a, onset_preds_a, row_a = model_report(
        "Model A", model_a_config, model_a_params
    )
    save_model_report("Model A", results_a, best_params_a, shap_a, onset_preds_a)

    # ---- Model B - conflict-only text, PCA
    model_b_conflict_pca_config = {
        **set_confg,
        "include_text": True,
        "conflict_only_embeddings": True,
        "use_pca": True,
    }
    model_b_conflict_pca_params = _get_best_params_from_results(
        all_results, model_b_conflict_pca_config
    )
    (
        results_b_conflict_pca,
        best_params_b_conflict_pca,
        shap_b_conflict_pca,
        onset_preds_b_conflict_pca,
        row_b_conflict_pca,
    ) = model_report(
        "Model B (conflict-only text PCA)",
        model_b_conflict_pca_config,
        model_b_conflict_pca_params,
    )
    save_model_report(
        "Model B (conflict-only text PCA)",
        results_b_conflict_pca,
        best_params_b_conflict_pca,
        shap_b_conflict_pca,
        onset_preds_b_conflict_pca,
    )

    # ---- Model B - conflict-only text, non-PCA
    model_b_conflict_nopca_config = {
        **set_confg,
        "include_text": True,
        "conflict_only_embeddings": True,
        "use_pca": False,
    }
    model_b_conflict_nopca_params = _get_best_params_from_results(
        all_results, model_b_conflict_nopca_config
    )
    (
        results_b_conflict_nopca,
        best_params_b_conflict_nopca,
        shap_b_conflict_nopca,
        onset_preds_b_conflict_nopca,
        row_b_conflict_nopca,
    ) = model_report(
        "Model B (conflict-only text non-PCA)",
        model_b_conflict_nopca_config,
        model_b_conflict_nopca_params,
    )
    save_model_report(
        "Model B (conflict-only text non-PCA)",
        results_b_conflict_nopca,
        best_params_b_conflict_nopca,
        shap_b_conflict_nopca,
        onset_preds_b_conflict_nopca,
    )

    # ---- Model B - all-event text, non-PCA
    model_b_all_nopca_config = {
        **set_confg,
        "include_text": True,
        "conflict_only_embeddings": False,
        "use_pca": False,
    }
    model_b_all_nopca_params = _get_best_params_from_results(
        all_results, model_b_all_nopca_config
    )
    (
        results_b_all_nopca,
        best_params_b_all_nopca,
        shap_b_all_nopca,
        onset_preds_b_all_nopca,
        row_b_all_nopca,
    ) = model_report(
        "Model B (all-event text non-PCA)",
        model_b_all_nopca_config,
        model_b_all_nopca_params,
    )
    save_model_report(
        "Model B (all-event text non-PCA)",
        results_b_all_nopca,
        best_params_b_all_nopca,
        shap_b_all_nopca,
        onset_preds_b_all_nopca,
    )

    # ---- Model B - all-event text, PCA
    model_b_all_pca_config = {
        **set_confg,
        "include_text": True,
        "conflict_only_embeddings": False,
        "use_pca": True,
    }
    model_b_all_pca_params = _get_best_params_from_results(
        all_results, model_b_all_pca_config
    )
    (
        results_b_all_pca,
        best_params_b_all_pca,
        shap_b_all_pca,
        onset_preds_b_all_pca,
        row_b_all_pca,
    ) = model_report(
        "Model B (all-event text PCA)", model_b_all_pca_config, model_b_all_pca_params
    )
    save_model_report(
        "Model B (all-event text PCA)",
        results_b_all_pca,
        best_params_b_all_pca,
        shap_b_all_pca,
        onset_preds_b_all_pca,
    )


if __name__ == "_main_":
    run_best_models(
        {
            "k": 1.75,
            "event_col": "sub_event_type",
            "include_food": True,
            "include_rain": False,
            "n_splits": 5,
            "seed": 999,
        }
    )
