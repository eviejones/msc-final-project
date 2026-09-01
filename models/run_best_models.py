"""Runs the best model configs. These are manually added! The reports are saved in the evaluation folder."""

from pathlib import Path

import pandas as pd

from models.train_models import train_evaluate_model
from utils.constants import COUNTRY
from utils.data_prep import get_clean_combined_data
from utils.logger import get_logger
from utils.reporting import save_model_report

REPORTS_DIR = Path("evaluation/model_reports")
reports_dir = Path(REPORTS_DIR)
reports_dir.mkdir(parents=True, exist_ok=True)

logger = get_logger("Run best models")


def get_best_params_from_results(all_results, config):
    mask = pd.Series(True, index=all_results.index)
    for col, val in config.items():
        if col in all_results.columns:
            mask &= all_results[col] == val

    matches = all_results[mask]
    if len(matches) != 1:
        raise ValueError(
            f"More than one matching row found: {len(matches)} for config {config}"
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
    return params


def summarise(label, subset):
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


def run_model(config, params):
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
        "price_recency": config["price_recency"],
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
    print("========MODEL REPORT========")
    print(f"---Model: {label}\n")
    print("---Results\n")
    print(results)
    print("---Best Params\n")
    print(best_params)
    print("---SHAP Importance\n")
    print(shap_importance)

    onset_predictions["year_month"] = onset_predictions["year_month"].astype(str)
    war_outbreak = "2023-04"
    print("Pre and post war:")
    pre_war = onset_predictions[onset_predictions["year_month"] < war_outbreak]
    post_war = onset_predictions[onset_predictions["year_month"] >= war_outbreak]

    pre_war_summary = summarise("Pre-war  (Jan-Mar 2023)", pre_war)
    post_war_summary = summarise("Post-war (Apr-Dec 2023)", post_war)

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

    print("-----Key war-affected regions\n")
    key_region_rows = onset_predictions[onset_predictions["region"].isin(key_regions)]
    key_regions_summary = summarise("Key regions (all onset months)", key_region_rows)

    khartoum_rows = onset_predictions[onset_predictions["region"] == "Khartoum"]
    khartoum_summary = summarise("  Khartoum", khartoum_rows)

    for region in key_regions:
        if region == "Khartoum":
            continue
        region_rows = onset_predictions[onset_predictions["region"] == region]
        if region_rows["y_true"].sum() > 0:
            summarise(f"  {region}", region_rows)

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
    model_a_params = get_best_params_from_results(all_results, model_a_config)
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
    model_b_conflict_pca_params = get_best_params_from_results(
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
    model_b_conflict_nopca_params = get_best_params_from_results(
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
    model_b_all_nopca_params = get_best_params_from_results(
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
    model_b_all_pca_params = get_best_params_from_results(
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


if __name__ == "__main__":
    run_best_models()
