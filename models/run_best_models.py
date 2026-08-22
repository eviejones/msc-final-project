"""Runs the best model configs. These are manually added! The reports are saved in the evaluation folder."""
from pathlib import Path

from train_models import train_evaluate_model

from utils.data_prep import get_clean_combined_data
from utils.logger import get_logger
from utils.reporting import save_model_report

REPORTS_DIR = Path("evaluation/model_reports")
reports_dir = Path(REPORTS_DIR)
reports_dir.mkdir(parents=True, exist_ok=True)

logger = get_logger("Run best models")


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
        conflict_only_embeddings=config["conflict_only"],
        price_recency=config["price_recency"]
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
        "Khartoum", "North Darfur", "South Darfur", "West Darfur",
        "Central Darfur", "East Darfur", "West Kordofan", "South Kordofan",
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

# def main():
#     """Executes the reporting pipeline for all best model configurations."""
    
#     # ---- Model A - Structural only
#     model_a_config = {
#         "include_food": True, "include_rain": True, "include_text": False,
#         "conflict_only": None, "k": 1.75, "event_col": "sub_event_type",
#         "n_splits": 5, "use_pca": False, "price_recency": False
#     }
#     model_a_xgb_params = {
#         "max_depth": 3, "min_child_weight": 1, "max_delta_step": 0, "gamma": 0,
#         "learning_rate": 0.01, "subsample": 0.6, "colsample_bytree": 0.8,
#         "reg_alpha": 2.0, "reg_lambda": 1, "colsample_bylevel": 1.0,
#     }
#     results_a, best_params_a, shap_a, onset_preds_a, row_a = model_report(
#         "Model A", model_a_config, model_a_xgb_params
#     )
#     save_model_report("Model A", results_a, best_params_a, shap_a, onset_preds_a)

#     # ---- Model B - conflict-only text, PCA
#     model_b_conflict_pca_config = {
#         "include_food": True, "include_rain": True, "include_text": True,
#         "conflict_only": True, "k": 1.75, "event_col": "sub_event_type",
#         "n_splits": 5, "use_pca": True, "price_recency": False
#     }
#     model_b_conflict_pca_xgb_params = {
#         "max_depth": 7, "min_child_weight": 1, "max_delta_step": 0, "gamma": 3,
#         "learning_rate": 0.01, "subsample": 1.0, "colsample_bytree": 1.0,
#         "reg_alpha": 2.0, "reg_lambda": 10, "colsample_bylevel": 0.8,
#     }
#     (results_b_conflict_pca, best_params_b_conflict_pca, shap_b_conflict_pca,
#      onset_preds_b_conflict_pca, row_b_conflict_pca) = model_report(
#         "Model B (conflict-only text PCA)", model_b_conflict_pca_config, model_b_conflict_pca_xgb_params
#     )
#     save_model_report("Model B (conflict-only text PCA)", results_b_conflict_pca,
#         best_params_b_conflict_pca, shap_b_conflict_pca, onset_preds_b_conflict_pca)

#     # ---- Model B - conflict-only text, non-PCA
#     model_b_conflict_nopca_config = {
#         "include_food": True, "include_rain": True, "include_text": True,
#         "conflict_only": True, "k": 1.75, "event_col": "event_type",
#         "n_splits": 5, "use_pca": False, "price_recency": False
#     }
#     model_b_conflict_nopca_xgb_params = {
#         "max_depth": 7, "min_child_weight": 1, "max_delta_step": 5, "gamma": 0,
#         "learning_rate": 0.01, "subsample": 1.0, "colsample_bytree": 0.8,
#         "reg_alpha": 0.1, "reg_lambda": 10, "colsample_bylevel": 0.6,
#     }
#     (results_b_conflict_nopca, best_params_b_conflict_nopca, shap_b_conflict_nopca,
#      onset_preds_b_conflict_nopca, row_b_conflict_nopca) = model_report(
#         "Model B (conflict-only text non-PCA)", model_b_conflict_nopca_config, model_b_conflict_nopca_xgb_params
#     )
#     save_model_report("Model B (conflict-only text non-PCA)", results_b_conflict_nopca,
#         best_params_b_conflict_nopca, shap_b_conflict_nopca, onset_preds_b_conflict_nopca)

#     # ---- Model B - all-event text, non-PCA
#     model_b_all_nopca_config = {
#         "include_food": True, "include_rain": True, "include_text": True,
#         "conflict_only": False, "k": 1.75, "event_col": "event_type",
#         "n_splits": 5, "use_pca": False, "price_recency": False
#     }
#     model_b_all_nopca_xgb_params = {
#         "max_depth": 5, "min_child_weight": 1, "max_delta_step": 1, "gamma": 5,
#         "learning_rate": 0.01, "subsample": 0.6, "colsample_bytree": 0.8,
#         "reg_alpha": 1.0, "reg_lambda": 5, "colsample_bylevel": 1.0,
#     }
#     (results_b_all_nopca, best_params_b_all_nopca, shap_b_all_nopca,
#      onset_preds_b_all_nopca, row_b_all_nopca) = model_report(
#         "Model B (all-event text non-PCA)", model_b_all_nopca_config, model_b_all_nopca_xgb_params
#     )
#     save_model_report("Model B (all-event text non-PCA)", results_b_all_nopca,
#         best_params_b_all_nopca, shap_b_all_nopca, onset_preds_b_all_nopca)

#     # ---- Model B - all-event text, PCA
#     model_b_all_pca_config = {
#         "include_food": True, "include_rain": True, "include_text": True,
#         "conflict_only": False, "k": 1.75, "event_col": "sub_event_type",
#         "n_splits": 5, "use_pca": True, "price_recency": False
#     }
#     model_b_all_pca_xgb_params = {
#         "max_depth": 5, "min_child_weight": 1, "max_delta_step": 1, "gamma": 5,
#         "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8,
#         "reg_alpha": 2.0, "reg_lambda": 10, "colsample_bylevel": 1.0,
#     }
#     (results_b_all_pca, best_params_b_all_pca, shap_b_all_pca,
#      onset_preds_b_all_pca, row_b_all_pca) = model_report(
#         "Model B (all-event text PCA)", model_b_all_pca_config, model_b_all_pca_xgb_params
#     )
#     save_model_report("Model B (all-event text PCA)", results_b_all_pca,
#         best_params_b_all_pca, shap_b_all_pca, onset_preds_b_all_pca)



def main():
    """Executes the reporting pipeline for all best model configurations."""
    
    # ---- Model A - Structural only
    model_a_config = {
        "include_food": True, "include_rain": True, "include_text": False,
        "conflict_only": None, "k": 1.75, "event_col": "sub_event_type",
        "n_splits": 5, "use_pca": False, "price_recency": True
    }
    model_a_xgb_params = {
        "max_depth": 3, "min_child_weight": 1, "max_delta_step": 0, "gamma": 0,
        "learning_rate": 0.01, "subsample": 0.6, "colsample_bytree": 0.8,
        "reg_alpha": 2.0, "reg_lambda": 1, "colsample_bylevel": 1.0,
    }
    results_a, best_params_a, shap_a, onset_preds_a, row_a = model_report(
        "Model A", model_a_config, model_a_xgb_params
    )
    save_model_report("Model A - food", results_a, best_params_a, shap_a, onset_preds_a)

    # ---- Model B - conflict-only text, PCA
    model_b_conflict_pca_config = {
        "include_food": True, "include_rain": True, "include_text": True,
        "conflict_only": True, "k": 1.75, "event_col": "sub_event_type",
        "n_splits": 5, "use_pca": True, "price_recency": True
    }
    model_b_conflict_pca_xgb_params = {
        "max_depth": 7, "min_child_weight": 1, "max_delta_step": 0, "gamma": 3,
        "learning_rate": 0.01, "subsample": 1.0, "colsample_bytree": 1.0,
        "reg_alpha": 2.0, "reg_lambda": 10, "colsample_bylevel": 0.8,
    }
    (results_b_conflict_pca, best_params_b_conflict_pca, shap_b_conflict_pca,
     onset_preds_b_conflict_pca, row_b_conflict_pca) = model_report(
        "Model B (conflict-only text PCA)", model_b_conflict_pca_config, model_b_conflict_pca_xgb_params
    )
    save_model_report("Model B (conflict-only text PCA) - food", results_b_conflict_pca,
        best_params_b_conflict_pca, shap_b_conflict_pca, onset_preds_b_conflict_pca)

    # ---- Model B - conflict-only text, non-PCA
    model_b_conflict_nopca_config = {
        "include_food": True, "include_rain": True, "include_text": True,
        "conflict_only": True, "k": 1.75, "event_col": "event_type",
        "n_splits": 5, "use_pca": False, "price_recency": True
    }
    model_b_conflict_nopca_xgb_params = {
        "max_depth": 7, "min_child_weight": 1, "max_delta_step": 5, "gamma": 0,
        "learning_rate": 0.01, "subsample": 1.0, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 10, "colsample_bylevel": 0.6,
    }
    (results_b_conflict_nopca, best_params_b_conflict_nopca, shap_b_conflict_nopca,
     onset_preds_b_conflict_nopca, row_b_conflict_nopca) = model_report(
        "Model B (conflict-only text non-PCA)", model_b_conflict_nopca_config, model_b_conflict_nopca_xgb_params
    )
    save_model_report("Model B (conflict-only text non-PCA) - food", results_b_conflict_nopca,
        best_params_b_conflict_nopca, shap_b_conflict_nopca, onset_preds_b_conflict_nopca)

    # ---- Model B - all-event text, non-PCA
    model_b_all_nopca_config = {
        "include_food": True, "include_rain": True, "include_text": True,
        "conflict_only": False, "k": 1.75, "event_col": "event_type",
        "n_splits": 5, "use_pca": False, "price_recency": True
    }
    model_b_all_nopca_xgb_params = {
        "max_depth": 5, "min_child_weight": 1, "max_delta_step": 1, "gamma": 5,
        "learning_rate": 0.01, "subsample": 0.6, "colsample_bytree": 0.8,
        "reg_alpha": 1.0, "reg_lambda": 5, "colsample_bylevel": 1.0,
    }
    (results_b_all_nopca, best_params_b_all_nopca, shap_b_all_nopca,
     onset_preds_b_all_nopca, row_b_all_nopca) = model_report(
        "Model B (all-event text non-PCA) - food", model_b_all_nopca_config, model_b_all_nopca_xgb_params
    )
    save_model_report("Model B (all-event text non-PCA) - food", results_b_all_nopca,
        best_params_b_all_nopca, shap_b_all_nopca, onset_preds_b_all_nopca)

    # ---- Model B - all-event text, PCA
    model_b_all_pca_config = {
        "include_food": True, "include_rain": True, "include_text": True,
        "conflict_only": False, "k": 1.75, "event_col": "sub_event_type",
        "n_splits": 5, "use_pca": True, "price_recency": True
    }
    model_b_all_pca_xgb_params = {
        "max_depth": 5, "min_child_weight": 1, "max_delta_step": 1, "gamma": 5,
        "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 2.0, "reg_lambda": 10, "colsample_bylevel": 1.0,
    }
    (results_b_all_pca, best_params_b_all_pca, shap_b_all_pca,
     onset_preds_b_all_pca, row_b_all_pca) = model_report(
        "Model B (all-event text PCA)", model_b_all_pca_config, model_b_all_pca_xgb_params
    )
    save_model_report("Model B (all-event text PCA) - food", results_b_all_pca,
        best_params_b_all_pca, shap_b_all_pca, onset_preds_b_all_pca)

if __name__ == "__main__":
    main()
