import json
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

logger = get_logger("Reporting")

REPORTS_DIR = Path("evaluation/model_reports")
reports_dir = Path(REPORTS_DIR)
reports_dir.mkdir(parents=True, exist_ok=True)


def save_model_report(
    label: str,
    results: dict,
    best_params: dict,
    shap_importance: pd.DataFrame,
    onset_predictions: pd.DataFrame,
):
    """Saves the evaluation artefacts for a specific machine learning model.

    Formats the provided label into a file-system-friendly string and saves
    the model's metrics, optimised hyperparameters, SHAP importances, and
    onset predictions to the reports directory.

    Args:
        label (str): The human-readable name or identifier of the model.
        results (dict): A dictionary containing the model's evaluation metrics.
        best_params (dict): A dictionary containing the model's optimised hyperparameters.
        shap_importance (pd.DataFrame): A DataFrame detailing the SHAP feature importances.
        onset_predictions (pd.DataFrame): A DataFrame containing the onset predictions.
    """
    label_formatted = label.replace(" ", "_").replace("(", "").replace(")", "")

    with open(REPORTS_DIR / f"{label_formatted}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(REPORTS_DIR / f"{label_formatted}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    shap_importance.to_csv(REPORTS_DIR / f"{label_formatted}_shap.csv", index=False)
    onset_predictions.to_csv(
        REPORTS_DIR / f"{label_formatted}_onset_predictions.csv", index=False
    )

    logger.info(
        f"Model reports for '{label}' successfully saved to: {REPORTS_DIR.resolve()}"
    )


def open_model_report(label: str):
    """Loads the saved evaluation artefacts for a specific model label.

    Args:
        label (str): The original human-readable identifier of the model
            used during saving.

    Returns:
        tuple: A 4-tuple containing:
            - results (pd.DataFrame): A 1-row DataFrame of the model's evaluation
              results, including the model label.
            - best_params (pd.DataFrame): A 1-row DataFrame of the best hyperparameters.
            - shap_importance (pd.DataFrame): The SHAP feature importances.
            - onset_predictions (pd.DataFrame): The onset predictions.
    """
    label_formatted = label.replace(" ", "_").replace("(", "").replace(")", "")

    with open(REPORTS_DIR / f"{label_formatted}_results.json") as f:
        # Converts the JSON dictionary into a 1-row DataFrame
        results = pd.DataFrame([json.load(f)])
        results["model"] = label

    with open(REPORTS_DIR / f"{label_formatted}_best_params.json") as f:
        best_params = pd.DataFrame([json.load(f)])

    shap_importance = pd.read_csv(REPORTS_DIR / f"{label_formatted}_shap.csv")
    onset_predictions = pd.read_csv(
        REPORTS_DIR / f"{label_formatted}_onset_predictions.csv"
    )

    return results, best_params, shap_importance, onset_predictions


def read_model_reports(files):
    """Compiles and aggregates evaluation reports from multiple models.

    Reads the artefacts for a provided list of model labels, concatenates them
    into unified DataFrames, and standardises the metrics into a melted format
    for easier downstream analysis and plotting.

    Args:
        files (list of str): A list of model labels (identifiers) to process.

    Returns:
        tuple: A 3-tuple containing:
            - df_melted (pd.DataFrame): A melted DataFrame of all model metrics
              (columns: 'model', 'metric', 'score').
            - all_shap (pd.DataFrame): Concatenated SHAP importances across all models.
            - all_onset (pd.DataFrame): Concatenated onset predictions across all models.
    """
    result_list = []
    shap_list = []
    onset_list = []

    for f in files:
        result, best_params, shap_importance, onset_predictions = open_model_report(f)

        model_name = result["model"].iloc[0]

        shap_importance["model"] = model_name
        onset_predictions["model"] = model_name

        result_list.append(result)
        shap_list.append(shap_importance)
        onset_list.append(onset_predictions)

    all_results = pd.concat(result_list)
    all_shap = pd.concat(shap_list)
    all_onset = pd.concat(onset_list)

    df_melted = all_results.melt(
        id_vars=["model"], var_name="metric", value_name="score"
    )
    df_melted["score"] = pd.to_numeric(df_melted["score"], errors="coerce")

    return df_melted, all_shap, all_onset
