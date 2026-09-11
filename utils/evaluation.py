import numpy as np
import pandas as pd
import shap
import xgboost as xgb


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


def shap_category(feature_name: str) -> str:
    """
    Maps the feature name to a broader category for SHAP analysis.

    Args:
        feature_name (str): The name of the feature.

    Returns:
        str: The category to which the feature belongs.
    """
    feature_name = str(feature_name).lower()
    if feature_name.startswith("emb_") or feature_name.startswith("pc"):
        return "Text Embeddings"
    if feature_name.startswith("rolling_"):
        return "Structural Baseline (Rolling Stats)"
    if "rain" in feature_name:
        return "Rainfall"
    if "price" in feature_name:
        return "Food Prices"
    if "months_since" in feature_name:
        return "Food Prices"
    return "Tabular ACLED Counts"
