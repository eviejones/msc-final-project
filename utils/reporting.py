import json
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("evaluation/model_reports")
reports_dir = Path(REPORTS_DIR)
reports_dir.mkdir(parents=True, exist_ok=True)

def save_model_report(label, results, best_params, shap_importance, onset_predictions):
    label_formatted = label.replace(" ", "_").replace("(", "").replace(")","")
    
    with open(REPORTS_DIR / f"{label_formatted}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(REPORTS_DIR / f"{label_formatted}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
        
    shap_importance.to_csv(REPORTS_DIR / f"{label_formatted}_shap.csv", index=False)
    onset_predictions.to_csv(REPORTS_DIR / f"{label_formatted}_onset_predictions.csv", index=False)
    
    
def open_model_report(label):
    label_formatted = label.replace(" ", "_").replace("(", "").replace(")","")

    with open(REPORTS_DIR / f"{label_formatted}_results.json") as f:
        # Converts the JSON dictionary into a 1-row DataFrame
        results = pd.DataFrame([json.load(f)])
        results["model"] = label
        
    with open(REPORTS_DIR / f"{label_formatted}_best_params.json") as f:
        best_params = pd.DataFrame([json.load(f)])
        
    shap_importance = pd.read_csv(REPORTS_DIR / f"{label_formatted}_shap.csv")
    onset_predictions = pd.read_csv(REPORTS_DIR / f"{label_formatted}_onset_predictions.csv")
    
    return results, best_params, shap_importance, onset_predictions