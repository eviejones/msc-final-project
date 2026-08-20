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

def read_model_reports(files):
    result_list = []
    shap_list = []
    onset_list = []
    
    for f in files:
        result, best_params, shap_importance, onset_predictions = open_model_report(f)
        
        model_name = result['model'].iloc[0] 
        
        shap_importance['model'] = model_name
        onset_predictions['model'] = model_name
        
        result_list.append(result)
        shap_list.append(shap_importance)
        onset_list.append(onset_predictions)
        
    all_results = pd.concat(result_list)
    all_shap = pd.concat(shap_list)
    all_onset = pd.concat(onset_list)
    
    df_melted = all_results.melt(
        id_vars=['model'], 
        var_name='metric', 
        value_name='score'
    )
    df_melted['score'] = pd.to_numeric(df_melted['score'], errors='coerce')
    
    return df_melted, all_shap, all_onset