# %%
import pandas as pd
import datetime as dt
from ingest.acled_client import Acled
import numpy as np
import xgboost as xgb
import logging
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %%
acled = Acled()
# %% [markdown]
# Train: Data from January 2018 to December 2022. This includes the 2018 Sudanese revolution but excludes the 2023 Civil War.
# Test onset civil war: Data from January 2023 to December 2023 which includes the escalation of the civil war.
# Test active civil war: Data from January 2024 to December 2025 which includes fluctuations in ongoing civil war.
# 
# %%
countries = ["Sudan"]
start_date = "2017-07-01" # TODO validation for 6 month warm up period
end_date = "2024-12-31"

train_start_date = "2018-01-01"
train_end_date = "2022-12-31"

onset_start_date = "2023-01-01"
onset_end_date = "2023-12-31"

active_start_date = "2024-01-01"
active_end_date = "2024-12-31"
# %%
# all_data = acled.get_data(countries, start_date, end_date)
all_data = pd.read_csv("../data/all_data.csv")
# %%
def mark_conflict_events(df: pd.DataFrame) -> pd.DataFrame:
    # 1 = Conflict event (Y)
    # 0 = Non-conflict (used for features)

    acled_subevent_mapping = {
        # BATTLES (Conflict)
        "Armed clash": 1,
        "Government regains territory": 1,
        "Non-state actor overtakes territory": 1,
        # EXPLOSIONS / REMOTE VIOLENCE (Conflict)
        "Air/drone strike": 1,
        "Chemical weapon": 1,
        "Remote explosive/landmine/IED": 1,
        "Shelling/artillery/missile attack": 1,
        "Suicide bomb": 1,
        "Grenade": 1,
        # VIOLENCE AGAINST CIVILIANS (Conflict)
        "Abduction/forced disappearance": 1,
        "Attack": 1,
        "Sexual violence": 1,
        # RIOTS (Conflict)
        "Mob violence": 1,
        "Violent demonstration": 1,
        # PROTESTS (Non-conflict)
        "Excessive force against protesters": 0,
        "Peaceful protest": 0,
        "Protest with intervention": 0,
        # STRATEGIC DEVELOPMENTS (Non-conflict)
        "Agreement": 0,
        "Arrests": 0,
        "Change to group/activity": 0,
        "Disrupted weapons use": 0,
        "Headquarters or base established": 0,
        "Looting/property destruction": 0,
        "Non-violent transfer of territory": 0,
        "Other": 0,
    }
    df["conflict"] = df["sub_event_type"].apply(lambda x: acled_subevent_mapping[x])
    return df #TODO add validation
# %%
def create_regional_monthly_baseline(df, k):
    df = df.copy()
    df_grouped = (
        df.groupby(["admin1", "year_month"])["conflict"]
        .sum()
        .reset_index(name="conflict_event_count")
    )

    # Build full dataset of all regions and months
    all_regions = df["admin1"].unique()
    all_months = pd.period_range(
        df["year_month"].min(), df["year_month"].max(), freq="M"
    )
    full_index = pd.MultiIndex.from_product(
        [all_regions, all_months], names=["admin1", "year_month"]
    )

    df_grouped = (
        df_grouped.set_index(["admin1", "year_month"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["admin1", "year_month"])
    )

    df_grouped = df_grouped.sort_values(by=["admin1", "year_month"])

    # Calculate rolling statistics ending at the previous month (t-1)
    df_grouped["rolling_mean_6m"] = df_grouped.groupby("admin1")[
        "conflict_event_count"
    ].transform(lambda x: x.rolling(window=6, min_periods=6).mean().shift(1))

    df_grouped["rolling_std_6m"] = df_grouped.groupby("admin1")[
        "conflict_event_count"
    ].transform(lambda x: x.rolling(window=6, min_periods=6).std().shift(1))

    df_grouped["escalation_threshold"] = df_grouped["rolling_mean_6m"] + (
        k * df_grouped["rolling_std_6m"]
    )

    # Define the binary target variable (Is current conflict > historical threshold?)
    df_grouped["target_escalation"] = np.where(
        df_grouped["conflict_event_count"] > df_grouped["escalation_threshold"], 1, 0
    )
    logger.info(f"Escalation target set at {k} standard deviations above the mean.")

    return df_grouped
# %%
def pre_process_data(df, k, event_col = "sub_event_type"):
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["year_month"]).dt.to_period("M")
    df = mark_conflict_events(df)

    pivot_df = pd.pivot_table(
        df,
        values="event_id_cnty",
        index=["admin1", "year_month"],
        columns=[event_col],
        aggfunc="count",
        fill_value=0,
    ).reset_index()

    logger.info(f"Data grouped by {event_col}")

    pivot_df.columns = (
        pivot_df.columns.str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    baseline_df = create_regional_monthly_baseline(df,k)

    fatalities_df = (
        df.groupby(["admin1", "year_month"])["fatalities"].sum().reset_index()
    )

    combined_df = pd.merge(
        baseline_df, pivot_df, on=["admin1", "year_month"], how="left")
    combined_df = pd.merge(
        combined_df, fatalities_df, on=["admin1", "year_month"], how="left")

    event_cols = pivot_df.columns.drop(["admin1", "year_month"]).tolist()
    combined_df[event_cols] = combined_df[event_cols].fillna(0)
    combined_df["fatalities"] = combined_df["fatalities"].fillna(0)

    current_event_cols = event_cols + ["fatalities"]
    lagged_event_cols = ["rolling_mean_6m", "rolling_std_6m", "escalation_threshold"]

    combined_df[current_event_cols] = combined_df[current_event_cols].fillna(0)
    combined_df[current_event_cols] = combined_df.groupby("admin1")[current_event_cols].shift(1)

    predictor_cols = current_event_cols + lagged_event_cols
    combined_df[predictor_cols] = combined_df[predictor_cols].fillna(0)

    combined_df = combined_df.rename(columns={"admin1": "region"})
    combined_df = combined_df.sort_values(by=["year_month", "region"]).reset_index(drop=True)
    return combined_df, predictor_cols
# %%
def calculate_conflict_ratio(df):
    count_0 = (df["target_escalation"] == 0).sum()
    count_1 = (df["target_escalation"] == 1).sum()
    ratio = count_0 / count_1

    return {"non-escalation": count_0, "escalation": count_1, "ratio": ratio}
# %%
def split_data(df, predictor_cols, target_col, start_date, end_date):
    split_df = df[
        (df["year_month"] >= start_date)
        & (df["year_month"] <= end_date)
    ].copy()

    y = split_df[target_col].copy()
    X = split_df[predictor_cols].copy()

    return split_df, y, X
# %%
def grouped_timeseries_cv_ids(dates, n_splits=4):
    dates = pd.Series(dates).reset_index(drop=True)
    unique_months = np.sort(dates.unique())

    timeseries_cv = TimeSeriesSplit(n_splits=n_splits)

    for train_month_pos, test_month_pos in timeseries_cv.split(unique_months):
        train_months = unique_months[train_month_pos]
        test_months = unique_months[test_month_pos]

        train_idx = dates[dates.isin(train_months)].index.to_numpy()
        test_idx = dates[dates.isin(test_months)].index.to_numpy()
        yield train_idx, test_idx
# %%
def verify_cv_splits(df, cv_splits, date_column = "year_month"):
    logger.info("Cross-validation testing splits:")
    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        train_dates = df.iloc[train_idx][date_column].unique()
        test_dates = df.iloc[test_idx][date_column].unique()

        train_dates = sorted(train_dates)
        test_dates = sorted(test_dates)
        print(f"--- Fold {fold + 1} ---")
        print(f"Train window: {train_dates[0]} to {train_dates[-1]} ({len(train_idx)} rows)")
        print(f"Test window:  {test_dates[0]} to {test_dates[-1]} ({len(test_idx)} rows)")

        overlap = set(train_dates).intersection(set(test_dates))
        if overlap:
            print(f"Overlapping months: {overlap}")

        if train_dates[-1] >= test_dates[0]:
            print("Training window overlaps or exceeds the test window!")

        print("-" * 30)

# %%
def train_evaluate_model(all_data, params):
    # Process data
    processed_df, predictor_cols = pre_process_data(all_data, params["k"], params["event_col"])

    # Split data
    train_df, y_train, X_train = split_data(processed_df, predictor_cols, "target_escalation", train_start_date, train_end_date)
    onset_df, y_onset, X_onset = split_data(processed_df, predictor_cols, "target_escalation", onset_start_date, onset_end_date)
    active_df, y_active, X_active = split_data(processed_df, predictor_cols, "target_escalation", active_start_date, active_end_date)

    ratios = calculate_conflict_ratio(train_df)

    scale_weight = ratios["non-escalation"] / ratios["escalation"]
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",  # As decided in proposal
        random_state=7,
    )

    grouped_timeseries_cv = list(grouped_timeseries_cv_ids(train_df["year_month"],n_splits=params["n_splits"]))

    verify_cv_splits(train_df, grouped_timeseries_cv)

    param_grid = params.copy()
    del param_grid["k"]
    del param_grid["event_col"]
    del param_grid["n_splits"]

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=grouped_timeseries_cv,
        scoring="average_precision",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Tune threshold on onset (validation/test partition)
    y_pred_proba_onset = best_model.predict_proba(X_onset)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_onset, y_pred_proba_onset)
    f1_scores = (2 * precisions * recalls / (precisions + recalls + 1e-10))[:-1]
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    # Evaluate on onset test set
    y_pred_custom_onset = (y_pred_proba_onset >= optimal_threshold).astype(int)

    # Evaluate on active test set
    y_pred_proba_active = best_model.predict_proba(X_active)[:, 1]
    y_pred_custom_active = (y_pred_proba_active >= optimal_threshold).astype(int)

    # Explicitly request output_dict=True to enable dictionary lookups
    onset_report = classification_report(y_onset, y_pred_custom_onset, output_dict=True, zero_division=0)
    active_report = classification_report(y_active, y_pred_custom_active, output_dict=True, zero_division=0)

    class_key = '1' if '1' in onset_report else 1

    results = {
        "optimal_threshold": f"{optimal_threshold:.4f}",
        # Onset Metrics
        "onset_aupr": f"{average_precision_score(y_onset, y_pred_proba_onset):.4f}",
        "onset_precision_class1": f"{onset_report[class_key]['precision']:.4f}",
        "onset_recall_class1": f"{onset_report[class_key]['recall']:.4f}",
        "onset_f1_class1": f"{onset_report[class_key]['f1-score']:.4f}",
        # Active Metrics
        "active_aupr": f"{average_precision_score(y_active, y_pred_proba_active):.4f}",
        "active_precision_class1": f"{active_report[class_key]['precision']:.4f}",
        "active_recall_class1": f"{active_report[class_key]['recall']:.4f}",
        "active_f1_class1": f"{active_report[class_key]['f1-score']:.4f}"
    }
    return results, best_params
# %%
def print_date_range(df: pd.DataFrame, col_name: str = "year_month") -> None:
  dates = df[col_name].dropna()
  start_date = dates.min()
  end_date = dates.max()

  print(f"Start Date: {start_date}")
  print(f"End Date:   {end_date}")
# %%
all_params = {
    # Previous best params
    "learning_rate": [0.03],
    "n_estimators": [300],
    "subsample": [0.8],
    # Others
    "colsample_bytree": [0.8],
    "max_depth": [3, 5],
    "min_child_weight": [1, 3, 5],
    "max_delta_step": [1, 5],
    "gamma": [0, 1, 5],
    "k": 0.75,
    "event_col": "event_type",
    "n_splits": 4
}
# %%
results, best_params = train_evaluate_model(all_data, all_params)
# %%
print(results)
# %%
print(best_params)
# %%
