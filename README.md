# msc-final-project

Final project as part of requirements for MSc Data Science at Birkbeck College.

This project predicts monthly conflict escalation by region using a combination of
ACLED conflict event data, WFP food prices, CHIRPS/HDX rainfall data, and ConfliBERT
text embeddings of ACLED event notes, feeding an XGBoost classifier.

## Installation

1. Clone the repository and move into it (`cd`).

2. Create a virtual environment (Python 3.13 was used for this project):

    ```
   python -m venv .venv
   ```

   Activate it:

   ```powershell
   # Windows
   .venv\Scripts\activate
   ```

   ```bash
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

## `.env` setup

Create a `.env` file in the project root (it is git-ignored) with the following
variables:

```
ACLED_USERNAME=your_acled_email
ACLED_PASSWORD=your_acled_password
ACLED_TOKEN_URL=https://acleddata.com/oauth/token
HF_TOKEN=your_huggingface_token
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

- **ACLED_USERNAME / ACLED_PASSWORD / ACLED_TOKEN_URL** - required by
  [`ingest/acled_client.py`](ingest/acled_client.py) to obtain an OAuth access token
  from the ACLED API. Register for an account at
  [acleddata.com](https://acleddata.com) to get credentials.
- **HF_TOKEN** - used by `transformers` when downloading the ConfliBERT model
  (`eventdata-utd/ConfliBERT-scr-uncased`) from Hugging Face for text embeddings.
- **MLFLOW_TRACKING_URI** - only required for logging model runs in `run_test_models.ipynb`.
  Not needed for `run_best_model.ipynb`, which doesn't log to MLflow.

Note: raw and cached data (`data/`), the MLflow store (`model/mlflow.db`, `model/mlruns`)
and other derived artifacts are also git-ignored, so a fresh checkout will fetch and
cache data from ACLED/HDX/Hugging Face the first time each pipeline runs.

## Running the pipelines

Both pipelines are run Jupyter notebooks in [`model/`](model/) and should be run with the
project root as the working directory (e.g. `jupyter lab` from the repo root, or via
VS Code's notebook UI with the `.venv` kernel selected).

## Updating constants

[`utils/constants.py`](utils/constants.py) controls which country, commodities, and
date windows the pipelines run against:

- `COUNTRY`  country name to pull ACLED/HDX data for (must resolve via `pycountry`).
- `PRIMARY_COMMODITIES` - food commodities to include from the WFP food price data. This is set as a constant as different countries rely more heavily on different commodities. 
- `TRAIN_START_DATE` / `TRAIN_END_DATE` - the training window.
- `ONSET_START_DATE` / `ONSET_END_DATE` - the near-term test window (used to check
  performance right at the start of an escalation).
- `ACTIVE_START_DATE` / `ACTIVE_END_DATE` - the longer-run test window.
- `FORCE_DOWNLOAD` - when `True`, re-downloads/re-computes all data (ACLED, HDX,
  text embeddings) instead of reading local caches. This should be set to `False` unless you want to delete and redownload everything. 

The file includes a commented-out Ethiopia configuration as an example of switching
countries. When switching `COUNTRY`, note that region name cleanup in
[`utils/name_mapping.py`](utils/name_mapping.py) (`SUDAN_STATE_MAPPING`) is currently
hard-coded for Sudan's admin1 names - a new country may need its own mapping added
there if the HDX and ACLED region names don't already match. #TODO sort thins

## Model architecture

Each pipeline builds a monthly, per-region panel dataset and trains an
`xgboost.XGBClassifier` to predict `target_escalation` (whether conflict events in a
region exceed `k` standard deviations above its rolling mean the following month):

- **Baseline data** - ACLED conflict event counts and rolling statistics, grouped by
  either `event_type` or `sub_event_type`.
- **Optional features** (selected per run via `data_sources`):
  - `food` - WFP retail food prices for the commodities in `PRIMARY_COMMODITIES`.
  - `rain` - CHIRPS/HDX subnational rainfall.
  - `text` - mean-pooled ConfliBERT embeddings of ACLED event `notes` column, optionally
    restricted to conflict-only events (`conflict_only`), optionally reduced with
    PCA fit on the training set only (`use_pca`) to avoid leakage.
- **Splitting** - data is split by date into train / onset-test / active-test windows
  (from `utils/constants.py`). Cross-validation within the training window uses
  grouped, time-series-respecting folds (`utils/cross_validation.py`).
- **Class imbalance** - handled via `scale_pos_weight` computed from the training
  set's escalation ratio.
- **Threshold selection** - the classification threshold is chosen to maximise F1 on
  out-of-fold training predictions, then applied to both test windows.
- **Hyperparameter search** - `RandomizedSearchCV` over an XGBoost parameter grid,
  scored on average precision, orchestrated by
  [`model/train_models.py`](model/train_models.py)'s `train_evaluate_model`.

### `run_best_model.ipynb` - evaluate the winning configuration(s) #TODO

Runs one or more fixed, hand-picked configurations directly (skipping the
hyperparameter search) using `train_evaluate_model(..., best_params=True)`, and does
**not** require MLflow or `MLFLOW_TRACKING_URI`.

It currently defines two configs:

- **Model A** - a numeric-only baseline (ACLED + food + rain, no text).
- **Model B** - the winning configuration (ACLED + food + conflict-only text
  embeddings with PCA), as noted in the notebook's markdown header.

Run the notebook cells in order. For each model it will:

- Load/merge the relevant data sources and fit XGBoost with the config's fixed
  hyperparameters.
- Print evaluation metrics (`results`), the fitted hyperparameters (`best_params`),
  and top SHAP feature importances (`shap_importance`).
- Compute per-row onset predictions and print a pre/post war-onset (15 April 2023)
  precision/recall breakdown via `summarise(...)`.

To evaluate a different configuration, copy one of the `model_*_config` /
`model_*_xgb_params` cell pairs, adjust the values (e.g. the winning params surfaced
by `run_test_models.ipynb`), and call `run_model(your_config, your_params)`.


### `run_test_models.ipynb` - hyperparameter/config search

Sweeps many combinations of feature sets, `k`, `event_col`, PCA and cross-validation
fold counts, running `RandomizedSearchCV` for each and logging results to MLflow.

This should only be used to find the best parameter selection and test the different configs. **It takes a very long time to complete as it runs and logs hundreds of configs and params**

1. Start a local MLflow tracking server (matching the comment in the notebook and
   the `MLFLOW_TRACKING_URI` you set in `.env`):

   ```
   mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
   ```

2. Run the notebook cells in order. It will:
   - Load `model/completed_runs.txt` to skip configs that have already finished
     (safe to stop and resume).
   - Fetch/cache the ACLED, food, rain and text data needed for each feature
     combination via `get_clean_combined_data`.
   - For each remaining config, run `train_evaluate_model` with `best_params=False`
     (i.e. it performs the `RandomizedSearchCV`), log params/metrics to MLflow, and
     append a backup row to `evaluation/{COUNTRY}_results.csv` (so results survive
     even if MLflow logging fails, which it has before!).
   - Append each finished run name to `model/completed_runs.txt`.

3. Inspect results either in `evaluation/{COUNTRY}_results.csv` or in the MLflow UI:

   ```
   mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
   ```

To re-run a config from scratch, delete its entry from `model/completed_runs.txt`
(and, if you want completely fresh data, set `FORCE_DOWNLOAD = True` in
`utils/constants.py`).

