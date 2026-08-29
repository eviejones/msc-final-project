# msc-final-project

Final project as part of requirements for MSc Data Science at Birkbeck College.

This project predicts monthly conflict escalation by region using a combination of
ACLED conflict event data, WFP food prices, CHIRPS/HDX rainfall data, and ConfliBERT
text embeddings of ACLED event notes, feeding an XGBoost classifier.

The purpose of the project is to evaluate whether adding text to a structural baseline model (`Model A`) improves the performance. Four variants of of the text models (`Model B`) are referenced throughout (all-text corpus with and without PCA, conflct-only text corpus with and without PCA).

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
  
4. Install as an editable package

  ```
  pip install -e .
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
  Not needed for `run_best_model.py`, which doesn't log to MLflow.

Note: raw and cached data (`data/`), the MLflow store (`models/mlflow.db`, `models/mlruns`)
and other derived artifacts are also git-ignored, so a fresh checkout will fetch and
cache data from ACLED/HDX/Hugging Face the first time each pipeline runs.

## Updating constants

[`utils/constants.py`](utils/constants.py) controls which country, commodities, and
date windows the pipelines run against. Only change the constants if you want to run the pipeline on another country.

- `COUNTRY`  country name to pull ACLED/HDX data for (must resolve via `pycountry`). The test case uses 'Sudan'.
- `PRIMARY_COMMODITIES` - food commodities to include from the WFP food price data. This is set as a constant as different countries rely more heavily on different commodities.
- `TRAIN_START_DATE` / `TRAIN_END_DATE` - the training window. For Sudan this is the five years before the 2023 Civil War. 
- `ONSET_START_DATE` / `ONSET_END_DATE` - the near-term test window (used to check
  performance right at the start of an escalation). For Sudan this is the whole of 2023, which includes the start of the Civil War in April.
- `ACTIVE_START_DATE` / `ACTIVE_END_DATE` - the longer-run test window for active war. For Sudan this is 2024-2025. 
- `FORCE_DOWNLOAD` - when `True`, re-downloads/re-computes all data (ACLED, HDX,
  text embeddings) instead of reading local caches. This should be set to `False` unless you want to delete and redownload all data. 

The file includes a commented-out Ethiopia configuration as an example of switching
countries. When switching `COUNTRY`, region name cleanup in
[`utils/name_mapping.py`](utils/name_mapping.py) fuzzy-matches WFP/HDX admin1 names
against the canonical ACLED region names, so most countries need no extra work. If a
region name differs too much for fuzzy matching to resolve (e.g. an outright rename),
add an entry for the country to `STATE_NAME_OVERRIDES` in that file - any override or
fuzzy match applied is logged so mismatches can be reviewed.

## Model architecture

Each pipeline builds a monthly, per-region panel dataset and trains an
`xgboost.XGBClassifier` to predict `target_escalation` (whether conflict events in a
region exceed `k` standard deviations above its rolling mean the following month):

- **Baseline data** - ACLED conflict event counts and rolling statistics, grouped by
  either `event_type` or `sub_event_type`.
- **Optional features** (selected per run via `data_sources`):
  - `food` - WFP retail food prices for the commodities in `PRIMARY_COMMODITIES`.
  - `rain` - CHIRPS/HDX subnational rainfall, 3 month anomalies.
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
  [`models/train_models.py`](models/train_models.py)'s `train_evaluate_model`.


## Data 

As the data ingest and preparation (particularly ConfliBERT embeddings) can take a long time, all data required for the Sudan test case is saved in the data folder. This means that the pipeline reads the saved `csv` and `pkl` files. If you want to run the full ingestion and embedding from the start change `FORCE_DOWNLOAD` in 
[`utils/constants.py`](utils/constants.py)  to `True`. 


## Notebooks
This project has three main notebooks.

### `01_run_test_models.ipynb` - hyperparameter/config search

Sweeps many combinations of feature sets, `k`, `event_col`, PCA and cross-validation
fold counts, running `RandomizedSearchCV` for each and logging results to MLflow.

This should only be used to find the best parameter selection and test the different configs. **It takes a very long time to complete as it runs and logs hundreds of configs and params**

1. Start a local MLflow tracking server (matching the comment in the notebook and
   the `MLFLOW_TRACKING_URI` you set in `.env`):

   ```
   mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
   ```
2. Run the notebook cells in order. It will:

   - Load `models/completed_runs.txt` to skip configs that have already finished
     (safe to stop and resume).
   - Fetch/cache the ACLED, food, rain and text data needed for each feature
     combination via `get_clean_combined_data`.
   - For each remaining config, run `train_evaluate_model` with `best_params=False`
     (i.e. it performs the `RandomizedSearchCV`), log params/metrics to MLflow, and
     append a backup row to `evaluation/{COUNTRY}_results.csv` (so results survive
     even if MLflow logging fails, which it has before!).
   - Append each finished run name to `models/completed_runs.txt`.
3. Inspect results either in `evaluation/{COUNTRY}_results.csv` or in the MLflow UI:

   ```
   mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
   ```

To re-run a config from scratch, delete its entry from `models/completed_runs.txt`
(and, if you want completely fresh data, set `FORCE_DOWNLOAD = True` in
`utils/constants.py`).

### `02_methodology_decisions.ipynb` - overview of methodology decisions made 

This notebook goes through some of the key decisions made and why. It is primarily to help inform the full project write up but can be useful for to understand the model.

### `03_results.ipynb` - results discussion and analysis

This notebook discusses the key results and findings of the project.

#### `run_best_model.py` - evaluate the winning configuration(s)

This is used inside `03_results.ipynb` but can also be run independently. 

It runs the models for each of `Model A` and `Model B` variants based on a determined configuration. 

It uses `train_evaluate_model(..., best_params=True)`, so rather than running a hyperparameter search, it selects the already decided best paramaters from the `evaluation/{country}_results.csv` and does **not** require MLflow or `MLFLOW_TRACKING_URI`.

The results are saved in the `evaluation/model_reports` folder. It saved:
- Best params
- Results
- SHAP feature importance
- Detailed onset predicted region-month and actual region-month for further comparison.

## A note on virtual environment

The results were ran with the following package versions and hardware. 
============================================================
ENVIRONMENT
============================================================
Python version:      3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
Platform:            macOS-26.6.2-arm64-arm-64bit-Mach-O
Processor:           arm
CPU count:           18
NumPy version:       2.4.6
NumPy default int:   int64
Pandas version:      2.3.3
scikit-learn version:1.9.0
XGBoost version:     3.4.0

Running on a Windows device appears to give slightly different results. 
