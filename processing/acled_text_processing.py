import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from utils.dates import (
    END_DATE,
    TRAIN_START_DATE,
    WARMUP_START_DATE_1_MONTH,
    get_padded_index,
)
from utils.logger import get_logger

logger = get_logger("Text processing")

load_dotenv()

def remove_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes date references from the 'notes' column and drops duplicate entries.

    Uses regex to identify and strip out date prefixes, month/day ranges,
    and years from the beginning of the text notes. The regex pattern was produced by AI.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'notes' column.

    Returns:
        pd.DataFrame: A cleaned DataFrame with duplicates removed and a new
            'notes_cleaned' column containing the parsed text.
    """
    # 1. Define the building blocks
    months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    days = r"\d{1,2}(?:st|nd|rd|th)?"
    day_range = rf"{days}(?:\s*-\s*{days})?"
    year = r"(?:\s+\d{4})?"

    # 2. Date combinations
    date_format = rf"(?:{day_range}\s+{months}{year}|{months}\s+{day_range}{year})"
    between_format = rf"Between\s+{date_format}\s+and\s+{date_format}"
    prefixes = r"(?:On\s+or\s+around|On|Around|Over)\s+"

    # 3. Assemble the final pattern
    # (?i) makes it case-insensitive
    # ^\s* targets ONLY the beginning of the text (and ignores any accidental leading spaces)
    # (?:\s*[,.]\s*|\s+|$) cleans up trailing commas, periods, spaces, or the end of the string
    final_pattern = rf"(?i)^\s*(?:{between_format}|(?:{prefixes})?{date_format})(?:\s*[,.]\s*|\s+|$)"

    original_len = len(df)
    df_cleaned = (
        df.dropna(subset=["notes"])
        .drop_duplicates(subset=["admin1", "year_month", "notes"])
        .copy()
    )
    new_len = len(df_cleaned)
    logger.info(f"Duplicates in notes column deleted: {original_len - new_len}")

    # 4. Apply to your dataframe
    df_cleaned["notes_cleaned"] = df_cleaned["notes"].str.replace(
        final_pattern, "", regex=True
    )
    return df_cleaned


def check_max_tokens(tokenizer: PreTrainedTokenizer, df: pd.DataFrame) -> None:
    """
    Verifies that no cleaned notes exceed the model's maximum token limit (512). If it does, raise an error.

    Args:
        tokenizer (PreTrainedTokenizer): The HuggingFace tokenizer used for encoding.
        df (pd.DataFrame): The DataFrame containing the 'notes_cleaned' column.

    Raises:
        ValueError: If any entry in 'notes_cleaned' exceeds 512 tokens.
    """
    token_lengths = [
        len(tokenizer.encode(note)) for note in df["notes_cleaned"].dropna()
    ]

    logger.info(f"Average tokens: {sum(token_lengths) / len(token_lengths)}")
    logger.info(f"Max tokens: {max(token_lengths)}")

    exceeding_count = sum(1 for t in token_lengths if t > 512)

    if exceeding_count > 0:
        raise ValueError(
            f"Token limit exceeded: {exceeding_count} notes have more than 512 tokens."
        )


def get_embedding(
    text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel
) -> list[float]:
    """
    Generates a mean-pooled embedding for a single text string using ConfliBERT.

    Args:
        text (str): The input text to be embedded.
        tokenizer (PreTrainedTokenizer): The tokenizer matching the model.
        model (PreTrainedModel): The pre-trained language model.

    Returns:
        List[float]: A list of floats representing the mean-pooled embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)  # Model is now safely passed in

    # Mean pooling across the sequence length dimension
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

#
# def get_monthly_regional_embeddings(
#     df: pd.DataFrame, tokenizer: PreTrainedTokenizer, model: PreTrainedModel
# ) -> pd.DataFrame:
#     """
#     Calculates document embeddings and aggregates them by region ('admin1') and month.
#
#     Note:
#         Assumes `train_start_date` and `end_date` are globally available (e.g., imported
#         from `utils.dates`).
#
#     Args:
#         df (pd.DataFrame): The DataFrame containing 'notes_cleaned', 'admin1', and 'year_month'.
#         tokenizer (PreTrainedTokenizer): The tokenizer to use for embeddings.
#         model (PreTrainedModel): The model to use for embeddings.
#
#     Returns:
#         pd.DataFrame: A DataFrame with mean embeddings grouped by region and month,
#             filtered by the global training date constraints.
#     """
#     logger.warning(
#         f"Calculating embeddings for {len(df)} rows. This might take a while."
#     )
#
#     df["notes_embeddings"] = None
#     df["notes_embeddings"] = df["notes_cleaned"].apply(
#         get_embedding, args=(tokenizer, model)
#     )
#
#     # Expand the list of embeddings into separate columns
#     embedding_cols = pd.DataFrame(df["notes_embeddings"].tolist(), index=df.index)
#     embedding_cols.columns = [f"emb_{i}" for i in range(embedding_cols.shape[1])]
#
#     df_expanded = pd.concat([df[["admin1", "year_month"]], embedding_cols], axis=1)
#     df_expanded = df_expanded.loc[:, ~df_expanded.columns.duplicated(keep="first")]
#
#     # Group by region and month, then average the embeddings
#     monthly_region_embeddings = (
#         df_expanded.groupby(["admin1", "year_month"]).mean().reset_index()
#     )
#
#     monthly_region_embeddings["year_month"] = pd.PeriodIndex(monthly_region_embeddings["year_month"], freq="M")
#     monthly_region_embeddings = monthly_region_embeddings.rename(columns={"admin1": "region"})
#     # start_period = pd.Period(train_start_date, freq="M")
#     # padded_start = start_period - 1
#     #
#     # df_final = monthly_region_embeddings[
#     #     (monthly_region_embeddings["year_month"] >= padded_start)
#     #     & (monthly_region_embeddings["year_month"] <= end_date)
#     #     ].copy()
#
#     return monthly_region_embeddings
#

def get_monthly_regional_embeddings(
        df: pd.DataFrame, tokenizer: PreTrainedTokenizer, model: PreTrainedModel
) -> pd.DataFrame:
    """
    Calculates document embeddings and aggregates them by region ('admin1') and month.
    """
    start_period = pd.Period(TRAIN_START_DATE, freq="M")
    padded_start = start_period - 1
    end_period = pd.Period(END_DATE, freq="M")

    df = df.copy()
    df["year_month"] = pd.to_datetime(df["year_month"]).dt.to_period("M")

    df_filtered = df[
        (df["year_month"] >= padded_start) & (df["year_month"] <= end_period)
        ].copy()

    logger.warning(
        f"Calculating embeddings for {len(df_filtered)} rows. This might take a while."
    )

    df_filtered["notes_embeddings"] = df_filtered["notes_cleaned"].apply(
        get_embedding, args=(tokenizer, model)
    )

    embedding_cols = pd.DataFrame(df_filtered["notes_embeddings"].tolist(), index=df_filtered.index)
    embedding_cols.columns = [f"emb_{i}" for i in range(embedding_cols.shape[1])]

    df_expanded = pd.concat([df_filtered[["admin1", "year_month"]], embedding_cols], axis=1)
    df_expanded = df_expanded.loc[:, ~df_expanded.columns.duplicated(keep="first")]

    # Group by region and month, then average the embeddings
    monthly_region_embeddings = (
        df_expanded.groupby(["admin1", "year_month"]).mean().reset_index()
    )

    monthly_region_embeddings = monthly_region_embeddings.rename(columns={"admin1": "region"})

    return monthly_region_embeddings

def apply_pca_train_only(
    train_df: pd.DataFrame,
    onset_df: pd.DataFrame,
    active_df: pd.DataFrame,
    predictor_cols: list[str],
    variance_threshold: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCA, list[str]]:
    """
    Fits a PCA model on the training dataset to prevent data leakage,
    then transforms the training, onset, and active datasets.

    Args:
        train_df (pd.DataFrame): Training dataset containing embedding columns.
        onset_df (pd.DataFrame): Onset dataset containing embedding columns.
        active_df (pd.DataFrame): Active dataset containing embedding columns.
        predictor_cols (List[str]): List of all predictor column names.
        variance_threshold (float, optional): The target cumulative variance
            to retain. Defaults to 0.90.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCA, List[str]]:
            A tuple containing:
            - The transformed train DataFrame.
            - The transformed onset DataFrame.
            - The transformed active DataFrame.
            - The fitted PCA model object.
            - The updated list of predictor column names.
    """
    emb_cols = [c for c in predictor_cols if c.startswith("emb_")]
    non_emb_cols = [c for c in predictor_cols if not c.startswith("emb_")]

    pca = PCA(n_components=variance_threshold, random_state=7)

    X_train_emb_pca = pca.fit_transform(train_df[emb_cols])
    X_onset_emb_pca = pca.transform(onset_df[emb_cols])
    X_active_emb_pca = pca.transform(active_df[emb_cols])

    logger.info(
        f"PCA fit on train embeddings only: {len(emb_cols)} raw dims -> "
        f"{pca.n_components_} components for {variance_threshold:.0%} variance."
    )

    pc_names = [f"PC{i + 1}" for i in range(pca.n_components_)]

    def rebuild(df: pd.DataFrame, emb_pca_array: np.ndarray) -> pd.DataFrame:
        """Helper to stitch non-embedding columns back with PCA components."""
        pcs = pd.DataFrame(emb_pca_array, columns=pc_names, index=df.index)
        return pd.concat(
            [df[non_emb_cols].reset_index(drop=True), pcs.reset_index(drop=True)],
            axis=1,
        )

    X_train = rebuild(train_df, X_train_emb_pca)
    X_onset = rebuild(onset_df, X_onset_emb_pca)
    X_active = rebuild(active_df, X_active_emb_pca)

    final_predictor_cols = non_emb_cols + pc_names
    return X_train, X_onset, X_active, pca, final_predictor_cols


def full_dataset(df: pd.DataFrame, all_regions, all_months) -> pd.DataFrame:
    """
    Creates a balanced panel dataset for all regions and months, imputes missing
    values with global means, and lags embeddings by one month.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'region', 'year_month',
            and embedding features.

    Returns:
        pd.DataFrame: A fully balanced panel DataFrame with lagged embedding features.
    """
    df = df.copy()

    df_padded, final_regions, padded_months, = get_padded_index(
        df, all_regions, all_months, WARMUP_START_DATE_1_MONTH
    )

    full_padded_index = pd.MultiIndex.from_product(
        [final_regions, padded_months], names=["region", "year_month"]
    )

    df_grouped = (
        df_padded.set_index(["region", "year_month"])
        .reindex(full_padded_index)
        .reset_index()
    )
    
    emb_cols = [c for c in df_grouped.columns if c.startswith("emb_")]

    # Create a col to flag if there are actually 0 events
    df_grouped["has_acled_event"] = df_grouped[emb_cols[0]].notna().astype(int)

    df_grouped[emb_cols] = df_grouped[emb_cols].fillna(0.0)

    df_grouped = df_grouped.sort_values(by=["region", "year_month"])

    df_grouped[emb_cols] = df_grouped.groupby("region")[emb_cols].shift(1)
    df_grouped["has_acled_event"] = df_grouped.groupby("region")["has_acled_event"].shift(1)

    # Shifting adds NaNs back in so need to fill again
    df_grouped[emb_cols] = df_grouped[emb_cols].fillna(0.0)
    df_grouped["has_acled_event"] = (
        df_grouped["has_acled_event"].fillna(0).astype(int)
    )
    df_grouped = df_grouped[df_grouped["year_month"] >= pd.Period(TRAIN_START_DATE, freq="M")].copy() # Remove buffered data

    return df_grouped

def get_clean_data(
    calculate: bool = False, df: pd.DataFrame | None = None, all_regions = None, all_months = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Main orchestration function to clean data, generate or load embeddings,
    balance the panel dataset, and extract predictor columns.

    Args:
        calculate (bool, optional): If True, computes embeddings from scratch using
            ConfliBERT. If False, loads pre-computed embeddings from disk. Defaults to False.
        df (Optional[pd.DataFrame], optional): The raw DataFrame. Required if `calculate`
            is True. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing the final processed
            DataFrame and a list of predictor column names.
    """
    if calculate:
        if df is None:
            raise ValueError("DataFrame 'df' must be provided when calculate=True")

        model_name = "eventdata-utd/ConfliBERT-scr-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()  # Tells model I am not training

        df_regex = remove_dates(df)
        check_max_tokens(tokenizer, df_regex)
        df_monthly = get_monthly_regional_embeddings(df_regex, tokenizer, model)

    else:
        df_monthly = pd.read_pickle("../data/monthly_regional_embeddigs.pkl")
        df_monthly = df_monthly.rename(columns={"admin1": "region"}) # TODO remove this when reran
        if not pd.api.types.is_period_dtype(df_monthly["year_month"]):
            df_monthly["year_month"] = pd.to_datetime(df_monthly["year_month"]).dt.to_period("M")
    df_final = full_dataset(df_monthly, all_regions, all_months)
    predictor_cols = [c for c in df_final.columns if c.startswith("emb_")]
    predictor_cols.append("has_acled_event")

    return df_final, predictor_cols