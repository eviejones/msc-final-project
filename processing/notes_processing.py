
import pandas as pd
from utils.dates import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Train NLP")
logger.setLevel(logging.INFO)

from dotenv import load_dotenv
load_dotenv()
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA

def remove_dates(df):
    ### Used Gemini
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
    df_cleaned = df.drop_duplicates(subset=['notes']).copy()
    new_len = len(df_cleaned)
    logger.info(f"Duplicates in notes column deleted: {original_len - new_len}")

    # 4. Apply to your dataframe
    df_cleaned["notes_cleaned"] = df_cleaned["notes"].str.replace(final_pattern, "", regex=True)
    return df_cleaned

def check_max_tokens(tokenizer, df):

    token_lengths = [len(tokenizer.encode(note)) for note in df["notes_cleaned"].dropna()]

    logger.info(f"Average tokens: {sum(token_lengths) / len(token_lengths)}")
    logger.info(f"Max tokens: {max(token_lengths)}")

    exceeding_count = sum(1 for t in token_lengths if t > 512)

    if exceeding_count > 0:
        raise ValueError(f"Token limit exceeded: {exceeding_count} notes have more than 512 tokens.")

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs) # Model is now safely passed in
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def get_monthly_regional_embeddings(df, tokenizer, model):
    logger.warning(f"Calculating embeddings for {len(df)} rows. This might take a while.")

    df["notes_embeddings"] = None
    df["notes_embeddings"] = df["notes_cleaned"].apply(get_embedding, args=(tokenizer, model))

    embedding_cols = pd.DataFrame(df["notes_embeddings"].tolist(), index=df.index)
    embedding_cols.columns = [f"emb_{i}" for i in range(embedding_cols.shape[1])]

    df_expanded = pd.concat([df[["admin1", "year_month"]], embedding_cols], axis=1)
    df_expanded = df_expanded.loc[:, ~df_expanded.columns.duplicated(keep='first')]

    monthly_region_embeddings = df_expanded.groupby(["admin1", "year_month"]).mean().reset_index()
    df_final = monthly_region_embeddings [(monthly_region_embeddings ["year_month"] >= train_start_date) & (monthly_region_embeddings ["year_month"] <= end_date)].copy()

    return df_final

def apply_pca_train_only(train_df, onset_df, active_df, predictor_cols, variance_threshold=0.90):

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

    pc_names = [f"PC{i+1}" for i in range(pca.n_components_)]

    def rebuild(df, emb_pca_array):
        pcs = pd.DataFrame(emb_pca_array, columns=pc_names, index=df.index)
        return pd.concat([df[non_emb_cols].reset_index(drop=True),
                           pcs.reset_index(drop=True)], axis=1)

    X_train = rebuild(train_df, X_train_emb_pca)
    X_onset = rebuild(onset_df, X_onset_emb_pca)
    X_active = rebuild(active_df, X_active_emb_pca)

    final_predictor_cols = non_emb_cols + pc_names
    return X_train, X_onset, X_active, pca, final_predictor_cols

def full_dataset(df):
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["year_month"]).dt.to_period("M")

    all_regions = df["region"].unique()
    all_months = pd.period_range(
        df["year_month"].min(), df["year_month"].max(), freq="M"
    )

    full_index = pd.MultiIndex.from_product(
        [all_regions, all_months], names=["region", "year_month"]
    )
    df_grouped = (
        df.set_index(["region", "year_month"])
        .reindex(full_index)
        .reset_index()
    )

    emb_cols = [c for c in df_grouped.columns if c.startswith("emb_")]

    global_mean = df[emb_cols].mean()
    df_grouped[emb_cols] = df_grouped[emb_cols].fillna(global_mean) #TODO Check this logic

    df_grouped = df_grouped.sort_values(by=["region", "year_month"])
    df_grouped[emb_cols] = df_grouped.groupby("region")[emb_cols].shift(1)
    return df_grouped #TODO move this function to a util as used by other parts of the scripts

def get_clean_data(calculate=False, df = None):
    if calculate:
        model_name = "eventdata-utd/ConfliBERT-scr-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval() # Tells model I am not training
        df_regex = remove_dates(df)
        check_max_tokens(tokenizer, df_regex)
        df_monthly = get_monthly_regional_embeddings(df_regex, tokenizer, model)
        # df_pcs = get_pcs(df_monthly)
    elif not calculate:
        df_monthly = pd.read_pickle("../data/monthly_regional_embeddigs.pkl")
        df_monthly = df_monthly.rename(columns={"admin1": "region"})
        # df_pcs = pd.read_pickle("../data/acled_pca_components.pkl")
    df_final = full_dataset(df_monthly)
    predictor_cols = [c for c in df_final.columns if c.startswith("emb_")]
    return df_final, predictor_cols
