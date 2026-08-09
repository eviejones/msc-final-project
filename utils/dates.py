import pandas as pd

TRAIN_START_DATE = pd.to_datetime( "2018-01-01")
TRAIN_END_DATE = pd.to_datetime("2022-12-31")

ONSET_START_DATE = pd.to_datetime("2023-01-01")
ONSET_END_DATE = pd.to_datetime("2023-12-31")

ACTIVE_START_DATE = pd.to_datetime("2024-01-01")
ACTIVE_END_DATE = pd.to_datetime("2024-12-31")

WARMUP_START_DATE_6_MONTHS = TRAIN_START_DATE - pd.DateOffset(months=6) # 6 months for ACLED baseline
WARMUP_START_DATE_1_MONTH = TRAIN_START_DATE - pd.DateOffset(months=1) # 1 month for other monthly data
END_DATE = ACTIVE_END_DATE # The end of all data

def get_padded_index(df, all_regions, all_months, warmup_start_date):
    """
    Creates the components for an index that accounts for the 1-month padding
    needed by food, rain and text data.

    As each of the monthly data sets are shifted, we need to have a preceding
    month that is then removed.

    Returns the padded DataFrame, the regions array, the padded months array,
    and the Period object for the true start date.
    """
    padded_start = pd.Period(warmup_start_date, freq="M")

    df_padded = df[
        (df["year_month"] >= padded_start) & (df["year_month"] <= END_DATE)
    ].copy()

    if all_regions is None:
        all_regions = df_padded["region"].unique()

    if all_months is None:
        all_months = pd.period_range(
            df_padded["year_month"].min(), df_padded["year_month"].max(), freq="M"
        )

    padded_all_months = pd.period_range(padded_start, all_months.max(), freq="M")

    return df_padded, all_regions, padded_all_months
