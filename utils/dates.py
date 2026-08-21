"""Functions for working with dates, such as validation."""
import pandas as pd

from utils.constants import (
    ACTIVE_END_DATE,
    ACTIVE_START_DATE,
    ONSET_END_DATE,
    ONSET_START_DATE,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
)

WARMUP_START_DATE_6_MONTHS = TRAIN_START_DATE - pd.DateOffset(
    months=6
)  # 6 months for ACLED baseline
WARMUP_START_DATE_1_MONTH = TRAIN_START_DATE - pd.DateOffset(
    months=1
)  # 1 month for other monthly data

END_DATE = ACTIVE_END_DATE  # The end of all data

def validate_date_ranges() -> None:
    """
    Validates the TRAIN/ONSET/ACTIVE date constants defined in utils.constants.

    Checks that each constant is a valid pd.Timestamp, that each window's
    start is not after its end, and that the three windows do not overlap
    each other (a gap between windows is allowed).

    Raises:
        TypeError: If any constant is not a pd.Timestamp.
        ValueError: If any window is backwards, or if two windows overlap.
    """
    periods = [
        ("TRAIN", TRAIN_START_DATE, TRAIN_END_DATE),
        ("ONSET", ONSET_START_DATE, ONSET_END_DATE),
        ("ACTIVE", ACTIVE_START_DATE, ACTIVE_END_DATE),
    ]

    for name, start, end in periods:
        for bound_name, value in ((f"{name}_START_DATE", start), (f"{name}_END_DATE", end)):
            if not isinstance(value, pd.Timestamp):
                raise TypeError(
                    f"{bound_name} must be a pd.Timestamp, got {type(value).__name__}: {value!r}"
                )

    for name, start, end in periods:
        if start > end:
            raise ValueError(
                f"{name}_START_DATE ({start.date()}) is after {name}_END_DATE ({end.date()})."
            )

    if ONSET_START_DATE <= TRAIN_END_DATE:
        raise ValueError(
            f"ONSET_START_DATE ({ONSET_START_DATE.date()}) overlaps or precedes "
            f"TRAIN_END_DATE ({TRAIN_END_DATE.date()})."
        )

    if ACTIVE_START_DATE <= ONSET_END_DATE:
        raise ValueError(
            f"ACTIVE_START_DATE ({ACTIVE_START_DATE.date()}) overlaps or precedes "
            f"ONSET_END_DATE ({ONSET_END_DATE.date()})."
        )


def validate_data_coverage(
    datasets: dict[str, pd.DataFrame],
    start_date=None,
    end_date=None,
    raise_on_failure: bool = True,
) -> list[str]:
    """
    Checks that each processed DataFrame has a row for every month between
    start_date and end_date (inclusive).

    This checks the *processed* output of each get_clean_data() call, not the
    raw source files - raw files are free to cover a different range (e.g.
    food/rainfall pulled past END_DATE) since anything outside the required
    window is dropped during processing. What matters is that no month is
    missing from what actually reaches the model.

    Args:
        datasets: mapping of a label (e.g. "Food prices") to its processed
            DataFrame, each expected to have a 'year_month' column.
        start_date: earliest month required. Defaults to TRAIN_START_DATE.
        end_date: latest month required. Defaults to END_DATE.
        raise_on_failure: if True, raises AssertionError listing every
            problem found. If False, just returns the list of problems
            (empty list = all clear) - useful for calling from a notebook
            cell without halting execution.

    Returns:
        list[str]: descriptions of any problems found (empty if none).
    """
    start_date = pd.to_datetime(start_date or TRAIN_START_DATE)
    end_date = pd.to_datetime(end_date or END_DATE)
    expected_months = pd.period_range(start_date, end_date, freq="M")

    problems = []
    for label, df in datasets.items():
        actual_months = pd.PeriodIndex(df["year_month"].unique())
        missing_months = expected_months.difference(actual_months)
        if len(missing_months) > 0:
            problems.append(
                f"{label}: missing {len(missing_months)} month(s) of processed "
                f"data within {start_date.date()} - {end_date.date()}: "
                f"{list(missing_months)}"
            )

    if problems and raise_on_failure:
        raise AssertionError(
            "Processed data coverage check failed:\n\n" + "\n\n".join(problems)
        )

    return problems


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
        (df["year_month"] >= padded_start)
        & (df["year_month"] <= pd.Period(END_DATE, freq="M"))
    ].copy()

    if all_regions is None:
        all_regions = df_padded["region"].unique()

    if all_months is None:
        all_months = pd.period_range(
            df_padded["year_month"].min(), df_padded["year_month"].max(), freq="M"
        )

    padded_all_months = pd.period_range(padded_start, all_months.max(), freq="M")

    return df_padded, all_regions, padded_all_months
