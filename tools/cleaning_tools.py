# tools/cleaning_tools.py
#
# Core building blocks used by the Data Cleaning Agent (agents/cleaning_agent.py).
# By design, this module implements ONLY the following functionalities:
#
#   1. Handle "NAN" / "NULL" placeholder values (treat them as real missing values)
#   2. Detect missing values
#   3. Fill missing values
#   4. Detect duplicate rows
#   5. Remove duplicate rows
#   6. Detect outliers
#   7. Handle outliers (cap or remove)
#   8. Standardize categorical values (one-hot encoding / mean encoding)
#   9. Remove extra spaces
#   10. Remove empty rows & columns
#
# Every function works on the whole column at once (vectorized pandas
# operations) instead of looping row by row in Python, so this stays fast
# even on datasets with millions of rows ("large scale" data).

import numpy as np
import pandas as pd

# Text values that commonly appear in raw data to mean "no value", but that
# pandas does NOT automatically recognize as missing on its own.
MISSING_TOKENS = {"nan", "null", "na", "n/a", "none", "?", "-", "--", ""}

# pandas 3.x introduced a new default "str" dtype for text columns (older
# pandas used "object"). select_dtypes(include="object") still matches the
# new "str" dtype for now (with a deprecation warning), but pandas <3 raises
# an error if "str" is passed to select_dtypes. We check once, up front,
# which option this environment supports, so text-column detection works
# correctly (and quietly) on both old and new pandas.
try:
    pd.DataFrame({"_probe": ["x"]}).select_dtypes(include=["object", "str"])
    _TEXT_DTYPES = ["object", "str"]
except TypeError:
    _TEXT_DTYPES = ["object"]


def _numeric_columns(df: pd.DataFrame, columns=None) -> list:
    """Helper: return the numeric columns of df, optionally filtered to `columns`."""
    cols = df.select_dtypes(include=np.number).columns
    if columns is not None:
        cols = [c for c in cols if c in columns]
    return list(cols)


def _text_columns(df: pd.DataFrame, columns=None) -> list:
    """Helper: return the text/categorical columns of df, optionally filtered to `columns`."""
    cols = df.select_dtypes(include=_TEXT_DTYPES).columns
    if columns is not None:
        cols = [c for c in cols if c in columns]
    return list(cols)


# ---------------------------------------------------------------------------
# 1. Handle "NAN" / "NULL" placeholder values
# ---------------------------------------------------------------------------

def handle_placeholder_missing_values(df: pd.DataFrame, extra_tokens=None) -> tuple:
    """
    Some datasets store missing data as literal text like "NAN", "NULL",
    "N/A", "?", "None", etc. instead of a real empty cell. Pandas does not
    treat these strings as missing by default, so we replace them with a
    real NaN here, before doing anything else.
    """
    tokens = set(MISSING_TOKENS)
    if extra_tokens:
        tokens.update(t.lower() for t in extra_tokens)

    replaced_counts = {}
    for col in _text_columns(df):
        # Normalize each value (trim spaces, lowercase) before comparing it
        # against our list of "means nothing" tokens.
        normalized = df[col].astype(str).str.strip().str.lower()
        mask = normalized.isin(tokens) & df[col].notna()
        count = int(mask.sum())
        if count:
            df.loc[mask, col] = np.nan  # turn the placeholder text into a real missing value
            replaced_counts[col] = count
    return df, replaced_counts


# ---------------------------------------------------------------------------
# 2. Detect missing values
# ---------------------------------------------------------------------------

def detect_missing_values(df: pd.DataFrame) -> dict:
    """
    Report how many values are missing in each column, and what percentage
    of that column they represent. Only columns that actually have missing
    values are included in the result.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    result = {}
    for col in df.columns:
        count = int(missing_counts[col])
        if count > 0:
            result[col] = {
                "count": count,
                "percent": round(count / total_rows * 100, 2) if total_rows else 0.0,
            }
    return result


# ---------------------------------------------------------------------------
# 3. Fill missing values
# ---------------------------------------------------------------------------

def fill_missing_values(df: pd.DataFrame, strategy: str = "auto", columns=None) -> tuple:
    """
    Fill in missing values so the dataset has no gaps left.

    strategy:
        "auto"   - numeric columns use the median, text columns use the mode
                   (the most frequently occurring value)
        "mean"   - numeric columns use the column average
        "median" - numeric columns use the column median
        "mode"   - any column uses its most frequent value
    """
    target_cols = columns if columns is not None else list(df.columns)
    details = {}

    for col in target_cols:
        if col not in df.columns:
            continue
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue  # nothing to fill in this column

        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if strategy == "mean" and is_numeric:
            fill_value = df[col].mean()
            method = "mean"
        elif strategy in ("median", "auto") and is_numeric:
            fill_value = df[col].median()
            method = "median"
        else:
            # Text columns (and the "mode" strategy) fall back to the most
            # common value in the column.
            mode_values = df[col].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            method = "mode"

        df[col] = df[col].fillna(fill_value)
        details[col] = {"missing": missing, "method": method}

    return df, details


# ---------------------------------------------------------------------------
# 4 & 5. Detect / remove duplicate rows
# ---------------------------------------------------------------------------

def detect_duplicate_rows(df: pd.DataFrame) -> int:
    """Count how many rows are an exact duplicate of an earlier row."""
    return int(df.duplicated().sum())


def remove_duplicate_rows(df: pd.DataFrame) -> tuple:
    """Remove duplicate rows, keeping the first occurrence of each one."""
    before = len(df)
    df = df.drop_duplicates(keep="first")
    removed = before - len(df)
    return df, removed


# ---------------------------------------------------------------------------
# 6 & 7. Detect / handle outliers
# ---------------------------------------------------------------------------

def _iqr_bounds(series: pd.Series, factor: float = 1.5) -> tuple:
    """
    Helper: compute the "normal range" for a numeric column using the IQR
    (Interquartile Range) method. Anything below Q1 - factor*IQR or above
    Q3 + factor*IQR is considered an outlier.
    """
    q1 = series.quantile(0.25)  # 25th percentile
    q3 = series.quantile(0.75)  # 75th percentile
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def detect_outliers(df: pd.DataFrame, columns=None, factor: float = 1.5) -> dict:
    """Detect outliers in every numeric column using the IQR method."""
    result = {}
    for col in _numeric_columns(df, columns):
        lower, upper = _iqr_bounds(df[col], factor)
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count:
            result[col] = {"count": count, "lower": float(lower), "upper": float(upper)}
    return result


def handle_outliers(df: pd.DataFrame, columns=None, factor: float = 1.5, strategy: str = "cap") -> tuple:
    """
    Fix outliers detected by the IQR method.

    strategy:
        "cap"    - clip outlier values down/up to the nearest normal bound.
                   No rows are lost, values are just brought into range.
        "remove" - drop the whole row if any of its values are an outlier.
    """
    if strategy == "remove":
        before = len(df)
        keep_mask = pd.Series(True, index=df.index)
        for col in _numeric_columns(df, columns):
            lower, upper = _iqr_bounds(df[col], factor)
            # Keep rows inside the normal range, or where the value is missing.
            keep_mask &= df[col].between(lower, upper) | df[col].isna()
        df = df[keep_mask]
        return df, {"rows_removed": before - len(df)}

    # Default strategy: cap ("winsorize") the values instead of dropping rows.
    capped_counts = {}
    for col in _numeric_columns(df, columns):
        lower, upper = _iqr_bounds(df[col], factor)
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count:
            df[col] = df[col].clip(lower, upper)
            capped_counts[col] = count
    return df, capped_counts


# ---------------------------------------------------------------------------
# 8. Standardize categorical values (one-hot encoding / mean encoding)
# ---------------------------------------------------------------------------

def standardize_categorical_values(
    df: pd.DataFrame,
    columns=None,
    method: str = "onehot",
    target_column: str = None,
    max_categories: int = 20,
) -> tuple:
    """
    Convert text/categorical columns into a numeric form so the data can be
    used for statistics, charts, and modeling. The original text column is
    REPLACED by the new numeric column(s) it gets encoded into (standard
    encoding behavior - e.g. "sex" becomes "sex_Female"/"sex_Male", and the
    text "sex" column is dropped).

    method:
        "onehot" - create one 0/1 column per category (e.g. "City_Boston").
                   Skips columns with more than `max_categories` unique
                   values (left as plain text), since that would create too
                   many new columns.
        "mean"   - replace each category with the average value of
                   `target_column` for that category (a.k.a. target/mean
                   encoding). Requires `target_column` to be provided, since
                   there is no way to "average" without knowing what to
                   average. Columns are left as plain text if no valid
                   target_column is given.
    """
    target_cols = _text_columns(df, columns)
    new_columns = []

    if method == "mean":
        if not target_column or target_column not in df.columns:
            # Without a target column there is nothing to average against,
            # so we do nothing rather than guess.
            return df, new_columns
        for col in target_cols:
            if col == target_column:
                continue
            # For every row, look up the average of `target_column` among
            # all rows sharing that same category.
            df[f"{col}_mean_encoded"] = df.groupby(col)[target_column].transform("mean")
            new_columns.append(f"{col}_mean_encoded")
            df = df.drop(columns=[col])  # replace the original text column
        return df, new_columns

    # Default: one-hot encoding.
    for col in target_cols:
        if df[col].nunique(dropna=True) > max_categories:
            continue  # too many categories - skip to avoid creating hundreds of columns
        dummy_columns = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy_columns], axis=1)
        df = df.drop(columns=[col])  # replace the original text column
        new_columns.extend(dummy_columns.columns.tolist())

    return df, new_columns


# ---------------------------------------------------------------------------
# 9. Remove extra spaces
# ---------------------------------------------------------------------------

def remove_extra_spaces(df: pd.DataFrame, columns=None) -> tuple:
    """
    Trim leading/trailing spaces and collapse repeated inner spaces (e.g.
    "  New   York " -> "New York") in every text column.
    """
    changed = {}
    for col in _text_columns(df, columns):
        cleaned = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        mask = df[col].notna() & (cleaned != df[col])
        count = int(mask.sum())
        if count:
            # `.where(isna, cleaned)` keeps real NaNs as NaN and only swaps
            # in the cleaned text for the non-missing values.
            df[col] = df[col].where(df[col].isna(), cleaned)
            changed[col] = count
    return df, changed


# ---------------------------------------------------------------------------
# 10. Remove empty rows & columns
# ---------------------------------------------------------------------------

def remove_empty_rows(df: pd.DataFrame) -> tuple:
    """Remove rows where every single column is missing."""
    before = len(df)
    df = df.dropna(how="all")
    return df, before - len(df)


def remove_empty_columns(df: pd.DataFrame) -> tuple:
    """Remove columns where every single row is missing."""
    empty_cols = list(df.columns[df.isna().all(axis=0)])
    df = df.drop(columns=empty_cols)
    return df, empty_cols
