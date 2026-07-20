# Reusable, large-scale-safe building blocks for the data cleaning agent.
# Every function is vectorized (no row-by-row Python loops over the full
# dataset) so they stay fast on datasets with millions of rows. Functions
# that must inspect individual values (mixed-type / fuzzy-spelling checks)
# are bounded to a sample or to the column's unique values instead of every
# row, so cost scales with cardinality, not row count.
import difflib
import hashlib
import re

import numpy as np
import pandas as pd

# Tokens commonly used in raw data to mean "no value" but that pandas
# won't recognize as missing on its own.
MISSING_TOKENS = {
    "null", "nan", "n/a", "na", "none", "nil", "?", "-", "--",
    "missing", "unknown", "undefined", "#n/a", "#na", "",
}

_SAMPLE_ROWS_FOR_TYPE_SCAN = 50_000  # cap for per-value python-level scans

# pandas >=3 defaults text columns to a new "str" dtype instead of "object".
# select_dtypes(include="object") still matches it today for backward
# compatibility (with a deprecation warning), but pandas <3 raises a TypeError
# if "str" is passed to select_dtypes at all. Probe once at import time so
# text-column detection keeps working - and stays warning-free - on both.
try:
    pd.DataFrame({"_probe": ["x"]}).select_dtypes(include=["object", "str"])
    _TEXT_DTYPES = ["object", "str"]
except TypeError:
    _TEXT_DTYPES = ["object"]


def _numeric_columns(df: pd.DataFrame, columns=None) -> list:
    cols = df.select_dtypes(include=np.number).columns
    if columns is not None:
        cols = [c for c in cols if c in columns]
    return list(cols)


def _object_columns(df: pd.DataFrame, columns=None) -> list:
    cols = df.select_dtypes(include=_TEXT_DTYPES).columns
    if columns is not None:
        cols = [c for c in cols if c in columns]
    return list(cols)


def summarize_list(items, max_show: int = 8) -> str:
    """Render a column/name list compactly so reports stay readable on wide datasets."""
    items = list(items)
    if not items:
        return "none"
    shown = ", ".join(str(i) for i in items[:max_show])
    if len(items) > max_show:
        shown += f" (+{len(items) - max_show} more)"
    return shown


# ---------------------------------------------------------------------------
# Missing values / NaN
# ---------------------------------------------------------------------------

def normalize_missing_tokens(df: pd.DataFrame, extra_tokens=None) -> tuple:
    """Replace placeholder strings like 'NULL', 'NaN', '?', 'None' with real NaN."""
    tokens = set(MISSING_TOKENS)
    if extra_tokens:
        tokens.update(t.lower() for t in extra_tokens)

    replaced_counts = {}
    for col in _object_columns(df):
        as_str = df[col].astype(str).str.strip()
        mask = as_str.str.lower().isin(tokens) & df[col].notna()
        count = int(mask.sum())
        if count:
            df.loc[mask, col] = np.nan
            replaced_counts[col] = count
    return df, replaced_counts


def detect_missing_values(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df.isnull().sum()
    return {
        col: {"count": int(counts[col]), "percent": round(float(counts[col]) / total * 100, 2) if total else 0.0}
        for col in df.columns if counts[col] > 0
    }


def detect_nan_values(df: pd.DataFrame) -> dict:
    """Numeric-focused NaN detection (subset of detect_missing_values)."""
    numeric_cols = _numeric_columns(df)
    counts = df[numeric_cols].isna().sum()
    return {col: int(counts[col]) for col in numeric_cols if counts[col] > 0}


def fill_missing_values(df: pd.DataFrame, strategy: str = "auto", columns=None, value=None) -> tuple:
    """strategy: 'auto' (median for numeric, mode for text), 'mean', 'median', 'mode', 'constant'."""
    target_cols = columns if columns is not None else list(df.columns)
    details = {}
    for col in target_cols:
        if col not in df.columns:
            continue
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue
        # Never invent values for datetime columns; leave NaT for the caller to decide.
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if strategy == "constant":
            fill_value = value
            method = f"constant ({value})"
        elif strategy == "mean" and is_numeric:
            fill_value = df[col].mean()
            method = "mean"
        elif strategy in ("median", "auto") and is_numeric:
            fill_value = df[col].median()
            method = "median"
        elif strategy == "mode" or not is_numeric:
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            method = "mode"
        else:
            fill_value = df[col].median() if is_numeric else "Unknown"
            method = "median" if is_numeric else "constant"

        df[col] = df[col].fillna(fill_value)
        details[col] = {"missing": missing, "method": method}
    return df, details


def fill_nan_values(df: pd.DataFrame, columns=None, method: str = "median") -> tuple:
    numeric_cols = columns if columns is not None else _numeric_columns(df)
    return fill_missing_values(df, strategy=method, columns=numeric_cols)


def remove_missing_rows(df: pd.DataFrame, how: str = "any", thresh=None, subset=None) -> tuple:
    before = len(df)
    if thresh is not None:
        df = df.dropna(thresh=thresh, subset=subset)
    else:
        df = df.dropna(how=how, subset=subset)
    return df, before - len(df)


def remove_missing_columns(df: pd.DataFrame, threshold: float = 0.9) -> tuple:
    """Drop columns whose missing fraction exceeds `threshold` (0-1)."""
    if len(df) == 0:
        return df, []
    missing_frac = df.isna().mean()
    drop_cols = list(missing_frac[missing_frac > threshold].index)
    df = df.drop(columns=drop_cols)
    return df, drop_cols


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def detect_duplicate_rows(df: pd.DataFrame, subset=None) -> int:
    return int(df.duplicated(subset=subset).sum())


def remove_duplicate_rows(df: pd.DataFrame, subset=None, keep: str = "first") -> tuple:
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    return df, before - len(df)


def _column_fingerprint(series: pd.Series) -> str:
    # Hash the column's values (not the index) so identical columns collide
    # to the same fingerprint without a full O(n*k^2) transpose/compare.
    values = pd.util.hash_pandas_object(series, index=False).values
    return hashlib.md5(values.tobytes()).hexdigest()


def detect_duplicate_columns(df: pd.DataFrame) -> dict:
    """Returns {kept_column: [duplicate_columns...]} for columns with identical values."""
    buckets = {}
    for col in df.columns:
        fp = _column_fingerprint(df[col])
        buckets.setdefault(fp, []).append(col)

    duplicates = {}
    for cols in buckets.values():
        if len(cols) < 2:
            continue
        keeper, rest = cols[0], cols[1:]
        # Verify equality within the (small) bucket to rule out hash collisions.
        confirmed = [c for c in rest if df[c].equals(df[keeper])]
        if confirmed:
            duplicates[keeper] = confirmed
    return duplicates


def remove_duplicate_columns(df: pd.DataFrame) -> tuple:
    duplicates = detect_duplicate_columns(df)
    to_drop = [c for dupes in duplicates.values() for c in dupes]
    df = df.drop(columns=to_drop)
    return df, to_drop


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def _iqr_bounds(series: pd.Series, factor: float = 1.5) -> tuple:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def detect_outliers(df: pd.DataFrame, columns=None, factor: float = 1.5) -> dict:
    result = {}
    for col in _numeric_columns(df, columns):
        lower, upper = _iqr_bounds(df[col], factor)
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count:
            result[col] = {"count": count, "lower": float(lower), "upper": float(upper)}
    return result


def remove_outliers(df: pd.DataFrame, columns=None, factor: float = 1.5) -> tuple:
    before = len(df)
    keep_mask = pd.Series(True, index=df.index)
    for col in _numeric_columns(df, columns):
        lower, upper = _iqr_bounds(df[col], factor)
        keep_mask &= df[col].between(lower, upper) | df[col].isna()
    df = df[keep_mask]
    return df, before - len(df)


def cap_outliers(df: pd.DataFrame, columns=None, factor: float = 1.5) -> tuple:
    """Winsorize values to the IQR bounds instead of dropping rows."""
    capped_counts = {}
    for col in _numeric_columns(df, columns):
        lower, upper = _iqr_bounds(df[col], factor)
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count:
            df[col] = df[col].clip(lower, upper)
            capped_counts[col] = count
    return df, capped_counts


def replace_outliers_with_median(df: pd.DataFrame, columns=None, factor: float = 1.5, strategy: str = "median") -> tuple:
    replaced_counts = {}
    for col in _numeric_columns(df, columns):
        lower, upper = _iqr_bounds(df[col], factor)
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count:
            replacement = df[col].mean() if strategy == "mean" else df[col].median()
            df.loc[mask, col] = replacement
            replaced_counts[col] = count
    return df, replaced_counts


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

def detect_incorrect_dtypes(df: pd.DataFrame, min_success_rate: float = 0.9, sample_size: int = 200) -> dict:
    """Flag object columns that actually hold numbers or dates."""
    suggestions = {}
    for col in _object_columns(df):
        series = df[col].dropna()
        if series.empty:
            continue

        looks_datey = "date" in str(col).lower() or "time" in str(col).lower()

        # Date/time-named columns are checked first so e.g. an all-digit
        # "signup_date" (20230101) isn't misclassified as plain numeric.
        if looks_datey:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() >= min_success_rate:
                suggestions[col] = "datetime"
                continue

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= min_success_rate:
            suggestions[col] = "numeric"
            continue

        if not looks_datey:
            # Cheap pre-check on a small sample before paying for a full-column
            # parse: undated free-text columns (names, cities, addresses) would
            # otherwise force a per-element date parse on every single row.
            sample = series if len(series) <= sample_size else series.sample(sample_size, random_state=0)
            if pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() < min_success_rate:
                continue
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() >= min_success_rate:
                suggestions[col] = "datetime"
    return suggestions


def convert_dtypes(df: pd.DataFrame, mapping: dict = None, auto: bool = True) -> tuple:
    details = {}
    if mapping:
        for col, dtype in mapping.items():
            if col not in df.columns:
                continue
            try:
                if dtype == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(dtype)
                details[col] = str(dtype)
            except (ValueError, TypeError):
                continue

    if auto:
        for col, suggestion in detect_incorrect_dtypes(df).items():
            if col in details:
                continue
            if suggestion == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif suggestion == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            details[col] = suggestion
    return df, details


def detect_mixed_dtypes(df: pd.DataFrame) -> list:
    """Object columns holding more than one Python type (e.g. int and str mixed)."""
    mixed = []
    for col in _object_columns(df):
        sample = df[col].dropna()
        if len(sample) > _SAMPLE_ROWS_FOR_TYPE_SCAN:
            sample = sample.sample(_SAMPLE_ROWS_FOR_TYPE_SCAN, random_state=0)
        if sample.map(type).nunique() > 1:
            mixed.append(col)
    return mixed


# ---------------------------------------------------------------------------
# Invalid values / ranges / infinities
# ---------------------------------------------------------------------------

def detect_invalid_values(df: pd.DataFrame, rules: dict) -> dict:
    """rules: {column: predicate(series) -> boolean mask of INVALID values}."""
    result = {}
    for col, predicate in rules.items():
        if col not in df.columns:
            continue
        count = int(predicate(df[col]).sum())
        if count:
            result[col] = count
    return result


def replace_invalid_values(df: pd.DataFrame, rules: dict, replacement=np.nan) -> tuple:
    details = {}
    for col, predicate in rules.items():
        if col not in df.columns:
            continue
        mask = predicate(df[col])
        count = int(mask.sum())
        if count:
            df.loc[mask, col] = replacement
            details[col] = count
    return df, details


def detect_negative_values(df: pd.DataFrame, columns=None) -> dict:
    result = {}
    for col in _numeric_columns(df, columns):
        count = int((df[col] < 0).sum())
        if count:
            result[col] = count
    return result


def validate_value_ranges(df: pd.DataFrame, ranges: dict) -> dict:
    """ranges: {column: (min, max)}. Returns count of out-of-range values per column."""
    result = {}
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        count = int(((df[col] < lo) | (df[col] > hi)).sum())
        if count:
            result[col] = count
    return result


def handle_infinite_values(df: pd.DataFrame, columns=None, strategy: str = "nan") -> tuple:
    """strategy: 'nan' replaces +/-inf with NaN, 'clip' clips to the column's finite min/max."""
    total_replaced = {}
    for col in _numeric_columns(df, columns):
        mask = np.isinf(df[col])
        count = int(mask.sum())
        if not count:
            continue
        if strategy == "clip":
            finite = df[col][np.isfinite(df[col])]
            if finite.empty:
                df.loc[mask, col] = np.nan
            else:
                df[col] = df[col].clip(finite.min(), finite.max())
        else:
            df.loc[mask, col] = np.nan
        total_replaced[col] = count
    return df, total_replaced


# ---------------------------------------------------------------------------
# Categorical / text cleaning
# ---------------------------------------------------------------------------

def remove_extra_spaces(df: pd.DataFrame, columns=None) -> tuple:
    changed = {}
    for col in _object_columns(df, columns):
        cleaned = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        mask = df[col].notna() & (cleaned != df[col])
        count = int(mask.sum())
        if count:
            df[col] = df[col].where(df[col].isna(), cleaned)
            changed[col] = count
    return df, changed


def remove_special_characters(df: pd.DataFrame, columns=None, keep_pattern: str = r"[^a-zA-Z0-9\s]") -> tuple:
    changed = {}
    pattern = re.compile(keep_pattern)
    for col in _object_columns(df, columns):
        cleaned = df[col].astype(str).str.replace(pattern, "", regex=True)
        mask = df[col].notna() & (cleaned != df[col])
        count = int(mask.sum())
        if count:
            df[col] = df[col].where(df[col].isna(), cleaned)
            changed[col] = count
    return df, changed


def standardize_categorical_values(df: pd.DataFrame, columns=None, case: str = "title") -> tuple:
    """Trim/collapse whitespace and normalize case for text columns."""
    changed = {}
    for col in _object_columns(df, columns):
        cleaned = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        if case == "lower":
            cleaned = cleaned.str.lower()
        elif case == "upper":
            cleaned = cleaned.str.upper()
        elif case == "title":
            cleaned = cleaned.str.title()
        mask = df[col].notna() & (cleaned != df[col])
        count = int(mask.sum())
        if count:
            df[col] = df[col].where(df[col].isna(), cleaned)
            changed[col] = count
    return df, changed


def fix_spelling_inconsistencies(df: pd.DataFrame, columns=None, similarity: float = 0.92, max_unique: int = 2000) -> tuple:
    """Merge near-duplicate category labels (case/spacing/typo variants) to a canonical form.

    Bounded by unique-value count (not row count), and skips high-cardinality
    "free text" columns where fuzzy merges would be unreliable and expensive.
    """
    details = {}
    for col in _object_columns(df, columns):
        value_counts = df[col].value_counts(dropna=True)
        uniques = value_counts.index.tolist()
        if not (2 <= len(uniques) <= max_unique):
            continue

        # Group case/whitespace-insensitive exact matches first (cheap, high confidence).
        norm_groups = {}
        for val in uniques:
            key = re.sub(r"\s+", " ", str(val).strip()).lower()
            norm_groups.setdefault(key, []).append(val)

        mapping = {}
        canon_by_key = {}
        for key, variants in norm_groups.items():
            # Keep the spelling that occurs most often in the actual data.
            canonical = max(variants, key=lambda v: value_counts[v])
            canon_by_key[key] = canonical
            for v in variants:
                if v != canonical:
                    mapping[v] = canonical

        # Fuzzy-match remaining distinct normalized keys for close typos.
        # Digit sequences must match exactly: string similarity alone would
        # otherwise merge distinct identifiers/codes like "Person 4" and
        # "Person 40", or "Region 1" and "Region 11".
        keys = list(canon_by_key.keys())
        used = set()
        for i, key in enumerate(keys):
            if key in used:
                continue
            close = difflib.get_close_matches(key, keys[i + 1:], n=len(keys), cutoff=similarity)
            key_digits = re.findall(r"\d+", key)
            for match in close:
                if match in used or re.findall(r"\d+", match) != key_digits:
                    continue
                used.add(match)
                mapping[canon_by_key[match]] = canon_by_key[key]

        if mapping:
            df[col] = df[col].replace(mapping)
            details[col] = mapping
    return df, details


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def rename_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    return df.rename(columns=mapping)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    new_names = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip())).strip("_").lower() or f"column_{i}"
        for i, c in enumerate(df.columns)
    ]
    # De-duplicate any collisions caused by normalization.
    seen = {}
    deduped = []
    for name in new_names:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)
    df.columns = deduped
    return df


def remove_constant_columns(df: pd.DataFrame) -> tuple:
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant_cols)
    return df, constant_cols


def remove_highly_correlated_columns(df: pd.DataFrame, threshold: float = 0.95, sample_rows: int = 200_000, max_numeric_cols: int = 500) -> tuple:
    numeric_cols = _numeric_columns(df)
    if len(numeric_cols) < 2 or len(numeric_cols) > max_numeric_cols:
        return df, []

    sample = df[numeric_cols]
    if len(sample) > sample_rows:
        sample = sample.sample(sample_rows, random_state=0)

    corr = sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    df = df.drop(columns=to_drop)
    return df, to_drop


# ---------------------------------------------------------------------------
# Empty rows / columns
# ---------------------------------------------------------------------------

def detect_empty_rows(df: pd.DataFrame) -> int:
    return int(df.isna().all(axis=1).sum())


def remove_empty_rows(df: pd.DataFrame) -> tuple:
    before = len(df)
    df = df.dropna(how="all")
    return df, before - len(df)


def detect_empty_columns(df: pd.DataFrame) -> list:
    return list(df.columns[df.isna().all(axis=0)])


def remove_empty_columns(df: pd.DataFrame) -> tuple:
    empty_cols = detect_empty_columns(df)
    df = df.drop(columns=empty_cols)
    return df, empty_cols


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def convert_date_formats(df: pd.DataFrame, columns=None, target_format: str = None) -> tuple:
    """Auto-detect date-like columns (by name or content) and parse them to datetime64."""
    candidates = columns
    if candidates is None:
        candidates = [c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()]

    converted = []
    for col in candidates:
        if col not in df.columns or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed" if columns is None else None)
        if parsed.notna().mean() == 0:
            continue
        df[col] = parsed.dt.strftime(target_format) if target_format else parsed
        converted.append(col)
    return df, converted


# ---------------------------------------------------------------------------
# Encoding & scaling (opt-in: not applied automatically, see cleaning_agent)
# ---------------------------------------------------------------------------

def encode_categorical_variables(df: pd.DataFrame, columns=None, method: str = "label", max_categories: int = 20) -> tuple:
    target_cols = _object_columns(df, columns)
    encoded_cols = []
    for col in target_cols:
        nunique = df[col].nunique(dropna=True)
        if method == "onehot":
            if nunique > max_categories:
                continue
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df, dummies], axis=1)
            encoded_cols.extend(dummies.columns.tolist())
        else:  # label encoding
            codes, _ = pd.factorize(df[col])
            df[f"{col}_encoded"] = codes
            encoded_cols.append(f"{col}_encoded")
    return df, encoded_cols


def scale_numerical_features(df: pd.DataFrame, columns=None, method: str = "standard") -> tuple:
    """method: 'standard' (z-score), 'minmax' (0-1), or 'robust' (median/IQR)."""
    scaled_cols = []
    for col in _numeric_columns(df, columns):
        series = df[col].astype(float)
        if method == "minmax":
            lo, hi = series.min(), series.max()
            spread = hi - lo
        elif method == "robust":
            lo = series.median()
            spread = series.quantile(0.75) - series.quantile(0.25)
        else:  # standard
            lo = series.mean()
            spread = series.std()

        if not spread or pd.isna(spread):
            continue  # constant/degenerate column, scaling is undefined
        df[col] = (series - lo) / spread
        scaled_cols.append(col)
    return df, scaled_cols


# ---------------------------------------------------------------------------
# Large-scale memory optimization
# ---------------------------------------------------------------------------

def optimize_dtypes(df: pd.DataFrame, large_threshold_mb: float = 200.0, categorize: bool = False) -> tuple:
    """Downcast numeric dtypes to shrink memory footprint on large datasets.

    Only kicks in once the frame exceeds `large_threshold_mb` so small/typical
    uploads keep their original dtypes untouched. `categorize` is off by
    default because converting text columns to pandas 'category' dtype would
    make them invisible to `select_dtypes(include="object")` checks used
    elsewhere in the app (e.g. bar-chart generation).
    """
    before_mb = df.memory_usage(deep=True).sum() / 1e6
    if before_mb <= large_threshold_mb:
        return df, None

    for col in df.select_dtypes(include=["int64", "int32", "int16"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    if categorize:
        n = len(df)
        for col in _object_columns(df):
            if n and df[col].nunique(dropna=True) / n < 0.5:
                df[col] = df[col].astype("category")

    after_mb = df.memory_usage(deep=True).sum() / 1e6
    return df, {"before_mb": round(before_mb, 1), "after_mb": round(after_mb, 1)}
