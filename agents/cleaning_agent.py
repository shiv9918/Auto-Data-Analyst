# This agent cleans raw data: missing/NaN values, duplicates, outliers, bad
# dtypes, invalid values, messy text, redundant columns, and more.
#
# The automatic pipeline (run_cleaning_agent) only applies fixes that are
# safe to do without human judgement (deduping, filling, type/format fixes,
# whitespace/case cleanup). Destructive or opinionated operations - dropping
# outlier rows, encoding categoricals, scaling numeric features, dropping
# correlated columns - are implemented in tools/cleaning_tools.py and can be
# turned on via the `config` argument, but are left off by default so the
# downstream EDA/visualization/insight agents still see interpretable data.
import pandas as pd
from dotenv import load_dotenv

from tools.pandas_tools import load_file, get_basic_info
from tools import cleaning_tools as ct
from agents.llm_fallback import kickoff_with_llm_fallback, is_rate_limit_error, get_shared_limit_message

load_dotenv()

DEFAULT_CONFIG = {
    "remove_outliers": False,
    "cap_outliers": True,
    "remove_correlated_columns": False,
    "correlation_threshold": 0.95,
    "encode_categorical": False,
    "scale_numeric": False,
    "spelling_fix_similarity": 0.92,
    "large_dataset_threshold_mb": 200.0,
    "rename_map": None,  # {original_column: new_name}, applied against the raw uploaded column names
}


def run_cleaning_agent(filepath: str, config: dict = None) -> tuple[pd.DataFrame, str]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    # Load the file (transparently chunk-reads very large CSVs) and capture info
    # about the raw data before any cleaning, for the AI summary prompt.
    df = load_file(filepath)
    info = get_basic_info(df)
    df = df.copy()

    report_lines = [f"Loaded {info['shape'][0]:,} rows x {info['shape'][1]:,} columns."]

    # --- User-requested column renames (applied against raw column names) ---
    if cfg.get("rename_map"):
        valid_map = {k: v for k, v in cfg["rename_map"].items() if k in df.columns and v and str(v).strip() and str(v).strip() != k}
        if valid_map:
            df = ct.rename_columns(df, valid_map)
            renamed_desc = ", ".join(f"'{k}' -> '{v}'" for k, v in valid_map.items())
            report_lines.append(f"Renamed {len(valid_map)} columns: {renamed_desc}.")

    # --- Missing value placeholders -------------------------------------
    df, token_hits = ct.normalize_missing_tokens(df)
    if token_hits:
        total = sum(token_hits.values())
        report_lines.append(
            f"Normalized {total} placeholder values (e.g. 'NULL', 'NaN', '?', 'None') to real missing values "
            f"in columns: {ct.summarize_list(token_hits.keys())}."
        )

    # --- Empty rows / columns --------------------------------------------
    df, removed_empty_rows = ct.remove_empty_rows(df)
    if removed_empty_rows:
        report_lines.append(f"Removed {removed_empty_rows} completely empty rows.")

    df, removed_empty_cols = ct.remove_empty_columns(df)
    if removed_empty_cols:
        report_lines.append(f"Removed {len(removed_empty_cols)} completely empty columns: {ct.summarize_list(removed_empty_cols)}.")

    # --- Duplicates --------------------------------------------------------
    df, removed_dupe_rows = ct.remove_duplicate_rows(df)
    if removed_dupe_rows:
        report_lines.append(f"Removed {removed_dupe_rows} duplicate rows.")

    df, removed_dupe_cols = ct.remove_duplicate_columns(df)
    if removed_dupe_cols:
        report_lines.append(f"Removed {len(removed_dupe_cols)} duplicate columns (identical to another column): {ct.summarize_list(removed_dupe_cols)}.")

    # --- Constant columns (zero information) -------------------------------
    df, constant_cols = ct.remove_constant_columns(df)
    if constant_cols:
        report_lines.append(f"Removed {len(constant_cols)} constant columns (single unique value): {ct.summarize_list(constant_cols)}.")

    # --- Text cleanup --------------------------------------------------------
    df, space_fixes = ct.remove_extra_spaces(df)
    if space_fixes:
        report_lines.append(f"Trimmed/collapsed whitespace in {sum(space_fixes.values())} values across columns: {ct.summarize_list(space_fixes.keys())}.")

    df, spelling_fixes = ct.fix_spelling_inconsistencies(df, similarity=cfg["spelling_fix_similarity"])
    if spelling_fixes:
        merged = sum(len(m) for m in spelling_fixes.values())
        report_lines.append(f"Merged {merged} inconsistent category labels (case/spacing/typo variants) in columns: {ct.summarize_list(spelling_fixes.keys())}.")

    # --- Data types (numeric-looking text, date-looking text) ---------------
    df, dtype_fixes = ct.convert_dtypes(df, auto=True)
    if dtype_fixes:
        numeric_fixed = [c for c, t in dtype_fixes.items() if t == "numeric"]
        date_fixed = [c for c, t in dtype_fixes.items() if t == "datetime"]
        if numeric_fixed:
            report_lines.append(f"Converted {len(numeric_fixed)} text columns stored as numbers to numeric dtype: {ct.summarize_list(numeric_fixed)}.")
        if date_fixed:
            report_lines.append(f"Converted {len(date_fixed)} text columns to datetime: {ct.summarize_list(date_fixed)}.")

    mixed_type_cols = ct.detect_mixed_dtypes(df)
    if mixed_type_cols:
        report_lines.append(f"Warning: {len(mixed_type_cols)} columns contain mixed Python types (e.g. numbers and text mixed): {ct.summarize_list(mixed_type_cols)}.")

    # --- Infinite values -----------------------------------------------------
    df, inf_fixes = ct.handle_infinite_values(df, strategy="nan")
    if inf_fixes:
        report_lines.append(f"Replaced {sum(inf_fixes.values())} infinite values (inf/-inf) with missing in columns: {ct.summarize_list(inf_fixes.keys())}.")

    # --- Missing value imputation (after dtype/inf fixes so stats are clean) -
    df, fill_details = ct.fill_missing_values(df, strategy="auto")
    if fill_details:
        total_filled = sum(d["missing"] for d in fill_details.values())
        report_lines.append(f"Filled {total_filled} missing values across {len(fill_details)} columns (median for numeric, mode for text): {ct.summarize_list(fill_details.keys())}.")

    # --- Outlier handling ---------------------------------------------------
    outliers = ct.detect_outliers(df)
    if outliers:
        total_outliers = sum(o["count"] for o in outliers.values())
        report_lines.append(
            f"Detected {total_outliers} potential outliers (IQR method) in {len(outliers)} columns: {ct.summarize_list(outliers.keys())}."
        )
        if cfg["remove_outliers"]:
            df, removed = ct.remove_outliers(df)
            report_lines.append(f"Removed {removed} outlier rows.")
        elif cfg["cap_outliers"]:
            df, capped = ct.cap_outliers(df)
            report_lines.append(f"Capped {sum(capped.values())} outlier values to IQR bounds in columns: {ct.summarize_list(capped.keys())}.")
        else:
            report_lines.append("Outliers left in place (outlier handling disabled) - see EDA report.")

    negative_values = ct.detect_negative_values(df)
    if negative_values:
        report_lines.append(f"Note: negative values found in columns: {ct.summarize_list(negative_values.keys())} (left as-is; may be valid).")

    # --- Column naming ---------------------------------------------------
    df = ct.standardize_column_names(df)
    report_lines.append("Standardized all column names to snake_case.")

    # --- Optional, opinionated steps (off by default) -----------------------
    if cfg["remove_correlated_columns"]:
        df, dropped_corr = ct.remove_highly_correlated_columns(df, threshold=cfg["correlation_threshold"])
        if dropped_corr:
            report_lines.append(f"Removed {len(dropped_corr)} highly correlated columns (>{cfg['correlation_threshold']}): {ct.summarize_list(dropped_corr)}.")

    if cfg["encode_categorical"]:
        df, encoded_cols = ct.encode_categorical_variables(df)
        if encoded_cols:
            report_lines.append(f"Added {len(encoded_cols)} encoded columns for categorical variables.")

    if cfg["scale_numeric"]:
        df, scaled_cols = ct.scale_numerical_features(df)
        if scaled_cols:
            report_lines.append(f"Scaled {len(scaled_cols)} numeric columns: {ct.summarize_list(scaled_cols)}.")

    # --- Memory optimization for large datasets -------------------------
    df, mem_stats = ct.optimize_dtypes(df, large_threshold_mb=cfg["large_dataset_threshold_mb"])
    if mem_stats:
        report_lines.append(f"Large dataset detected - downcast numeric dtypes to reduce memory from {mem_stats['before_mb']}MB to {mem_stats['after_mb']}MB.")

    report_lines.append(f"Final dataset: {len(df):,} rows x {len(df.columns):,} columns.")

    # Ask AI to write a summary of what was cleaned
    cleaning_summary_prompt = f"""
    You are a data cleaning expert. Here is what was done to clean the dataset:
    {chr(10).join(report_lines)}
    Original info: {info}
    Write a short professional summary (5-7 sentences) of what was cleaned and why it matters.
    """
    # Use CrewAI to generate AI summary (see llm_fallback.py for agent setup)
    try:
        ai_summary, _ = kickoff_with_llm_fallback(
            role="Data Cleaning Specialist",
            goal="Clean and prepare data for analysis",
            backstory="Expert in data wrangling with 10 years of experience.",
            prompt=cleaning_summary_prompt,
            expected_output="A short professional cleaning summary.",
        )
    except Exception as e:
        if is_rate_limit_error(e):
            ai_summary = get_shared_limit_message(e)
        else:
            raise

    # Combine cleaning report with AI summary
    clean_report = "\n".join(report_lines) + "\n\nAI Summary:\n" + ai_summary
    return df, clean_report
