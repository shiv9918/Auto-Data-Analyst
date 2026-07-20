# agents/cleaning_agent.py
#
# The Cleaning Agent. It loads a raw CSV/Excel file and runs it through a
# fixed set of cleaning steps, in this order:
#
#   1. Handle "NAN" / "NULL" placeholder values (convert to real missing values)
#   2. Remove empty rows & columns (nothing useful left to clean in them)
#   3. Detect duplicate rows                 -> 4. Remove duplicate rows
#      (done before filling missing values - see comment further down)
#   5. Detect missing values                 -> 6. Fill missing values
#   7. Remove extra spaces (text columns)
#   8. Detect outliers                       -> 9. Handle outliers (cap or remove)
#   10. Standardize categorical values (one-hot or mean encoding)
#
# After cleaning, an AI agent (via CrewAI/Groq) writes a short plain-English
# summary of what was done, which is appended to the report.
import pandas as pd
from dotenv import load_dotenv

from tools.pandas_tools import load_file, get_basic_info
from tools import cleaning_tools as ct
from agents.llm_fallback import kickoff_with_llm_fallback, is_rate_limit_error, get_shared_limit_message

load_dotenv()

# Default settings for the two steps that need a choice made:
#   - how to handle outliers ("cap" keeps every row, "remove" drops outlier rows)
#   - how to encode categorical columns ("onehot" needs no extra info and is
#     the safe default; "mean" needs a target_column to average against)
DEFAULT_CONFIG = {
    "outlier_strategy": "cap",       # "cap", "remove", or "none"
    "encoding_method": "onehot",     # "onehot", "mean", or "none"
    "target_column": None,           # required only when encoding_method == "mean"
    "max_categories": 20,            # skip one-hot encoding a column with more unique values than this
}


def run_cleaning_agent(filepath: str, config: dict = None) -> tuple[pd.DataFrame, str]:
    # Merge any caller-supplied options on top of the defaults above.
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    # Load the raw file and capture some basic info about it (used later in
    # the AI summary prompt for context).
    df = load_file(filepath)
    info = get_basic_info(df)
    df = df.copy()  # work on our own copy, never mutate the caller's data

    report_lines = [f"Loaded {info['shape'][0]:,} rows x {info['shape'][1]:,} columns."]

    # --- 1. Handle "NAN" / "NULL" placeholder text --------------------------
    df, token_hits = ct.handle_placeholder_missing_values(df)
    if token_hits:
        total = sum(token_hits.values())
        report_lines.append(
            f"Converted {total} placeholder values (e.g. 'NAN', 'NULL', '?', 'None') to real missing values "
            f"in columns: {', '.join(token_hits.keys())}."
        )

    # --- 2. Remove empty rows & columns -------------------------------------
    df, removed_empty_rows = ct.remove_empty_rows(df)
    if removed_empty_rows:
        report_lines.append(f"Removed {removed_empty_rows} completely empty rows.")

    df, removed_empty_cols = ct.remove_empty_columns(df)
    if removed_empty_cols:
        report_lines.append(f"Removed {len(removed_empty_cols)} completely empty columns: {', '.join(removed_empty_cols)}.")

    # --- 3 & 4. Detect and remove duplicate rows ----------------------------
    # Deliberately run BEFORE filling missing values: two genuinely different
    # rows that are each missing a different field would otherwise both get
    # filled with the same fallback value and start looking like duplicates,
    # causing real (distinct) rows to be silently dropped.
    duplicate_row_count = ct.detect_duplicate_rows(df)
    if duplicate_row_count:
        report_lines.append(f"Detected {duplicate_row_count} duplicate rows.")
        df, removed_dupe_rows = ct.remove_duplicate_rows(df)
        report_lines.append(f"Removed {removed_dupe_rows} duplicate rows.")

    # --- 5 & 6. Detect and fill missing values ------------------------------
    missing_before = ct.detect_missing_values(df)
    if missing_before:
        total_missing = sum(v["count"] for v in missing_before.values())
        report_lines.append(f"Detected {total_missing} missing values across {len(missing_before)} columns: {', '.join(missing_before.keys())}.")

        df, fill_details = ct.fill_missing_values(df, strategy="auto")
        total_filled = sum(d["missing"] for d in fill_details.values())
        report_lines.append(f"Filled {total_filled} missing values (median for numbers, mode for text): {', '.join(fill_details.keys())}.")

    # --- 7. Remove extra spaces in text columns -----------------------------
    df, space_fixes = ct.remove_extra_spaces(df)
    if space_fixes:
        total_space_fixes = sum(space_fixes.values())
        report_lines.append(f"Trimmed/collapsed extra spaces in {total_space_fixes} values across columns: {', '.join(space_fixes.keys())}.")

    # --- 8 & 9. Detect and handle outliers -----------------------------------
    outliers = ct.detect_outliers(df)
    if outliers:
        total_outliers = sum(o["count"] for o in outliers.values())
        report_lines.append(f"Detected {total_outliers} potential outliers (IQR method) in columns: {', '.join(outliers.keys())}.")

        if cfg["outlier_strategy"] == "remove":
            df, removed = ct.handle_outliers(df, strategy="remove")
            report_lines.append(f"Removed {removed['rows_removed']} rows containing outliers.")
        elif cfg["outlier_strategy"] == "cap":
            df, capped = ct.handle_outliers(df, strategy="cap")
            report_lines.append(f"Capped outlier values to the normal IQR range in columns: {', '.join(capped.keys())}.")
        # "none" -> leave outliers untouched

    # --- 10. Standardize categorical values (encoding) -----------------------
    if cfg["encoding_method"] in ("onehot", "mean"):
        df, encoded_cols = ct.standardize_categorical_values(
            df,
            method=cfg["encoding_method"],
            target_column=cfg["target_column"],
            max_categories=cfg["max_categories"],
        )
        if encoded_cols:
            report_lines.append(
                f"Standardized categorical values using {cfg['encoding_method']} encoding: replaced the original "
                f"text columns with {len(encoded_cols)} new numeric columns."
            )

    report_lines.append(f"Final dataset: {len(df):,} rows x {len(df.columns):,} columns.")

    # --- AI-written summary of the cleaning report ---------------------------
    cleaning_summary_prompt = f"""
    You are a data cleaning expert. Here is what was done to clean the dataset:
    {chr(10).join(report_lines)}
    Original info: {info}
    Write a short professional summary (5-7 sentences) of what was cleaned and why it matters.
    """
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

    clean_report = "\n".join(report_lines) + "\n\nAI Summary:\n" + ai_summary
    return df, clean_report
