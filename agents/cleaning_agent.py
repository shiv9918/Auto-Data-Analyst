"""Clean uploaded datasets with CrewAI tools and a deterministic fallback."""

import logging

import pandas as pd

from tools.pandas_tools import load_file
from tools import cleaning_tools as ct
from agents.llm_fallback import kickoff_with_llm_fallback, is_rate_limit_error, get_shared_limit_message

logger = logging.getLogger("auto_data_analyst.cleaning")

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


def _deterministic_cleaning(original: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    df = original.copy()
    report_lines = [f"Loaded {len(df):,} rows x {len(df.columns):,} columns."]

    # --- 1. Handle "NAN" / "NULL" placeholder text --------------------------
    df, token_hits = ct.handle_placeholder_missing_values(df)
    if token_hits:
        total = sum(token_hits.values())
        report_lines.append(
            f"Converted {total} placeholder values (e.g. 'NAN', 'NULL', '?', 'None') to real missing values "
            f"in columns: {', '.join(map(str, token_hits))}."
        )

    # --- 2. Remove empty rows & columns -------------------------------------
    df, removed_empty_rows = ct.remove_empty_rows(df)
    if removed_empty_rows:
        report_lines.append(f"Removed {removed_empty_rows} completely empty rows.")

    df, removed_empty_cols = ct.remove_empty_columns(df)
    if removed_empty_cols:
        report_lines.append(f"Removed {len(removed_empty_cols)} completely empty columns: {', '.join(map(str, removed_empty_cols))}.")

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
        report_lines.append(f"Detected {total_missing} missing values across {len(missing_before)} columns: {', '.join(map(str, missing_before))}.")

        df, fill_details = ct.fill_missing_values(df, strategy="auto")
        total_filled = sum(d["missing"] for d in fill_details.values())
        report_lines.append(f"Filled {total_filled} missing values (median for numbers, mode for text): {', '.join(map(str, fill_details))}.")

    # --- 7. Remove extra spaces in text columns -----------------------------
    df, space_fixes = ct.remove_extra_spaces(df)
    if space_fixes:
        total_space_fixes = sum(space_fixes.values())
        report_lines.append(f"Trimmed/collapsed extra spaces in {total_space_fixes} values across columns: {', '.join(map(str, space_fixes))}.")

    # --- 8 & 9. Detect and handle outliers -----------------------------------
    outliers = ct.detect_outliers(df)
    if outliers:
        total_outliers = sum(o["count"] for o in outliers.values())
        report_lines.append(f"Detected {total_outliers} potential outliers (IQR method) in columns: {', '.join(map(str, outliers))}.")

        if cfg["outlier_strategy"] == "remove":
            df, removed = ct.handle_outliers(df, strategy="remove")
            report_lines.append(f"Removed {removed['rows_removed']} rows containing outliers.")
        elif cfg["outlier_strategy"] == "cap":
            df, capped = ct.handle_outliers(df, strategy="cap")
            report_lines.append(f"Capped outlier values to the normal IQR range in columns: {', '.join(map(str, capped))}.")
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

    return df, report_lines


def run_cleaning_agent(filepath: str, config: dict = None) -> tuple[pd.DataFrame, str]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    original = load_file(filepath)
    logger.info("Data cleaning started shape=%s", original.shape)

    try:
        from agents.cleaning_tool_calls import CleaningSession, CleaningToolError

        session = CleaningSession(original, cfg)
        ai_summary, _ = kickoff_with_llm_fallback(
            role="Data Cleaning Specialist",
            goal="Inspect, clean and verify the dataset with the registered tools.",
            backstory="You clean datasets with audited Pandas operations.",
            prompt=(
                "Call inspect_dataset first. Review its aggregate problem counts and user settings. "
                "Select cleaning tools for issues listed in pending, in the listed order. "
                "After each call, review the updated pending issues before choosing another tool. "
                "Use auto for fill_missing unless another supported strategy is warranted. "
                "Use the exact selected outlier strategy and encoding method when applicable. "
                "When pending is empty, call verify_dataset. After it succeeds, summarize the actual "
                "changes in a few sentences, or explain that no changes were needed."
            ),
            expected_output="A short cleaning summary based on the successful tool results.",
            tools=session.tools(),
        )
        if not session.verified or not ai_summary.strip():
            raise CleaningToolError("The model did not verify and summarize the cleaned dataset.")
        df, report_lines = session.frame, session.report_lines
        logger.info("Data cleaning completed with AI tools shape=%s", df.shape)
    except Exception as err:
        logger.warning("Data cleaning used the deterministic fallback shape=%s reason=%s",
                       original.shape, type(err).__name__)
        if is_rate_limit_error(err):
            ai_summary = get_shared_limit_message(err)
        else:
            ai_summary = "The deterministic cleaning rules were used for this run."
        df, report_lines = _deterministic_cleaning(original, cfg)
        ct.validate_cleaned_dataset(original, df)
        logger.info("Deterministic data cleaning completed shape=%s", df.shape)

    report_lines.append(f"Final dataset: {len(df):,} rows x {len(df.columns):,} columns.")
    clean_report = "\n".join(report_lines) + "\n\nAI Summary:\n" + ai_summary
    return df, clean_report
