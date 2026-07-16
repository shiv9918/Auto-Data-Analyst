# This agent cleans raw data by removing duplicates, converting data types, and filling missing values
import pandas as pd
import numpy as np
from tools.pandas_tools import load_file, get_basic_info
from dotenv import load_dotenv
import os
from agents.llm_fallback import kickoff_with_llm_fallback, is_rate_limit_error, get_shared_limit_message

load_dotenv()


# Main function that cleans data and generates a report
def run_cleaning_agent(filepath: str) -> tuple[pd.DataFrame, str]:
    # Load the file and get initial information
    df = load_file(filepath)
    info = get_basic_info(df)
    report_lines = []

    # Remove duplicate rows from data
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    report_lines.append(f"Removed {removed} duplicate rows.")

    # Convert columns with 'date' or 'time' in their name to datetime format
    for col in df.columns:
        col_name = str(col).lower()
        if "date" in col_name or "time" in col_name:
            try:
                df[col] = pd.to_datetime(df[col])
                report_lines.append(f"Converted '{col}' to datetime.")
            except:
                pass

    # Fill missing values: use median for numbers, mode for text
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing == 0:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
            report_lines.append(f"Filled {missing} missing values in '{col}' with median.")
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
            report_lines.append(f"Filled {missing} missing values in '{col}' with mode.")

    # Standardize column names to lowercase with underscores
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    report_lines.append("Normalized all column names to snake_case.")

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
