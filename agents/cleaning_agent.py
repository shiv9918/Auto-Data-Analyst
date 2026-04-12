import pandas as pd
import numpy as np
from tools.pandas_tools import load_file, get_basic_info
from dotenv import load_dotenv
import os
from agents.llm_fallback import kickoff_with_llm_fallback, is_rate_limit_error, get_shared_limit_message

load_dotenv()



def run_cleaning_agent(filepath: str) -> tuple[pd.DataFrame, str]:
    df = load_file(filepath)
    info = get_basic_info(df)
    report_lines = []

    # 1. Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    report_lines.append(f"Removed {removed} duplicate rows.")

    # 2. Fix data types — try parsing date columns
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                report_lines.append(f"Converted '{col}' to datetime.")
            except:
                pass

    # 3. Handle missing values
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

    # 4. Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    report_lines.append("Normalized all column names to snake_case.")

    # 5. Use LLM to summarize cleaning
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