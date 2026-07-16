# This agent performs Exploratory Data Analysis (EDA) - finding patterns, correlations, and outliers
import pandas as pd
from tools.pandas_tools import get_statistics, detect_outliers, get_correlation
from dotenv import load_dotenv
import os
import json
from agents.llm_fallback import kickoff_with_llm_fallback, is_rate_limit_error, get_shared_limit_message

load_dotenv()


# Main function that analyzes data for patterns, correlations and anomalies
def run_eda_agent(df: pd.DataFrame) -> dict:
    # Calculate statistical measures for all numeric columns
    stats = get_statistics(df)
    # Find values that are far from normal (outliers)
    outliers = detect_outliers(df)
    # Find relationships between numeric columns
    correlation = get_correlation(df)

    # Prepare prompt for AI to analyze the data findings
    prompt = f"""
    You are an expert data analyst. Analyze the following dataset statistics and provide key findings.

    Statistics:
    {json.dumps(stats['describe'], indent=2)}

    Skewness: {json.dumps(stats['skewness'], indent=2)}

    Outliers detected: {json.dumps(outliers, indent=2)}

    Top correlations: {json.dumps({k: v for k, v in list(correlation.items())[:5]}, indent=2)}

    Provide:
    1. Key patterns found
    2. Important correlations
    3. Columns with outliers and what they mean
    4. Data distribution observations
    5. Any trends or anomalies

    Be specific and use the actual column names and numbers.
    """

    # Use CrewAI to generate AI analysis (see llm_fallback.py for agent setup)
    try:
        eda_summary, _ = kickoff_with_llm_fallback(
            role="EDA Specialist",
            goal="Find meaningful patterns and correlations in data",
            backstory="Senior data scientist specializing in exploratory analysis.",
            prompt=prompt,
            expected_output="Detailed EDA findings with patterns, correlations, and anomalies.",
        )
    except Exception as e:
        if is_rate_limit_error(e):
            eda_summary = get_shared_limit_message(e)
        else:
            raise

    # Return all analysis results together
    return {
        "stats": stats,
        "outliers": outliers,
        "correlation": correlation,
        "eda_summary": eda_summary
    }
