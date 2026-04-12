import pandas as pd
from dotenv import load_dotenv
import os
import json
from agents.llm_fallback import kickoff_with_llm_fallback, is_rate_limit_error, get_shared_limit_message

load_dotenv()

def run_insight_agent(df: pd.DataFrame, eda_results: dict) -> str:
    sample = df.head(5).to_string()
    columns = list(df.columns)
    stats = json.dumps(eda_results.get("stats", {}).get("describe", {}), indent=2)
    outliers = json.dumps(eda_results.get("outliers", {}), indent=2)
    eda_summary = eda_results.get("eda_summary", "")

    prompt = f"""
    You are a senior business analyst. Based on the dataset analysis below, generate 7-10 actionable business insights.

    Dataset columns: {columns}
    Sample data:
    {sample}

    Statistical summary:
    {stats}

    Outliers: {outliers}

    EDA findings:
    {eda_summary}

    Generate insights in this format:
    1. [INSIGHT TITLE]: Detailed explanation with specific numbers/columns mentioned.
    
    Focus on:
    - Revenue / sales patterns (if applicable)
    - Customer behavior patterns
    - Performance anomalies
    - Risk areas
    - Opportunities for growth
    - Seasonal trends
    - Underperforming segments

    Be specific. Use actual column names and numbers from the data.
    """

    try:
        result, _ = kickoff_with_llm_fallback(
            role="Business Insight Generator",
            goal="Convert raw data analysis into actionable business insights",
            backstory="Senior business analyst with expertise in translating data to strategy.",
            prompt=prompt,
            expected_output="7-10 numbered actionable business insights with specific data references.",
        )
        return str(result)
    except Exception as e:
        if is_rate_limit_error(e):
            return get_shared_limit_message(e)
        raise