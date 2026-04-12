import pandas as pd
from tools.chart_generator import (
    plot_histogram, plot_bar,
    plot_correlation_heatmap, plot_line, plot_boxplot
)
from dotenv import load_dotenv
import os
from agents.llm_fallback import kickoff_with_llm_fallback, is_rate_limit_error, get_shared_limit_message

load_dotenv()

def run_visualization_agent(df: pd.DataFrame) -> dict:
    chart_paths = []

    # Generate all chart types
    chart_paths += plot_histogram(df)
    chart_paths += plot_bar(df)
    chart_paths += plot_boxplot(df)
    chart_paths += plot_line(df)

    heatmap = plot_correlation_heatmap(df)
    if heatmap:
        chart_paths.append(heatmap)

    # Ask LLM which charts are most important
    prompt = f"""
    You are a data visualization expert.
    The following charts were generated for a dataset with columns: {list(df.columns)}
    Charts: {chart_paths}

    Explain in 3-5 sentences:
    - Which charts are most important to look at first
    - What each chart type reveals about the data
    - Any visualization recommendations
    """

    try:
        viz_summary, _ = kickoff_with_llm_fallback(
            role="Visualization Expert",
            goal="Generate and explain the best charts for the dataset",
            backstory="Expert in data visualization with Matplotlib and Seaborn.",
            prompt=prompt,
            expected_output="Chart importance explanation and visualization recommendations.",
        )
    except Exception as e:
        if is_rate_limit_error(e):
            viz_summary = get_shared_limit_message(e)
        else:
            raise

    return {
        "chart_paths": chart_paths,
        "viz_summary": viz_summary
    }