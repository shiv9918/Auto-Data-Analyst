# This agent creates visualizations (charts and graphs) to show data patterns
import pandas as pd
from tools.chart_generator import (
    plot_histogram, plot_bar,
    plot_correlation_heatmap, plot_line, plot_boxplot
)
from dotenv import load_dotenv
from agents.llm_fallback import kickoff_with_llm_fallback, is_rate_limit_error, get_shared_limit_message

load_dotenv()


# Main function that generates different types of charts
def run_visualization_agent(df: pd.DataFrame) -> dict:
    # Store all generated chart file paths
    chart_paths = []

    # Generate different chart types
    chart_paths += plot_histogram(df)  # Show data distribution
    chart_paths += plot_bar(df)  # Show category counts
    chart_paths += plot_boxplot(df)  # Show outliers and quartiles
    chart_paths += plot_line(df)  # Show trends over time

    # Generate correlation heatmap showing relationships between numbers
    heatmap = plot_correlation_heatmap(df)
    if heatmap:
        chart_paths.append(heatmap)

    # Prepare prompt for AI to explain the visualizations
    prompt = f"""
    You are a data visualization expert.
    The following charts were generated for a dataset with columns: {list(df.columns)}
    Charts: {chart_paths}

    Explain in 3-5 sentences:
    - Which charts are most important to look at first
    - What each chart type reveals about the data
    - Any visualization recommendations
    """

    # Use CrewAI to generate AI explanation of charts (see llm_fallback.py for agent setup)
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

    # Return chart files and AI explanation
    return {
        "chart_paths": chart_paths,
        "viz_summary": viz_summary
    }
