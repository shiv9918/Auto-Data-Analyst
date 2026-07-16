# Fix text encoding issues for terminal output
import os
import sys

# Ensure UTF-8 encoding is used for printing
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# Main function that runs the entire data analysis pipeline
def run_pipeline(filepath: str, on_step=None) -> dict:
    # Import all the agent functions that use CrewAI
    from agents.cleaning_agent import run_cleaning_agent
    from agents.eda_agent import run_eda_agent
    from agents.visualization_agent import run_visualization_agent
    from agents.insight_agent import run_insight_agent
    from agents.report_agent import run_report_agent
    from agents.llm_fallback import reset_shared_limit_message

    # Helper function to show progress messages
    def notify(message: str) -> None:
        print(message)
        if on_step:
            on_step(message)

    # Reset rate limit tracker
    reset_shared_limit_message()

    # Step 1: Clean the data
    notify("Step 1/5 - Cleaning agent is running (removing duplicates, fixing types, filling missing values)...")
    df_clean, clean_report = run_cleaning_agent(filepath)

    # Save cleaned data to file
    os.makedirs("outputs", exist_ok=True)
    source_name = os.path.splitext(os.path.basename(filepath))[0]
    cleaned_data_path = os.path.join("outputs", f"{source_name}_cleaned.csv")
    df_clean.to_csv(cleaned_data_path, index=False)

    # Step 2: Analyze data patterns
    notify("Step 2/5 - EDA agent is running (statistics, outliers, correlations)...")
    eda_results = run_eda_agent(df_clean)

    # Step 3: Create visualizations
    notify("Step 3/5 - Visualization agent is running (generating charts)...")
    viz_results = run_visualization_agent(df_clean)

    # Step 4: Generate business insights
    notify("Step 4/5 - Insight agent is running (generating business insights)...")
    insights = run_insight_agent(df_clean, eda_results)

    # Step 5: Create final report
    notify("Step 5/5 - Report agent is running (building PDF report)...")
    report_path = run_report_agent(clean_report, eda_results, viz_results, insights)

    # Show completion message
    notify(f"Done! Report saved to: {report_path}")

    # Return all results
    return {
        "clean_report": clean_report,
        "eda_results": eda_results,
        "viz_results": viz_results,
        "insights": insights,
        "report_path": report_path,
        "cleaned_data_path": cleaned_data_path,
    }

if __name__ == "__main__":
    run_pipeline("uploads/sample.csv")
