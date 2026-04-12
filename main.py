from agents.cleaning_agent import run_cleaning_agent
from agents.eda_agent import run_eda_agent
from agents.visualization_agent import run_visualization_agent
from agents.insight_agent import run_insight_agent
from agents.report_agent import run_report_agent
from agents.llm_fallback import reset_shared_limit_message
import os

def run_pipeline(filepath: str) -> dict:
    reset_shared_limit_message()
    print("Step 1: Cleaning data...")
    df_clean, clean_report = run_cleaning_agent(filepath)

    os.makedirs("outputs", exist_ok=True)
    source_name = os.path.splitext(os.path.basename(filepath))[0]
    cleaned_data_path = os.path.join("outputs", f"{source_name}_cleaned.csv")
    df_clean.to_csv(cleaned_data_path, index=False)

    print("Step 2: Running EDA...")
    eda_results = run_eda_agent(df_clean)

    print("Step 3: Generating visualizations...")
    viz_results = run_visualization_agent(df_clean)

    print("Step 4: Generating insights...")
    insights = run_insight_agent(df_clean, eda_results)

    print("Step 5: Creating report...")
    report_path = run_report_agent(clean_report, eda_results, viz_results, insights)

    print(f"Done! Report saved to: {report_path}")

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