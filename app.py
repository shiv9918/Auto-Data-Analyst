# Web UI for the data analyst - built with Streamlit
# Users upload files here and see the analysis results
import os
import warnings

warnings.filterwarnings("ignore")

# Import data libraries
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import BadRequestError

# Import backend functions for data processing
from backend.query_engine import (
    dataset_examples,
    extract_limit_message,
    extract_python_code,
    invoke_llm_with_fallback,
    prepare_dashboard_df,
    rate_limit_message,
    remove_limit_message,
    run_generated_pandas_code,
)
# Import the main pipeline that runs all agents
from main import run_pipeline

# Create uploads folder if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure Streamlit page
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")


# Function to read CSV or Excel files
def load_dataframe(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)


# Display EDA (Exploratory Data Analysis) results
def render_eda_tab(eda_results: dict) -> None:
    # Get the AI-generated summary text
    eda_text = str(eda_results.get("eda_summary", ""))
    # Check if there was a rate limit error
    if extract_limit_message(eda_text):
        st.info("See limit banner above.")
    else:
        st.subheader("AI-generated findings")
        st.write(eda_text)

    # Display statistical summary table
    stats = eda_results.get("stats", {})
    describe = stats.get("describe", {})
    if describe:
        st.subheader("Descriptive statistics")
        stats_df = pd.DataFrame(describe).T
        stats_df["skew"] = pd.Series(stats.get("skewness", {}))
        stats_df["kurtosis"] = pd.Series(stats.get("kurtosis", {}))
        st.dataframe(stats_df.style.format("{:.3f}"), width="stretch")

    # Display outliers in a side-by-side layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Outliers (IQR method)")
        outliers = eda_results.get("outliers", {})
        if outliers:
            st.bar_chart(pd.Series(outliers, name="outlier_count"))
        else:
            st.caption("No significant outliers detected.")

    # Display correlation matrix
    with col2:
        st.subheader("Correlation matrix")
        correlation = eda_results.get("correlation", {})
        if correlation and len(correlation) >= 2:
            corr_df = pd.DataFrame(correlation)
            st.dataframe(
                corr_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"),
                width="stretch",
            )
        else:
            st.caption("Not enough numeric columns for a correlation matrix.")


# Display dashboard with filters and charts
def render_dashboard_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available for the dashboard.")
        return

    # Prepare dataframe for dashboard
    dashboard_df = prepare_dashboard_df(df)
    # Find available columns for filtering
    date_cols = list(dashboard_df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns)
    category_cols = [
        col
        for col in dashboard_df.select_dtypes(include=["object", "category", "bool"]).columns
        if dashboard_df[col].nunique(dropna=True) <= 50
    ]
    numeric_cols = list(dashboard_df.select_dtypes(include=np.number).columns)

    if not numeric_cols:
        st.info("Dashboard requires at least one numeric column.")
        return

    # Show filter options
    st.subheader("Filters")
    f1, f2, f3 = st.columns(3)
    date_filter_col = f1.selectbox("Date filter column", ["None"] + date_cols, key="dash_date_col")

    # Apply date filter
    filtered = dashboard_df.copy()
    if date_filter_col != "None":
        series = filtered[date_filter_col].dropna()
        if not series.empty:
            min_d, max_d = series.min().date(), series.max().date()
            date_range = f2.date_input(
                "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d, key="dash_date_range"
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                filtered = filtered[(filtered[date_filter_col] >= start) & (filtered[date_filter_col] <= end)]

    # Apply category filter
    category_filter_col = f3.selectbox("Category filter column", ["None"] + category_cols, key="dash_cat_col")
    if category_filter_col != "None":
        options = sorted(filtered[category_filter_col].dropna().astype(str).unique().tolist())
        selected = st.multiselect(
            f"Filter {category_filter_col}", options, default=options[: min(8, len(options))], key="dash_cat_values"
        )
        if selected:
            filtered = filtered[filtered[category_filter_col].astype(str).isin(selected)]

    if filtered.empty:
        st.info("No records match the selected filters.")
        return

    # Create trend visualization
    st.subheader("Trend / breakdown")
    axis_options = date_cols + category_cols or [numeric_cols[0]]
    c1, c2, c3 = st.columns(3)
    x_axis = c1.selectbox("X axis", axis_options, key="dash_x_axis")
    metric = c2.selectbox("Metric", numeric_cols, key="dash_metric")
    agg = c3.selectbox("Aggregation", ["sum", "mean", "count"], key="dash_agg")

    # Aggregate and visualize
    y_col = f"{agg}_{metric}"
    grouped = filtered.groupby(x_axis, dropna=False)[metric].agg(agg).reset_index(name=y_col)

    # Choose chart type based on axis type
    if x_axis in date_cols:
        grouped = grouped.sort_values(x_axis)
        fig = px.line(grouped, x=x_axis, y=y_col, title=f"{agg.upper()} of {metric} by {x_axis}")
    else:
        grouped = grouped.sort_values(y_col, ascending=False).head(30)
        fig = px.bar(grouped, x=x_axis, y=y_col, title=f"{agg.upper()} of {metric} by {x_axis}")
    st.plotly_chart(fig, width="stretch")

    ec1, ec2 = st.columns(2)
    with ec1:
        st.plotly_chart(
            px.histogram(filtered, x=metric, nbins=30, title=f"Distribution of {metric}"),
            width="stretch",
        )
    with ec2:
        if len(numeric_cols) >= 2:
            second_metric = numeric_cols[1] if numeric_cols[0] == metric else numeric_cols[0]
            scatter = px.scatter(
                filtered,
                x=metric,
                y=second_metric,
                color=category_filter_col if category_filter_col != "None" else None,
                title=f"{metric} vs {second_metric}",
            )
            st.plotly_chart(scatter, width="stretch")
        else:
            st.info("Add another numeric column to enable scatter plot.")


def render_ask_result(df: pd.DataFrame, question: str) -> None:
    print(f"\n=== ASK: \"{question}\" ===")
    context = df.head(20).to_string()
    answer_text = ""
    can_run_structured_query = True

    print("Stage 1/4 - Asking Groq for a plain-English answer...")
    try:
        answer_text = invoke_llm_with_fallback(
            f"Dataset sample:\n{context}\n\nQuestion: {question}\n"
            "Answer with data-backed reasoning in plain English. "
            "If a calculation is relevant, include only the formula in text form. "
            "Do not provide Python code or code blocks.",
            label="answer",
        )
        print("Stage 1/4 - Answer ready.")
        st.write(answer_text)
    except BadRequestError as err:
        answer_text = f"Groq request failed. Update MODEL_NAME in .env.\n\nDetails: {err}"
        print(f"Stage 1/4 - Groq request failed: {err}")
        st.error(answer_text)
        can_run_structured_query = False
    except Exception as err:
        msg = rate_limit_message(err) or f"Could not answer the question: {err}"
        answer_text = msg
        print(f"Stage 1/4 - Failed: {err}")
        st.error(msg)
        can_run_structured_query = False

    if answer_text:
        st.session_state.chat_messages.append({"role": "assistant", "content": answer_text})

    if not can_run_structured_query:
        print("=== ASK: done (structured query skipped) ===\n")
        return

    dtype_info = ", ".join([f"{col}:{dtype}" for col, dtype in df.dtypes.astype(str).items()])
    pandas_prompt = f"""
You are a pandas code generator.
DataFrame name is df.
Columns with dtypes: {dtype_info}

Task: {question}

Rules:
- Return ONLY python code.
- Do not import anything.
- Use existing df only.
- Save final answer in variable named result.
- result must be a pandas DataFrame or Series.
- Optional chart config:
  - chart = 'bar' or 'line'
  - chart_x = '<column_name>'
  - chart_y = '<column_name>'
- Keep code concise and deterministic.
""".strip()

    try:
        print("Stage 2/4 - Asking Groq to generate pandas code for a structured result...")
        code_response_text = invoke_llm_with_fallback(pandas_prompt, label="pandas_code")
        generated_code = extract_python_code(code_response_text)
        print(f"Stage 2/4 - Generated code:\n{generated_code}")

        print("Stage 3/4 - Asking Groq to convert the pandas logic into a formula...")
        formula_text = invoke_llm_with_fallback(
            "Convert this pandas logic into concise math/business formula text only. "
            "No Python, no code blocks, no technical implementation details.\n\n"
            f"Question: {question}\n"
            f"Pandas logic:\n{generated_code}",
            label="formula",
        )
        print("Stage 3/4 - Formula ready.")
        st.markdown("**Formula used**")
        st.write(formula_text)

        print("Stage 4/4 - Executing generated pandas code locally...")
        result, chart_cfg, exec_err = run_generated_pandas_code(df, generated_code)
        if exec_err or result is None:
            print(f"Stage 4/4 - No structured result (exec_err={exec_err!r}).")
            print("=== ASK: done ===\n")
            st.caption("No structured table/chart result generated for this question.")
            return
        print(f"Stage 4/4 - Execution succeeded (result type: {type(result).__name__}).")
        print("=== ASK: done ===\n")

        st.markdown("**Query result**")
        if isinstance(result, pd.Series):
            out_df = result.reset_index()
            out_df.columns = ["index", "value"]
            st.dataframe(out_df, width="stretch")
        elif isinstance(result, pd.DataFrame):
            st.dataframe(result, width="stretch")
        else:
            st.write(result)

        st.markdown("**Chart preview**")
        if isinstance(result, pd.Series):
            st.bar_chart(result)
        elif isinstance(result, pd.DataFrame):
            chart_type = chart_cfg.get("chart") if chart_cfg else None
            chart_x = chart_cfg.get("chart_x") if chart_cfg else None
            chart_y = chart_cfg.get("chart_y") if chart_cfg else None

            if chart_type in {"bar", "line"} and chart_x in result.columns and chart_y in result.columns:
                chart_df = result[[chart_x, chart_y]].set_index(chart_x)
                st.line_chart(chart_df) if chart_type == "line" else st.bar_chart(chart_df)
            else:
                numeric_cols = list(result.select_dtypes(include=np.number).columns)
                if numeric_cols:
                    st.bar_chart(result[numeric_cols].head(20))
                else:
                    st.caption("No numeric columns found for charting this result.")
    except BadRequestError as err:
        print(f"ASK failed - Groq request failed: {err}")
        print("=== ASK: done ===\n")
        st.error(f"Groq request failed. Update MODEL_NAME in .env.\n\nDetails: {err}")
    except Exception as err:
        print(f"ASK failed - {err}")
        print("=== ASK: done ===\n")
        st.error(rate_limit_message(err) or f"Could not process structured query output: {err}")


st.title("📊 Automated AI Data Analyst")
st.caption("Upload your CSV or Excel file and let AI agents clean it, analyze it, and report back.")

for key, default in {
    "last_uploaded_file": None,
    "analysis_results": None,
    "dashboard_df": None,
    "chat_messages": [],
}.items():
    st.session_state.setdefault(key, default)

uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls"])

qna_example = "Why did sales drop in March?"
df = None

if uploaded_file:
    filepath = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.analysis_results = None
        st.session_state.dashboard_df = None
        print(f"Uploaded file: {uploaded_file.name}")
        st.success(f"File uploaded: {uploaded_file.name}")

    try:
        df = load_dataframe(filepath)
    except Exception as err:
        st.error(f"Could not read this file: {err}")
        df = None

    if df is not None:
        qna_example, _ = dataset_examples(df)

        st.subheader("Data preview")
        st.dataframe(df.head(10), width="stretch")
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        if st.button("Run AI Analysis", type="primary"):
            try:
                with st.status("Running AI agents…", expanded=True) as status:
                    def on_step(message: str, _status=status) -> None:
                        _status.write(message)
                    results = run_pipeline(filepath, on_step=on_step)
                    status.update(label="Analysis complete!", state="complete")
                st.session_state.analysis_results = results
                cleaned_path = results.get("cleaned_data_path")
                st.session_state.dashboard_df = (
                    pd.read_csv(cleaned_path) if cleaned_path and os.path.exists(cleaned_path) else df.copy()
                )
                st.success("Analysis complete!")
            except Exception as err:
                st.error(rate_limit_message(err) or f"Analysis failed: {err}")

        results = st.session_state.analysis_results
        if results:
            limit_candidates = [
                str(results.get("clean_report", "")),
                str(results.get("eda_results", {}).get("eda_summary", "")),
                str(results.get("insights", "")),
                str(results.get("viz_results", {}).get("viz_summary", "")),
            ]
            shared_limit_msg = next(
                (m for c in limit_candidates if (m := extract_limit_message(c))), ""
            )
            if shared_limit_msg:
                st.warning(shared_limit_msg)

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cleaning", "EDA", "Charts", "Insights", "Dashboard"])

            with tab1:
                st.subheader("Data cleaning report")
                st.text(remove_limit_message(str(results["clean_report"])))

            with tab2:
                render_eda_tab(results["eda_results"])

            with tab3:
                viz_results = results["viz_results"]
                viz_summary = str(viz_results.get("viz_summary", ""))
                if viz_summary and not extract_limit_message(viz_summary):
                    st.subheader("Visualization recommendations")
                    st.write(viz_summary)

                st.subheader("Generated charts")
                chart_paths = [p for p in viz_results.get("chart_paths", []) if os.path.exists(p)]
                if not chart_paths:
                    st.caption("No charts were generated for this dataset.")
                else:
                    cols = st.columns(2)
                    for i, path in enumerate(chart_paths):
                        name = os.path.basename(path).replace("_", " ").replace(".png", "")
                        with cols[i % 2]:
                            st.image(path, caption=name, width="stretch")

            with tab4:
                st.subheader("Business insights")
                insights_text = str(results.get("insights", ""))
                if insights_text and not extract_limit_message(insights_text):
                    st.write(insights_text)
                else:
                    st.caption("No insights available.")

            with tab5:
                st.subheader("Interactive dashboard")
                dashboard_df = st.session_state.dashboard_df
                if isinstance(dashboard_df, pd.DataFrame):
                    render_dashboard_tab(dashboard_df)
                else:
                    st.caption("Run AI Analysis to generate dashboard data.")

            dl1, dl2 = st.columns(2)
            report_path = results.get("report_path")
            if report_path and os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    dl1.download_button(
                        "⬇️ Download PDF Report", data=f, file_name="data_analysis_report.pdf", mime="application/pdf"
                    )
            cleaned_data_path = results.get("cleaned_data_path")
            if cleaned_data_path and os.path.exists(cleaned_data_path):
                with open(cleaned_data_path, "rb") as f:
                    dl2.download_button(
                        "⬇️ Download Cleaned Data (CSV)",
                        data=f,
                        file_name=os.path.basename(cleaned_data_path),
                        mime="text/csv",
                    )

st.divider()
st.subheader("Ask a question about your data")
st.caption("Answers are generated live from your uploaded file via Groq.")

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(f"e.g. {qna_example}")

if question is not None:
    if df is None:
        st.warning("Upload a file first to ask data questions.")
    elif not question.strip():
        st.warning("Enter a question before submitting.")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            render_ask_result(df, question.strip())
