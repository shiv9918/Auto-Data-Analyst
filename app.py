import streamlit as st
import os
import pandas as pd
import numpy as np
import re
import plotly.express as px
from main import run_pipeline
from agents.llm_fallback import is_rate_limit_error, build_limit_exceeded_message
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import BadRequestError


def _extract_python_code(text: str) -> str:
    code_match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    return text.strip()


def _validate_generated_code(code: str) -> tuple[bool, str]:
    blocked_patterns = [
        "import ",
        "from ",
        "open(",
        "exec(",
        "eval(",
        "__",
        "os.",
        "sys.",
        "subprocess",
        "shutil",
        "pathlib",
        "requests",
    ]
    lower_code = code.lower()
    for pattern in blocked_patterns:
        if pattern in lower_code:
            return False, f"Blocked operation detected: {pattern.strip()}"
    return True, ""


def _run_generated_pandas_code(df: pd.DataFrame, code: str):
    is_valid, err = _validate_generated_code(code)
    if not is_valid:
        return None, None, err

    local_vars = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "result": None,
        "chart": None,
        "chart_x": None,
        "chart_y": None,
    }

    try:
        exec(code, {"__builtins__": {}}, local_vars)
    except Exception as e:
        return None, None, str(e)

    result = local_vars.get("result")
    chart_cfg = {
        "chart": local_vars.get("chart"),
        "chart_x": local_vars.get("chart_x"),
        "chart_y": local_vars.get("chart_y"),
    }
    return result, chart_cfg, None


def _rate_limit_message(err: Exception) -> str:
    if is_rate_limit_error(err):
        return build_limit_exceeded_message(err)
    return ""


def _extract_limit_message(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"Groq limit exceeded\. Use after [^.\n]+\.",
        r"Groq limit exceeded\. Please try again later\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""


def _remove_limit_message(text: str) -> str:
    limit_msg = _extract_limit_message(text)
    if not limit_msg:
        return text
    cleaned = text.replace(f"\n\nAI Summary:\n{limit_msg}", "")
    cleaned = cleaned.replace(limit_msg, "")
    return cleaned.strip()


def _invoke_llm_with_fallback(prompt: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    
    load_dotenv()

    groq_model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if "last_provider_used" not in st.session_state:
        st.session_state.last_provider_used = "groq"

    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    system_message = """You are a data analyst assistant. When answering questions about data:
1. Include relevant formulas/equations (use markdown notation like `formula` or math expressions)
2. Keep your response concise and to-the-point (max 3-4 sentences unless asked for details)
3. Lead with the answer, then briefly explain the approach
4. Use bullet points for multiple items
5. Show calculations only when necessary for clarity"""

    groq_llm = ChatGroq(model=groq_model, api_key=groq_key)
    st.session_state.last_provider_used = "groq"
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]
    
    return groq_llm.invoke(messages).content


def _render_provider_badge() -> None:
    provider = st.session_state.get("last_provider_used", "groq")
    provider_label = "Groq" if provider == "groq" else "Groq"
    color = "#0b6cff"
    st.markdown(
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"border:1px solid {color};color:{color};font-size:12px;font-weight:600;'>"
        f"Provider: {provider_label}</span>",
        unsafe_allow_html=True,
    )


def _dataset_examples(df: pd.DataFrame) -> tuple[str, str]:
    cols = list(df.columns)
    if not cols:
        return (
            "What are the main trends in this dataset?",
            "Show top 10 rows sorted by a key metric",
        )

    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    def pick_column(candidates, contains_terms):
        for col in candidates:
            col_name = str(col).lower()
            if any(term in col_name for term in contains_terms):
                return col
        return candidates[0] if candidates else None

    customer_col = pick_column(categorical_cols + cols, ["customer", "client", "user", "name", "account"])
    metric_col = pick_column(numeric_cols + cols, ["revenue", "sales", "amount", "price", "profit", "total"])
    time_col = pick_column(cols, ["date", "month", "year", "time", "period"])

    qna_example = f"What are the top 5 {customer_col} by {metric_col}?" if customer_col and metric_col else (
        f"What insights can you share about {metric_col}?" if metric_col else f"What does {cols[0]} tell us about performance?"
    )

    if customer_col and metric_col:
        pandas_example = f"Show top 10 {customer_col} by {metric_col}"
    elif time_col and metric_col:
        pandas_example = f"Show {metric_col} trend by {time_col}"
    elif metric_col:
        pandas_example = f"Show summary statistics for {metric_col}"
    else:
        pandas_example = f"Show top 10 values in {cols[0]}"

    return qna_example, pandas_example


def _prepare_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    dashboard_df = df.copy()
    for col in dashboard_df.columns:
        if dashboard_df[col].dtype == "object":
            parsed = pd.to_datetime(dashboard_df[col], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                dashboard_df[col] = parsed
    return dashboard_df


def _render_interactive_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Interactive Dashboard")

    if df.empty:
        st.info("No data available for dashboard.")
        return

    dashboard_df = _prepare_dashboard_df(df)
    date_cols = list(dashboard_df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns)
    category_cols = [
        col for col in dashboard_df.select_dtypes(include=["object", "category", "bool"]).columns
        if dashboard_df[col].nunique(dropna=True) <= 50
    ]
    numeric_cols = list(dashboard_df.select_dtypes(include=np.number).columns)

    if not numeric_cols:
        st.info("Dashboard requires at least one numeric column.")
        return

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    date_filter_col = filter_col1.selectbox(
        "Date filter column",
        options=["None"] + date_cols,
        index=0,
        key="dashboard_date_col",
    )

    filtered_df = dashboard_df.copy()
    if date_filter_col != "None" and date_filter_col in filtered_df.columns:
        series = filtered_df[date_filter_col].dropna()
        if not series.empty:
            min_date = series.min().date()
            max_date = series.max().date()
            date_range = filter_col2.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="dashboard_date_range",
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                filtered_df = filtered_df[
                    (filtered_df[date_filter_col] >= start_date)
                    & (filtered_df[date_filter_col] <= end_date)
                ]

    category_filter_col = filter_col3.selectbox(
        "Category filter column",
        options=["None"] + category_cols,
        index=0,
        key="dashboard_category_col",
    )

    if category_filter_col != "None" and category_filter_col in filtered_df.columns:
        options = sorted(filtered_df[category_filter_col].dropna().astype(str).unique().tolist())
        selected = st.multiselect(
            f"Filter {category_filter_col}",
            options=options,
            default=options[: min(8, len(options))],
            key="dashboard_category_values",
        )
        if selected:
            filtered_df = filtered_df[filtered_df[category_filter_col].astype(str).isin(selected)]

    if filtered_df.empty:
        st.info("No records match the selected filters.")
        return

    axis_options = date_cols + category_cols
    if not axis_options:
        axis_options = [numeric_cols[0]]

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    x_axis = chart_col1.selectbox("X axis", options=axis_options, key="dashboard_x_axis")
    metric = chart_col2.selectbox("Metric", options=numeric_cols, key="dashboard_metric")
    agg = chart_col3.selectbox("Aggregation", options=["sum", "mean", "count"], key="dashboard_agg")

    if agg == "count":
        grouped = filtered_df.groupby(x_axis, dropna=False)[metric].count().reset_index(name=f"count_{metric}")
        y_col = f"count_{metric}"
    elif agg == "mean":
        grouped = filtered_df.groupby(x_axis, dropna=False)[metric].mean().reset_index(name=f"mean_{metric}")
        y_col = f"mean_{metric}"
    else:
        grouped = filtered_df.groupby(x_axis, dropna=False)[metric].sum().reset_index(name=f"sum_{metric}")
        y_col = f"sum_{metric}"

    if x_axis in date_cols:
        grouped = grouped.sort_values(x_axis)
        fig_main = px.line(grouped, x=x_axis, y=y_col, title=f"{agg.upper()} of {metric} by {x_axis}")
    else:
        grouped = grouped.sort_values(y_col, ascending=False).head(30)
        fig_main = px.bar(grouped, x=x_axis, y=y_col, title=f"{agg.upper()} of {metric} by {x_axis}")

    st.plotly_chart(fig_main, use_container_width=True)

    extra_col1, extra_col2 = st.columns(2)
    fig_hist = px.histogram(filtered_df, x=metric, nbins=30, title=f"Distribution of {metric}")
    extra_col1.plotly_chart(fig_hist, use_container_width=True)

    if len(numeric_cols) >= 2:
        second_metric = numeric_cols[1] if numeric_cols[0] == metric else numeric_cols[0]
        scatter = px.scatter(
            filtered_df,
            x=metric,
            y=second_metric,
            color=category_filter_col if category_filter_col != "None" else None,
            title=f"{metric} vs {second_metric}",
        )
        extra_col2.plotly_chart(scatter, use_container_width=True)
    else:
        extra_col2.info("Add another numeric column to enable scatter plot.")

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 Automated AI Data Analyst")
st.markdown("Upload your CSV or Excel file and let AI analyze it completely.")

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "last_pandas_code" not in st.session_state:
    st.session_state.last_pandas_code = ""
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "dashboard_df" not in st.session_state:
    st.session_state.dashboard_df = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# File upload
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls"])

qna_example = "Why did sales drop in March?"
pandas_example = "Show top 10 customers by revenue"

if uploaded_file:
    # Save to uploads/
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.success(f"File uploaded: {uploaded_file.name}")
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.analysis_results = None
        st.session_state.dashboard_df = None

    # Preview
    df = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)
    qna_example, pandas_example = _dataset_examples(df)
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Run pipeline
    if st.button("Run AI Analysis", type="primary"):
        try:
            with st.spinner("Running AI agents... this may take 1-2 minutes..."):
                results = run_pipeline(filepath)
            st.session_state.analysis_results = results
            cleaned_path = results.get("cleaned_data_path")
            if cleaned_path and os.path.exists(cleaned_path):
                st.session_state.dashboard_df = pd.read_csv(cleaned_path)
            else:
                st.session_state.dashboard_df = df.copy()
        except Exception as err:
            rate_msg = _rate_limit_message(err)
            if rate_msg:
                st.error(rate_msg)
            else:
                st.error(f"Analysis failed: {err}")
            st.stop()

        st.success("Analysis Complete!")

    results = st.session_state.get("analysis_results")
    if results:
        limit_candidates = [
            str(results.get("clean_report", "")),
            str(results.get("eda_results", {}).get("eda_summary", "")),
            str(results.get("insights", "")),
            str(results.get("viz_results", {}).get("viz_summary", "")),
        ]
        shared_limit_msg = ""
        for candidate in limit_candidates:
            shared_limit_msg = _extract_limit_message(candidate)
            if shared_limit_msg:
                break

        if shared_limit_msg:
            st.error(shared_limit_msg)

        # Tabs for results
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cleaning", "EDA", "Charts", "Insights", "Dashboard"])

        with tab1:
            st.subheader("Data Cleaning Report")
            clean_report_text = _remove_limit_message(str(results["clean_report"]))
            st.text(clean_report_text)

        with tab2:
            st.subheader("EDA Findings")
            eda_text = str(results["eda_results"]["eda_summary"])
            if _extract_limit_message(eda_text):
                st.info("See limit banner above.")
            else:
                st.write(eda_text)

        with tab3:
            st.subheader("Generated Charts")
            chart_paths = results["viz_results"]["chart_paths"]
            cols = st.columns(2)
            for i, path in enumerate(chart_paths):
                if os.path.exists(path):
                    cols[i % 2].image(path, use_container_width=True)

        with tab4:
            st.subheader("Business Insights")
            insights_text = str(results["insights"])
            if _extract_limit_message(insights_text):
                st.info("See limit banner above.")
            else:
                st.write(insights_text)

        with tab5:
            dashboard_df = st.session_state.get("dashboard_df")
            if isinstance(dashboard_df, pd.DataFrame):
                _render_interactive_dashboard(dashboard_df)
            else:
                st.info("Run AI Analysis to generate dashboard data.")

        # Download report
        report_path = results["report_path"]
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name="data_analysis_report.pdf",
                    mime="application/pdf"
                )

        # Download cleaned data
        cleaned_data_path = results.get("cleaned_data_path")
        if cleaned_data_path and os.path.exists(cleaned_data_path):
            with open(cleaned_data_path, "rb") as f:
                st.download_button(
                    label="Download Cleaned Data (CSV)",
                    data=f,
                    file_name=os.path.basename(cleaned_data_path),
                    mime="text/csv"
                )

# Natural language query
st.divider()
st.subheader("Ask a Question About Your Data")
_render_provider_badge()

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question_payload = st.chat_input(
    f"e.g. {qna_example}",
)

question = question_payload

if question is not None:
    if not uploaded_file:
        st.warning("Upload a file first to ask data questions.")
    elif not question.strip():
        st.warning("Enter a question before submitting.")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        context = df.head(20).to_string()
        can_run_structured_query = True
        answer_text = ""

        with st.chat_message("assistant"):
            try:
                answer_text = _invoke_llm_with_fallback(
                    f"Dataset sample:\n{context}\n\nQuestion: {question}\n"
                    "Answer with data-backed reasoning in plain English. "
                    "If a calculation is relevant, include only the formula in text form. "
                    "Do not provide Python code or code blocks."
                )
                st.write(answer_text)
            except BadRequestError as err:
                answer_text = f"Groq request failed. Update MODEL_NAME in .env.\n\nDetails: {err}"
                st.error(answer_text)
                can_run_structured_query = False
            except Exception as err:
                rate_msg = _rate_limit_message(err)
                if rate_msg:
                    answer_text = rate_msg
                    st.error(rate_msg)
                else:
                    answer_text = f"Could not answer the question: {err}"
                    st.error(answer_text)
                can_run_structured_query = False

        if answer_text:
            st.session_state.chat_messages.append({"role": "assistant", "content": answer_text})

        if can_run_structured_query:
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
                code_response_text = _invoke_llm_with_fallback(pandas_prompt)
                generated_code = _extract_python_code(code_response_text)
                st.session_state.last_pandas_code = generated_code

                formula_response_text = _invoke_llm_with_fallback(
                    "Convert this pandas logic into concise math/business formula text only. "
                    "No Python, no code blocks, no technical implementation details.\n\n"
                    f"Question: {question}\n"
                    f"Pandas logic:\n{generated_code}"
                )
                st.markdown("Formula used")
                st.write(formula_response_text)

                result, chart_cfg, exec_err = _run_generated_pandas_code(df, generated_code)
                if exec_err:
                    st.info("No structured table/chart result generated for this question.")
                elif result is None:
                    st.info("No structured table/chart result generated for this question.")
                else:
                    st.markdown("Query result")
                    if isinstance(result, pd.Series):
                        out_df = result.reset_index()
                        out_df.columns = ["index", "value"]
                        st.dataframe(out_df, use_container_width=True)
                    elif isinstance(result, pd.DataFrame):
                        st.dataframe(result, use_container_width=True)
                    else:
                        st.write(result)

                    st.markdown("Chart preview")
                    if isinstance(result, pd.Series):
                        st.bar_chart(result)
                    elif isinstance(result, pd.DataFrame):
                        chart_type = chart_cfg.get("chart") if chart_cfg else None
                        chart_x = chart_cfg.get("chart_x") if chart_cfg else None
                        chart_y = chart_cfg.get("chart_y") if chart_cfg else None

                        if chart_type in {"bar", "line"} and chart_x in result.columns and chart_y in result.columns:
                            chart_df = result[[chart_x, chart_y]].set_index(chart_x)
                            if chart_type == "line":
                                st.line_chart(chart_df)
                            else:
                                st.bar_chart(chart_df)
                        else:
                            numeric_cols = list(result.select_dtypes(include=np.number).columns)
                            if numeric_cols:
                                preview = result[numeric_cols].head(20)
                                st.bar_chart(preview)
                            else:
                                st.info("No numeric columns found for charting this result.")
            except BadRequestError as err:
                st.error(f"Groq request failed. Update MODEL_NAME in .env.\n\nDetails: {err}")
            except Exception as err:
                rate_msg = _rate_limit_message(err)
                if rate_msg:
                    st.error(rate_msg)
                else:
                    st.error(f"Could not process structured query output: {err}")