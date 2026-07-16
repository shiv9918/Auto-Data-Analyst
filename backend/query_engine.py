import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agents.llm_fallback import is_rate_limit_error, build_limit_exceeded_message


def extract_python_code(text: str) -> str:
    code_match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    return text.strip()


def validate_generated_code(code: str) -> tuple[bool, str]:
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


def run_generated_pandas_code(df: pd.DataFrame, code: str):
    is_valid, err = validate_generated_code(code)
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


def rate_limit_message(err: Exception) -> str:
    if is_rate_limit_error(err):
        return build_limit_exceeded_message(err)
    return ""


def extract_limit_message(text: str) -> str:
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


def remove_limit_message(text: str) -> str:
    limit_msg = extract_limit_message(text)
    if not limit_msg:
        return text
    cleaned = text.replace(f"\n\nAI Summary:\n{limit_msg}", "")
    cleaned = cleaned.replace(limit_msg, "")
    return cleaned.strip()


def invoke_llm_with_fallback(prompt: str, label: str = "request") -> str:
    import time

    from langchain_core.messages import SystemMessage, HumanMessage

    load_dotenv()

    groq_model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    system_message = """You are a data analyst assistant. When answering questions about data:
1. Include relevant formulas/equations (use markdown notation like `formula` or math expressions)
2. Keep your response concise and to-the-point (max 3-4 sentences unless asked for details)
3. Lead with the answer, then briefly explain the approach
4. Use bullet points for multiple items
5. Show calculations only when necessary for clarity"""

    groq_llm = ChatGroq(model=groq_model, api_key=groq_key)

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt),
    ]

    print(f"  [LLM] -> calling Groq ({label}) | model={groq_model} | prompt_chars={len(prompt)}")
    start = time.perf_counter()
    response = groq_llm.invoke(messages).content
    elapsed = time.perf_counter() - start
    print(f"  [LLM] <- Groq responded ({label}) | {elapsed:.2f}s | response_chars={len(response)}")

    return response


def dataset_examples(df: pd.DataFrame) -> tuple[str, str]:
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


def prepare_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    dashboard_df = df.copy()
    for col in dashboard_df.columns:
        if dashboard_df[col].dtype == "object":
            parsed = pd.to_datetime(dashboard_df[col], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                dashboard_df[col] = parsed
    return dashboard_df
