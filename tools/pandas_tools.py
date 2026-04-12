import pandas as pd
import numpy as np

def load_file(filepath: str) -> pd.DataFrame:
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith((".xlsx", ".xls")):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Only CSV and Excel files supported.")

def get_basic_info(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": list(df.select_dtypes(include=np.number).columns),
        "categorical_columns": list(df.select_dtypes(include="object").columns),
    }

def get_statistics(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] == 0:
        return {
            "describe": {},
            "skewness": {},
            "kurtosis": {},
        }
    return {
        "describe": numeric_df.describe().to_dict(),
        "skewness": numeric_df.skew().to_dict(),
        "kurtosis": numeric_df.kurt().to_dict(),
    }

def detect_outliers(df: pd.DataFrame) -> dict:
    outliers = {}
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        count = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
        if count > 0:
            outliers[col] = count
    return outliers

def get_correlation(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return {}
    return numeric_df.corr().to_dict()