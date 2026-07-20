# Utility tools for data analysis using pandas
# Provides functions for loading, analyzing, and summarizing data
import os

import pandas as pd
import numpy as np

# Above this file size, CSVs are read in chunks to keep parser peak memory down.
LARGE_FILE_THRESHOLD_MB = 200

# pandas >=3 defaults text columns to a new "str" dtype instead of "object".
# select_dtypes(include="object") still matches it today for backward
# compatibility (with a deprecation warning), but pandas <3 raises a TypeError
# if "str" is passed to select_dtypes at all. Probe once so text-column
# detection stays correct - and warning-free - on both major versions.
try:
    pd.DataFrame({"_probe": ["x"]}).select_dtypes(include=["object", "str"])
    _TEXT_DTYPES = ["object", "str"]
except TypeError:
    _TEXT_DTYPES = ["object"]


# Load CSV or Excel files into a dataframe
def load_file(filepath: str, chunksize: int = None) -> pd.DataFrame:
    if filepath.endswith(".csv"):
        if chunksize is None:
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
            except OSError:
                size_mb = 0
            if size_mb > LARGE_FILE_THRESHOLD_MB:
                chunksize = 200_000
        if chunksize:
            return pd.concat(pd.read_csv(filepath, chunksize=chunksize), ignore_index=True)
        return pd.read_csv(filepath)
    elif filepath.endswith((".xlsx", ".xls")):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Only CSV and Excel files supported.")


# Get basic information about the dataframe
def get_basic_info(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,  # Number of rows and columns
        "columns": list(df.columns),  # Column names
        "dtypes": df.dtypes.astype(str).to_dict(),  # Data types
        "missing_values": df.isnull().sum().to_dict(),  # Missing values per column
        "duplicate_rows": int(df.duplicated().sum()),  # Number of duplicate rows
        "numeric_columns": list(df.select_dtypes(include=np.number).columns),  # Number columns
        "categorical_columns": list(df.select_dtypes(include=_TEXT_DTYPES).columns),  # Text columns
    }


# Calculate statistical measures for numeric columns
def get_statistics(df: pd.DataFrame) -> dict:
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    # If no numeric columns, return empty stats
    if numeric_df.shape[1] == 0:
        return {
            "describe": {},
            "skewness": {},
            "kurtosis": {},
        }
    # Return statistical summary: mean, median, std, min, max etc.
    return {
        "describe": numeric_df.describe().to_dict(),  # Basic stats
        "skewness": numeric_df.skew().to_dict(),  # How skewed the distribution is
        "kurtosis": numeric_df.kurt().to_dict(),  # How peaked the distribution is
    }


# Find outliers (values that are far from the normal range) using IQR method
def detect_outliers(df: pd.DataFrame) -> dict:
    outliers = {}
    # Check each numeric column
    for col in df.select_dtypes(include=np.number).columns:
        # Calculate quartiles and interquartile range
        Q1 = df[col].quantile(0.25)  # 25th percentile
        Q3 = df[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1  # Distance between quartiles
        # Count values outside normal range (more than 1.5*IQR from quartiles)
        count = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
        if count > 0:
            outliers[col] = count
    return outliers


# Calculate correlation between numeric columns
def get_correlation(df: pd.DataFrame) -> dict:
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    # Need at least 2 columns to have correlations
    if numeric_df.shape[1] < 2:
        return {}
    # Return correlation matrix as dictionary
    return numeric_df.corr().to_dict()
