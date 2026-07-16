# Visualization tools for creating charts and graphs
# Uses matplotlib and seaborn to generate data visualizations
import os

import matplotlib

matplotlib.use("Agg", force=True)  # Use Agg backend to save charts without display

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Create folder to save chart images
OUTPUT_DIR = "outputs/charts/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Helper function to save chart to file
def save_chart(filename: str):
    # Create full path and save the chart
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    return path


# Create distribution histograms for numeric columns
def plot_histogram(df: pd.DataFrame) -> list:
    paths = []
    # Create histogram for each numeric column
    for col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True, color="steelblue")  # kde shows smooth curve
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        path = save_chart(f"hist_{col}.png")
        paths.append(path)
    return paths


# Create bar charts for categorical columns
def plot_bar(df: pd.DataFrame) -> list:
    paths = []
    # Create bar chart for each text column
    for col in df.select_dtypes(include="object").columns:
        # Only if not too many unique values
        if df[col].nunique() <= 20:
            plt.figure(figsize=(8, 4))
            df[col].value_counts().plot(kind="bar", color="coral", edgecolor="black")
            plt.title(f"Bar Chart: {col}")
            plt.xticks(rotation=45)
            path = save_chart(f"bar_{col}.png")
            paths.append(path)
    return paths


# Create correlation heatmap showing relationships between numbers
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    # Need at least 2 columns to show correlation
    if numeric_df.shape[1] < 2:
        return None
    plt.figure(figsize=(10, 7))
    # Create heatmap with correlation values
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    return save_chart("heatmap.png")


# Create line charts for time series or trends
# Create line charts for time series data
def plot_line(df: pd.DataFrame) -> list:
    paths = []
    # Find date columns and numeric columns
    date_cols = df.select_dtypes(include=["datetime64"]).columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    # Create line chart for each numeric column over each date column
    for dcol in date_cols:
        for ncol in numeric_cols:
            plt.figure(figsize=(9, 4))
            # Sort by date and plot
            df.sort_values(dcol).plot(x=dcol, y=ncol, ax=plt.gca(), color="green")
            plt.title(f"{ncol} over {dcol}")
            path = save_chart(f"line_{ncol}_{dcol}.png")
            paths.append(path)
    return paths


# Create boxplots to show data distribution and outliers
def plot_boxplot(df: pd.DataFrame) -> list:
    paths = []
    # Create boxplot for each numeric column
    for col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(6, 4))
        # Boxplot shows quartiles and outliers
        sns.boxplot(y=df[col].dropna(), color="lightgreen")
        plt.title(f"Boxplot: {col}")
        path = save_chart(f"box_{col}.png")
        paths.append(path)
    return paths
