import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "outputs/charts/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_chart(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def plot_histogram(df: pd.DataFrame) -> list:
    paths = []
    for col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True, color="steelblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        path = save_chart(f"hist_{col}.png")
        paths.append(path)
    return paths

def plot_bar(df: pd.DataFrame) -> list:
    paths = []
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() <= 20:
            plt.figure(figsize=(8, 4))
            df[col].value_counts().plot(kind="bar", color="coral", edgecolor="black")
            plt.title(f"Bar Chart: {col}")
            plt.xticks(rotation=45)
            path = save_chart(f"bar_{col}.png")
            paths.append(path)
    return paths

def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None
    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    return save_chart("heatmap.png")

def plot_line(df: pd.DataFrame) -> list:
    paths = []
    date_cols = df.select_dtypes(include=["datetime64"]).columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for dcol in date_cols:
        for ncol in numeric_cols:
            plt.figure(figsize=(9, 4))
            df.sort_values(dcol).plot(x=dcol, y=ncol, ax=plt.gca(), color="green")
            plt.title(f"{ncol} over {dcol}")
            path = save_chart(f"line_{ncol}_{dcol}.png")
            paths.append(path)
    return paths

def plot_boxplot(df: pd.DataFrame) -> list:
    paths = []
    for col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=df[col].dropna(), color="lightgreen")
        plt.title(f"Boxplot: {col}")
        path = save_chart(f"box_{col}.png")
        paths.append(path)
    return paths