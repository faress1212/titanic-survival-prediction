"""
Exploratory Data Analysis (EDA) plotting utilities for the Titanic
Survival Prediction project.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import FIGURES_DIR


def _save_and_show(save_path: str = None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_survival_count(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot the overall count of survivors vs. non-survivors.

    Args:
        df (pd.DataFrame): Input dataframe with a 'survived' column.
        save_path (str, optional): If provided, save the figure to this path.
    """
    survival_counts = df["survived"].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(["Not Survived", "Survived"], survival_counts.values)
    plt.title("Survival Count")
    plt.xlabel("Survival")
    plt.ylabel("Number of Passengers")
    _save_and_show(save_path)


def plot_survival_by_gender(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot survival counts split by gender.

    Args:
        df (pd.DataFrame): Input dataframe with 'sex' and 'survived' columns.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="sex", hue="survived")
    plt.title("Survival by Gender")
    _save_and_show(save_path)


def plot_survival_by_class(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot survival counts split by passenger class.

    Args:
        df (pd.DataFrame): Input dataframe with 'class' and 'survived' columns.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="class", hue="survived")
    plt.title("Survival by Passenger Class")
    _save_and_show(save_path)


def plot_age_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot the age distribution of passengers.

    Args:
        df (pd.DataFrame): Input dataframe with an 'age' column.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df["age"], bins=30, kde=True)
    plt.title("Age Distribution")
    _save_and_show(save_path)


def plot_fare_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot the fare distribution of passengers.

    Args:
        df (pd.DataFrame): Input dataframe with a 'fare' column.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(df["fare"], bins=20, edgecolor="black")
    plt.title("Fare Distribution")
    plt.xlabel("Fare")
    plt.ylabel("Frequency")
    _save_and_show(save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot a correlation heatmap for all numeric columns.

    Args:
        df (pd.DataFrame): Input dataframe (already numerically encoded).
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    _save_and_show(save_path)


def generate_all_plots(df_raw: pd.DataFrame, df_encoded: pd.DataFrame, figures_dir: str = FIGURES_DIR) -> None:
    """
    Generate and save the full set of EDA plots in one call.

    Args:
        df_raw (pd.DataFrame): Cleaned but not-yet-encoded dataframe
                                (used for gender/class/fare/age plots, which
                                read better with original category labels).
        df_encoded (pd.DataFrame): Numerically encoded dataframe (used for
                                    the correlation heatmap).
        figures_dir (str): Directory to save figures into.
    """
    os.makedirs(figures_dir, exist_ok=True)

    plot_survival_count(df_raw, os.path.join(figures_dir, "survival_count.png"))
    plot_survival_by_gender(df_raw, os.path.join(figures_dir, "survival_by_gender.png"))
    plot_survival_by_class(df_raw, os.path.join(figures_dir, "survival_by_class.png"))
    plot_age_distribution(df_raw, os.path.join(figures_dir, "age_distribution.png"))
    plot_fare_distribution(df_raw, os.path.join(figures_dir, "fare_distribution.png"))
    plot_correlation_heatmap(df_encoded, os.path.join(figures_dir, "correlation_heatmap.png"))
