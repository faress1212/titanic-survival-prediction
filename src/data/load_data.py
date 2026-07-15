"""
Data loading and quality-checking utilities for the Titanic Survival
Prediction project.

Dataset: Titanic passenger data, loaded via seaborn's built-in dataset
loader (a cleaned version of the classic Kaggle Titanic dataset).
"""

import pandas as pd
import seaborn as sns


def load_data() -> pd.DataFrame:
    """
    Load the Titanic dataset via seaborn's built-in dataset loader.

    Returns:
        pd.DataFrame: Raw Titanic passenger dataframe.
    """
    return sns.load_dataset("titanic")


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Print basic data quality checks: shape, info, missing values.

    Args:
        df (pd.DataFrame): Input dataframe.
    """
    print("Shape:", df.shape)
    print("\nInfo:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Titanic dataframe:
    - Fill missing 'age' with the median
    - Drop the 'deck' column (too many missing values)
    - Fill missing 'embarked' / 'embark_town' with the mode

    Args:
        df (pd.DataFrame): Raw Titanic dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()

    df["age"] = df["age"].fillna(df["age"].median())
    df = df.drop(columns=["deck"], errors="ignore")
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

    return df


def remove_leakage_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Drop columns that leak the target variable.

    'alive' is just a string version of 'survived' (yes/no), so keeping it
    would let the model "cheat" instead of learning from real features.

    Args:
        df (pd.DataFrame): Cleaned Titanic dataframe.
        columns (list, optional): Column names to drop. Defaults to
                                   config.LEAKAGE_COLUMNS.

    Returns:
        pd.DataFrame: Dataframe with leakage columns removed.
    """
    from src.utils.config import LEAKAGE_COLUMNS

    columns = columns or LEAKAGE_COLUMNS
    return df.drop(columns=columns, errors="ignore")
