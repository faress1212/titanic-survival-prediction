"""
Feature engineering and train/test split utilities for the Titanic
Survival Prediction project.

No SMOTE / class-balancing here on purpose: class imbalance (if any) is
handled downstream via `class_weight="balanced"` inside the models
themselves, to keep the pipeline simple.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import CATEGORICAL_COLUMNS, RANDOM_STATE, TEST_SIZE


def encode_categorical_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Label-encode categorical columns so they can be used by ML models.

    Note: a separate LabelEncoder is fit per column (rather than one shared
    encoder reused across columns), since each column has its own distinct
    set of categories.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list, optional): Categorical column names to encode.
                                   Defaults to config.CATEGORICAL_COLUMNS.

    Returns:
        pd.DataFrame: Dataframe with the given columns label-encoded.
    """
    df = df.copy()
    columns = columns or CATEGORICAL_COLUMNS

    for col in columns:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'family_size' feature: siblings/spouses + parents/children + self.

    Args:
        df (pd.DataFrame): Input dataframe with 'sibsp' and 'parch' columns.

    Returns:
        pd.DataFrame: Dataframe with an added 'family_size' column.
    """
    df = df.copy()
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    return df


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Split features/target into train/test sets, then standardize features.

    The scaler is fit only on the training data and applied to both train
    and test sets, to avoid leaking test-set statistics into training.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test, fitted_scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
