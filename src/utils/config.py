"""
Central configuration for the Titanic Survival Prediction project.

Simple constants module (no YAML) holding file paths and shared settings
used across the pipeline.
"""

import os

# --- Project paths -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

MODELS_DIR = os.path.join(BASE_DIR, "models")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# --- Data settings -------------------------------------------------------
# The dataset is loaded via seaborn's built-in Titanic loader, so no raw
# CSV file is required. RAW_DATA_DIR is kept for consistency with the
# project structure and for anyone who wants to cache a local copy.
LEAKAGE_COLUMNS = ["alive"]
CATEGORICAL_COLUMNS = ["sex", "embarked", "class", "who", "adult_male", "alone", "embark_town"]
TARGET_COLUMN = "survived"

# --- Model settings ------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    """Create all project output directories if they don't already exist."""
    for directory in (RAW_DATA_DIR, MODELS_DIR, FIGURES_DIR):
        os.makedirs(directory, exist_ok=True)
