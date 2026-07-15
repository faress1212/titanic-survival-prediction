"""
Model training and evaluation utilities for the Titanic Survival
Prediction project.

Class imbalance is handled via `class_weight="balanced"` where supported,
instead of a resampling technique like SMOTE, to keep things simple.
"""

import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.utils.config import MODELS_DIR, RANDOM_STATE


def get_models(random_state: int = RANDOM_STATE) -> dict:
    """
    Build the dictionary of candidate classification models.

    `class_weight="balanced"` is used on the models that support it, so the
    models automatically account for any class imbalance in 'survived'
    without needing a separate resampling step (e.g. SMOTE).

    Args:
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Mapping of model name -> unfitted scikit-learn estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            random_state=random_state, class_weight="balanced"
        ),
        "SVM": SVC(class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test) -> dict:
    """
    Fit each model and compute its accuracy on the test set.

    Args:
        models (dict): Mapping of model name -> unfitted estimator.
        X_train, X_test, y_train, y_test: Train/test split data.

    Returns:
        dict: Mapping of model name -> accuracy score.
    """
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)

    return results


def save_model(model, name: str, models_dir: str = MODELS_DIR) -> str:
    """
    Save a fitted model to disk as a .pkl file.

    Args:
        model: Fitted scikit-learn estimator.
        name (str): Model name, used to build the output filename.
        models_dir (str): Directory to save the model into.

    Returns:
        str: Path to the saved model file.
    """
    os.makedirs(models_dir, exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(models_dir, f"{safe_name}.pkl")

    with open(path, "wb") as f:
        pickle.dump(model, f)

    return path
