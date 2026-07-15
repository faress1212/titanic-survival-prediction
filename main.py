"""
End-to-end pipeline for the Titanic Survival Prediction project.

Run with:
    python main.py
"""

import os

from src.data.load_data import (
    check_data_quality,
    clean_data,
    load_data,
    remove_leakage_columns,
)
from src.features.preprocessing import add_family_size, encode_categorical_columns, split_and_scale
from src.models.train_models import get_models, save_model, train_and_evaluate
from src.utils.config import TARGET_COLUMN, ensure_dirs
from src.visualization.eda_plots import generate_all_plots


def main():
    ensure_dirs()

    # 1. Load & inspect data
    df = load_data()
    check_data_quality(df)

    # 2. Clean data (handle missing values, drop high-null columns)
    df = clean_data(df)

    # 3. Remove target-leakage columns (e.g. 'alive' directly encodes 'survived')
    df = remove_leakage_columns(df)

    # 4. EDA plots (uses the cleaned-but-not-yet-encoded dataframe, plus an
    #    encoded copy for the correlation heatmap)
    df_encoded_for_eda = encode_categorical_columns(df)
    generate_all_plots(df_raw=df, df_encoded=df_encoded_for_eda)

    # 5. Feature engineering
    df = encode_categorical_columns(df)
    df = add_family_size(df)

    # 6. Train / test split + scaling
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # 7. Train & evaluate models
    models = get_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print("\nModel accuracy comparison:")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")

    best_model_name = max(results, key=results.get)
    print(f"\nBest model: {best_model_name} ({results[best_model_name]:.4f})")

    # 8. Save all trained models to disk
    print("\nSaving models:")
    for name, model in models.items():
        path = save_model(model, name)
        print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()
