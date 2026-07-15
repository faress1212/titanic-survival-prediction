# Titanic Survival Prediction 🚢

An end-to-end machine learning project predicting passenger survival on the Titanic, covering data cleaning, EDA, feature engineering, and a comparison of 4 classification models.

## 📌 Project Overview

This project builds a complete ML pipeline to predict whether a Titanic passenger survived, based on features like class, gender, age, and family size. It covers:

- Data cleaning (missing values, high-null columns)
- Removing target leakage (the `alive` column directly encodes the target)
- Exploratory Data Analysis (survival by gender, class, age, fare distributions)
- Feature engineering (`family_size`)
- Training and comparing 4 classification models, using `class_weight="balanced"` to handle any class imbalance (no SMOTE, to keep things simple)
- Saving trained models and generated figures to disk

## 🗂️ Project Structure

```
titanic-survival-prediction/
├── data/
│   ├── raw/                        # Optional local dataset cache
│   └── README.md
├── src/
│   ├── data/
│   │   └── load_data.py            # Loading + data quality checks + cleaning
│   ├── features/
│   │   └── preprocessing.py        # Feature engineering + train/test split (no SMOTE)
│   ├── models/
│   │   └── train_models.py         # Model training, evaluation, saving
│   ├── visualization/
│   │   └── eda_plots.py            # EDA plots
│   └── utils/
│       └── config.py               # Paths and shared settings (plain Python, no YAML)
├── models/                          # Saved trained models (.pkl)
├── reports/figures/                 # Saved EDA plots
├── main.py                          # Runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## 📊 Dataset

The Titanic dataset is loaded directly via `seaborn.load_dataset("titanic")` — a cleaned version of the classic [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic). It contains 891 passenger records with demographic and ticket information. See `data/README.md` for details.

## ⚠️ A Note on Data Leakage

The raw dataset includes an `alive` column — a plain string version of the `survived` target ("yes"/"no"). Training on this column would let a model achieve near-perfect accuracy without learning anything meaningful. **This project explicitly drops `alive` before training** to keep the results honest.

## 🤖 Models Compared

| Model               | Description                          |
|---------------------|---------------------------------------|
| Logistic Regression | Linear baseline classifier            |
| Random Forest       | Ensemble of decision trees            |
| SVM                 | Support Vector Machine classifier     |
| KNN                 | Instance-based classifier             |

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python main.py
```

This will:
- Load and clean the data
- Generate EDA plots into `reports/figures/`
- Train and evaluate 4 models
- Save each trained model into `models/`

## 📈 Results

All four models achieve roughly 80-83% test accuracy after cleaning the data and removing the leakage column — consistent with widely published benchmarks on this dataset. Gender and passenger class are the strongest predictors, reflecting the historical "women and children first" evacuation priority.

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
