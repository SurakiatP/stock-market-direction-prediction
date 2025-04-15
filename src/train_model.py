# src/train_model.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, classification_report, confusion_matrix

from utils import load_dataset, get_feature_columns, create_directory
import mlflow
import mlflow.sklearn

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from src/
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "sp500_cleaned_model_ready.csv")
MODEL_DIR = "../models"
REPORT_CSV = "../reports/model_training_results.csv"
REPORT_IMG = "../reports/images/model/precision_comparison.png"
TRAIN_CUTOFF = "2024-01-01"


def load_and_split_data(path: str, cutoff: str):
    df = load_dataset(path)
    features = get_feature_columns(df)
    X = df[features]
    y = df["Target"]

    X_train = X[df.index < cutoff]
    y_train = y[df.index < cutoff]
    X_test = X[df.index >= cutoff]
    y_test = y[df.index >= cutoff]

    return X_train, y_train, X_test, y_test, features


def get_models():
    return {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "params": {
                "n_estimators": [100, 200],
                "min_samples_split": [2, 10]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(solver='liblinear', class_weight='balanced'),
            "params": {
                "C": [0.1, 1, 10]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "params": {
                "n_estimators": [100],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1]
            }
        }
    }


def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = get_models()
    best_models = []
    results = []

    mlflow.set_experiment("stock-market-direction")

    for name, config in models.items():
        print(f"\n Training {name} ...")

        with mlflow.start_run(run_name=name):
            grid = GridSearchCV(
                config["model"],
                config["params"],
                cv=5,
                scoring="precision",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            precision = precision_score(y_test, y_pred, zero_division=0)

            print(f" Best Params: {grid.best_params_}")
            print(f" Precision: {precision:.4f}")
            print(" Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

            # MLflow Logging
            mlflow.log_param("model_type", name)
            for param_name, param_value in grid.best_params_.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_metric("precision", precision)
            mlflow.sklearn.log_model(best_model, "model")

            results.append({
                "Model": name,
                "Best Precision": precision,
                "Best Params": grid.best_params_
            })
            best_models.append((name, best_model))

    return results, best_models


def save_best_model(results, best_models):
    results_df = pd.DataFrame(results).sort_values(by="Best Precision", ascending=False)
    best_name = results_df.iloc[0]["Model"]
    best_model = dict(best_models)[best_name]

    create_directory(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, f"best_model_{best_name}.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n Best model '{best_name}' saved to {model_path}")

    return results_df, best_name


def plot_results(results_df):
    create_directory(os.path.dirname(REPORT_IMG))
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Model", y="Best Precision", data=results_df)
    plt.title("Model Comparison - Precision on Test Set")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(REPORT_IMG)
    plt.show()
    print(f" Chart saved to {REPORT_IMG}")

    # Log to MLflow
    with mlflow.start_run(run_name="Model Comparison (Chart)") as run:
        mlflow.log_artifact(REPORT_IMG, artifact_path="charts")

def main():
    print(f"\n Loading data from: {DATA_PATH}")
    X_train, y_train, X_test, y_test, features = load_and_split_data(DATA_PATH, TRAIN_CUTOFF)

    print(f" Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    results, best_models = train_and_evaluate(X_train, y_train, X_test, y_test)

    results_df, best_name = save_best_model(results, best_models)

    create_directory(os.path.dirname(REPORT_CSV))
    results_df.to_csv(REPORT_CSV, index=False)
    print(f" Model training results saved to {REPORT_CSV}")

    # Log chart to MLflow
    plot_results(results_df)

    print("\n Training pipeline completed.")


if __name__ == "__main__":
    main()
