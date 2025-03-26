"""
backtest.py

This module provides functions for performing walk-forward backtesting
on time-series data, as well as evaluating the backtest performance.

Functions:
    - walk_forward_backtest: Performs a walk-forward backtest on the dataset.
    - evaluate_backtest: Computes overall and yearly metrics (precision, recall, F1)
      from the backtest predictions.
      
Usage:
    from src.backtest import walk_forward_backtest, evaluate_backtest

    # Example:
    backtest_df = walk_forward_backtest(data, model_path, features, target="Target", start=2500, step=250, threshold=0.5)
    metrics = evaluate_backtest(backtest_df, target="Target")
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

def walk_forward_backtest(data: pd.DataFrame, model_path: str, features: list, target: str = "Target",
                          start: int = 2500, step: int = 250, threshold: float = 0.5) -> pd.DataFrame:
    """
    Perform walk-forward backtesting on time-series data.
    
    Parameters:
        data (DataFrame): The complete dataset with features and target.
        model_path (str): Path to the saved model file (e.g., best_model_RandomForest.pkl).
        features (list): List of feature column names.
        target (str): Name of the target column.
        start (int): Number of initial rows to use as training data.
        step (int): Number of rows to move forward in each backtesting iteration.
        threshold (float): Threshold to convert predicted probabilities into binary predictions.
    
    Returns:
        DataFrame: A concatenated DataFrame containing actual target, predictions, and prediction probabilities.
    """
    all_predictions = []
    
    # Loop over the dataset with a rolling window
    for i in range(start, len(data), step):
        train = data.iloc[:i]
        test = data.iloc[i:i+step]
        
        # Load a fresh copy of the model for each iteration
        model = joblib.load(model_path)
        # Retrain the model on the training data up to the current iteration
        model.fit(train[features], train[target])
        
        # Predict probabilities for the test set and apply threshold
        probs = model.predict_proba(test[features])[:, 1]
        preds = (probs > threshold).astype(int)
        
        # Combine predictions with the test set
        temp_df = test.copy()
        temp_df["Prediction"] = preds
        temp_df["Prob_Up"] = probs
        all_predictions.append(temp_df)
    
    return pd.concat(all_predictions)


def evaluate_backtest(backtest_df: pd.DataFrame, target: str = "Target") -> dict:
    """
    Evaluate backtest predictions by computing overall precision, recall, and F1 score,
    as well as yearly metrics.
    
    Parameters:
        backtest_df (DataFrame): DataFrame containing actual target and predictions.
        target (str): Name of the target column.
    
    Returns:
        dict: A dictionary containing:
            - overall_precision: Precision score over all backtest data.
            - overall_recall: Recall score over all backtest data.
            - overall_f1: F1 score over all backtest data.
            - yearly_metrics: DataFrame with precision, recall, and F1 for each year.
    """
    overall_precision = precision_score(backtest_df[target], backtest_df["Prediction"], zero_division=0)
    overall_recall = recall_score(backtest_df[target], backtest_df["Prediction"], zero_division=0)
    overall_f1 = f1_score(backtest_df[target], backtest_df["Prediction"], zero_division=0)
    
    # Ensure a 'Year' column exists in the index
    if "Year" not in backtest_df.columns:
        backtest_df["Year"] = backtest_df.index.year
    
    yearly_metrics = backtest_df.groupby("Year").apply(
        lambda x: pd.Series({
            "Precision": precision_score(x[target], x["Prediction"], zero_division=0),
            "Recall": recall_score(x[target], x["Prediction"], zero_division=0),
            "F1": f1_score(x[target], x["Prediction"], zero_division=0)
        })
    )
    
    return {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "yearly_metrics": yearly_metrics
    }


# Example usage for testing the module
if __name__ == "__main__":
    # For demonstration, load the processed model-ready dataset.
    # In your actual project, ensure the file path is correct.
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from src/
    data_path = os.path.join(ROOT_DIR, "data", "processed", "sp500_cleaned_model_ready.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    features = [col for col in data.columns if col not in ["Tomorrow", "Target"]]
    target = "Target"
    
    # Define model path (adjust based on your best model)
    model_path = "../models/best_model_RandomForest.pkl"
    
    # Perform walk-forward backtesting
    backtest_results = walk_forward_backtest(data, model_path, features, target, start=2500, step=250, threshold=0.5)
    print("Backtest completed. Total rows in backtest results:", len(backtest_results))
    
    # Evaluate backtest results
    metrics = evaluate_backtest(backtest_results, target)
    print("\n=== Overall Backtest Metrics ===")
    print(f"Precision: {metrics['overall_precision']:.4f}")
    print(f"Recall:    {metrics['overall_recall']:.4f}")
    print(f"F1 Score:  {metrics['overall_f1']:.4f}")
    print("\n=== Yearly Metrics ===")
    print(metrics["yearly_metrics"])
    
    # Optionally, save backtest results to CSV
    os.makedirs("../data/processed", exist_ok=True)
    backtest_results.to_csv("../data/processed/backtest_predictions.csv")
    print("Backtest predictions saved to ../data/processed/backtest_predictions.csv")
