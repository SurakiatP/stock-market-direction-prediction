"""
evaluate.py

This module provides functions for evaluating machine learning models,
including backtest evaluation, plotting metrics, and threshold sensitivity analysis.
Functions included:
    - evaluate_overall: Calculate overall precision, recall, and F1 score.
    - evaluate_yearly: Compute yearly metrics (precision, recall, F1) from backtest predictions.
    - plot_yearly_metrics: Plot yearly evaluation metrics.
    - plot_confidence_histogram: Plot the histogram of predicted probabilities with a threshold marker.
    - plot_actual_vs_predicted: Plot actual vs predicted values over time.
    - threshold_sensitivity_test: Evaluate performance over a range of thresholds.
    - plot_threshold_sensitivity: Plot threshold sensitivity results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate_overall(y_true, y_pred):
    """
    Compute overall precision, recall, and F1 score.
    
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        
    Returns:
        dict: Dictionary containing precision, recall, f1 score, and classification report.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    
    return {"precision": precision, "recall": recall, "f1": f1, "report": report}

def evaluate_yearly(backtest_df, target="Target"):
    """
    Compute evaluation metrics (precision, recall, f1) for each year.
    
    Parameters:
        backtest_df (DataFrame): DataFrame containing predictions with an index of datetime.
        target (str): Target column name.
        
    Returns:
        DataFrame: Yearly metrics as a DataFrame.
    """
    # Ensure there is a 'Year' column
    if "Year" not in backtest_df.columns:
        backtest_df["Year"] = backtest_df.index.year
    
    yearly_stats = backtest_df.groupby("Year").apply(
        lambda x: pd.Series({
            "Precision": precision_score(x[target], x["Prediction"], zero_division=0),
            "Recall": recall_score(x[target], x["Prediction"], zero_division=0),
            "F1": f1_score(x[target], x["Prediction"], zero_division=0)
        })
    )
    return yearly_stats

def plot_yearly_metrics(yearly_stats, output_path="../reports/images/model/backtest_yearly_metrics.png"):
    """
    Plot yearly evaluation metrics and save the plot.
    
    Parameters:
        yearly_stats (DataFrame): DataFrame containing yearly metrics.
        output_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_stats.index, yearly_stats["Precision"], marker='o', label="Precision")
    plt.plot(yearly_stats.index, yearly_stats["Recall"], marker='o', label="Recall")
    plt.plot(yearly_stats.index, yearly_stats["F1"], marker='o', label="F1")
    plt.title("Yearly Backtest Metrics")
    plt.xlabel("Year")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_confidence_histogram(y_proba, threshold=0.5, output_path="../reports/images/model/unseen_confidence_hist.png"):
    """
    Plot a histogram of predicted probabilities with a vertical line for the threshold.
    
    Parameters:
        y_proba (array-like): Predicted probabilities for class 1.
        threshold (float): Threshold value used for classification.
        output_path (str): File path to save the histogram image.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(y_proba, bins=20, edgecolor="black")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold}")
    plt.title("Unseen Data - Confidence Distribution")
    plt.xlabel("Probability of Price Going Up")
    plt.ylabel("Count")
    plt.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_actual_vs_predicted(df, target="Target", output_path="../reports/images/model/unseen_prediction_vs_actual.png"):
    """
    Plot actual vs predicted target values over time.
    
    Parameters:
        df (DataFrame): DataFrame containing actual target and predictions. The index should be datetime.
        target (str): Name of the target column.
        output_path (str): File path to save the plot.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df[target], label="Actual", alpha=0.7)
    plt.plot(df.index, df["Prediction"], label="Predicted", alpha=0.7)
    plt.title("Unseen Data - Actual vs Predicted")
    plt.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def threshold_sensitivity_test(y_true, y_proba, thresholds=np.arange(0.3, 0.7, 0.05)):
    """
    Evaluate precision, recall, and F1 score over a range of thresholds.
    
    Parameters:
        y_true (array-like): True target values.
        y_proba (array-like): Predicted probabilities for class 1.
        thresholds (array-like): Array of threshold values to test.
    
    Returns:
        DataFrame: A DataFrame summarizing precision, recall, and F1 for each threshold.
    """
    sens_results = []
    for t in thresholds:
        y_pred_t = (y_proba > t).astype(int)
        prec_t = precision_score(y_true, y_pred_t, zero_division=0)
        rec_t = recall_score(y_true, y_pred_t, zero_division=0)
        f1_t = f1_score(y_true, y_pred_t, zero_division=0)
        sens_results.append({"Threshold": t, "Precision": prec_t, "Recall": rec_t, "F1": f1_t})
    
    return pd.DataFrame(sens_results)

def plot_threshold_sensitivity(sens_df, output_path="../reports/images/model/unseen_threshold_sensitivity.png"):
    """
    Plot threshold sensitivity results.
    
    Parameters:
        sens_df (DataFrame): DataFrame containing threshold, precision, recall, and F1 scores.
        output_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(sens_df["Threshold"], sens_df["Precision"], marker='o', label="Precision")
    plt.plot(sens_df["Threshold"], sens_df["Recall"], marker='o', label="Recall")
    plt.plot(sens_df["Threshold"], sens_df["F1"], marker='o', label="F1")
    plt.title("Threshold Sensitivity on Unseen Data")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

# If you want to test the functions in this module, you can add a main block.
if __name__ == "__main__":
    # Example test with dummy data:
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=1000, n_features=20, random_state=42)
    y_proba_dummy = np.random.rand(1000)
    
    # Evaluate overall dummy metrics
    dummy_metrics = {
        "precision": precision_score(y_dummy, (y_proba_dummy > 0.5).astype(int), zero_division=0),
        "recall": recall_score(y_dummy, (y_proba_dummy > 0.5).astype(int), zero_division=0),
        "f1": f1_score(y_dummy, (y_proba_dummy > 0.5).astype(int), zero_division=0)
    }
    print("Dummy overall metrics:", dummy_metrics)
    
    # Run threshold sensitivity test
    sens_df = threshold_sensitivity_test(y_dummy, y_proba_dummy)
    print(sens_df)
    
    # Plot dummy threshold sensitivity
    plot_threshold_sensitivity(sens_df, output_path="dummy_threshold_sensitivity.png")
    
    # Plot dummy confidence histogram
    plot_confidence_histogram(y_proba_dummy, threshold=0.5, output_path="dummy_confidence_hist.png")
    
    print("Dummy evaluation plots saved.")
