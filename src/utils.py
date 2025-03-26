"""
utils.py

This module contains utility functions used throughout the project.

Functions:
    - load_dataset: Load a CSV dataset.
    - get_feature_columns: Get a list of feature columns excluding specified columns.
    - load_model: Load a saved model from a pickle file.
    - compute_classification_metrics: Compute precision, recall, F1 score, and classification report.
    - create_directory: Ensure a directory exists.
    - plot_and_save: Plot a figure and save it to a specified file.
"""

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def load_dataset(path: str, index_col=0, parse_dates=True) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    
    Parameters:
        path (str): The path to the CSV file.
        index_col (int or str): Column to use as the index.
        parse_dates (bool): Whether to parse dates.
        
    Returns:
        DataFrame: The loaded dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)


def get_feature_columns(df: pd.DataFrame, exclude: list = ["Tomorrow", "Target"]) -> list:
    """
    Return a list of feature columns, excluding the specified columns.
    
    Parameters:
        df (DataFrame): The input DataFrame.
        exclude (list): List of columns to exclude.
        
    Returns:
        list: List of feature column names.
    """
    return [col for col in df.columns if col not in exclude]


def load_model(model_path: str):
    """
    Load a saved model from a pickle file.
    
    Parameters:
        model_path (str): The path to the model file.
        
    Returns:
        The loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def compute_classification_metrics(y_true, y_pred) -> dict:
    """
    Compute classification metrics: precision, recall, F1 score, and classification report.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        
    Returns:
        dict: A dictionary containing precision, recall, F1 score, and classification report.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report
    }


def create_directory(path: str):
    """
    Create a directory if it does not exist.
    
    Parameters:
        path (str): The directory path.
    """
    os.makedirs(path, exist_ok=True)


def plot_and_save(figure, output_path: str, show: bool = True):
    """
    Save a matplotlib figure to a file and optionally display it.
    
    Parameters:
        figure: The matplotlib figure object.
        output_path (str): The file path to save the figure.
        show (bool): Whether to display the plot.
    """
    create_directory(os.path.dirname(output_path))
    figure.savefig(output_path)
    if show:
        figure.show()
    plt.close(figure)


# Example usage for testing this module
if __name__ == "__main__":
    # Test loading dataset (modify path accordingly)
    try:
        df = load_dataset("../data/processed/sp500_cleaned_model_ready.csv")
        print("Dataset loaded successfully, shape:", df.shape)
    except FileNotFoundError as e:
        print(e)
    
    # Test feature columns extraction
    features = get_feature_columns(df)
    print("Feature columns:", features[:5])
    
    # Example: Compute metrics on dummy data
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    print("Dummy classification metrics:")
    print(metrics)
    
    # Test creating a directory
    create_directory("../reports/images/test")
    print("Test directory created.")
