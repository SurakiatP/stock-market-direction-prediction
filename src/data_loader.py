"""
data_loader.py

This module is responsible for loading datasets from CSV files.
It provides functions to load raw data, processed data, and model-ready data.
"""

import os
import pandas as pd

def load_raw_data(path: str = "../data/raw/sp500_raw.csv", index_col=0, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    
    Parameters:
        path (str): Path to the raw data CSV file.
        index_col (int or str): Column to be used as the index.
        parse_dates (bool): Whether to parse dates.
        
    Returns:
        pd.DataFrame: The loaded raw data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")
    df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    return df

def load_processed_data(path: str = "../data/processed/sp500_cleaned.csv", index_col=0, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load processed data (full version with all features) from a CSV file.
    
    Parameters:
        path (str): Path to the processed data CSV file.
        index_col (int or str): Column to be used as the index.
        parse_dates (bool): Whether to parse dates.
        
    Returns:
        pd.DataFrame: The loaded processed data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found: {path}")
    df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    return df

def load_model_ready_data(path: str = "../data/processed/sp500_cleaned_model_ready.csv", index_col=0, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load the model-ready data from a CSV file.
    This dataset is prepared for modeling (excluding leakage columns such as 'Tomorrow').
    
    Parameters:
        path (str): Path to the model-ready data CSV file.
        index_col (int or str): Column to be used as the index.
        parse_dates (bool): Whether to parse dates.
        
    Returns:
        pd.DataFrame: The loaded model-ready data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model-ready data file not found: {path}")
    df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    return df


if __name__ == "__main__":
    # Example usage for testing the functions
    try:
        raw_df = load_raw_data()
        print("Raw data loaded successfully. Shape:", raw_df.shape)
    except Exception as e:
        print(e)
    
    try:
        processed_df = load_processed_data()
        print("Processed data loaded successfully. Shape:", processed_df.shape)
    except Exception as e:
        print(e)
    
    try:
        model_ready_df = load_model_ready_data()
        print("Model-ready data loaded successfully. Shape:", model_ready_df.shape)
    except Exception as e:
        print(e)
