"""
run_all.py

This script runs the entire machine learning pipeline:
1. Feature engineering (data preprocessing and feature creation)
2. Model training with hyperparameter tuning
3. Walk-forward backtesting and evaluation

Run this script from the project root:
    python run_all.py
"""

import os

def main():
    print("========== Running Feature Engineering Pipeline ==========")
    # Run feature engineering pipeline from src/feature_engineering.py
    # This will load raw data, create features, and export processed CSV files.
    os.system("python src/feature_engineering.py")
    
    print("\n========== Training Models ==========")
    # Run model training from src/train_model.py, which performs hyperparameter tuning and saves the best model.
    os.system("python src/train_model.py")
    
    print("\n========== Performing Walk-forward Backtesting ==========")
    # Run backtesting evaluation using the functions in src/backtest.py.
    # This example assumes the best model is saved as best_model_RandomForest.pkl.
    # You can adjust the model path as needed.
    os.system("python src/run_backtest.py")
    
    print("\n========== Build File For Report This Project ==========")

    os.system("python generate_eda_summary.py")
    os.system("python generate_feature_insights.py")
    os.system("python generate_model_performance.py")

    print("\n========== Pipeline Execution Complete ==========")


if __name__ == "__main__":
    main()
