"""
model.py

This module provides functions to train, tune, save, load, predict, and evaluate machine learning models
for the stock market prediction project.

Functions:
    - train_models: Train and tune multiple models using GridSearchCV.
    - save_model: Save a trained model to a file.
    - load_model: Load a model from a file.
    - predict_with_threshold: Predict probabilities and apply a threshold to generate binary predictions.
    - evaluate_predictions: Evaluate model predictions using precision and classification report.
    - walk_forward_backtest: Perform a walk-forward backtest for time series data.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, classification_report, confusion_matrix

def train_models(X_train, y_train, model_configs, scoring="precision", cv=5, n_jobs=-1):
    """
    Train multiple models using GridSearchCV based on the provided model configurations.
    
    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        model_configs (dict): Dictionary containing model instances and hyperparameter grids.
        scoring (str): Scoring metric for GridSearchCV.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of parallel jobs.
        
    Returns:
        best_models (dict): Dictionary mapping model names to their best estimator.
        results_df (DataFrame): Summary DataFrame containing each model's best parameters and CV score.
    """
    results = []
    best_models = {}
    
    for name, config in model_configs.items():
        print(f"\nTraining {name} ...")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )
        grid.fit(X_train, y_train)
        
        best_estimator = grid.best_estimator_
        best_models[name] = best_estimator
        
        # Use cross-validation best score as a metric
        results.append({
            "Model": name,
            "Best Params": grid.best_params_,
            "Best CV Score": grid.best_score_
        })
        
        print(f"   Best Params: {grid.best_params_}")
        print(f"   Best CV Score: {grid.best_score_:.4f}")
    
    results_df = pd.DataFrame(results).sort_values(by="Best CV Score", ascending=False)
    return best_models, results_df

def save_model(model, output_path):
    """
    Save the trained model to a pickle file.
    
    Parameters:
        model: Trained model.
        output_path (str): File path to save the model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f" Model saved to {output_path}")

def load_model(model_path):
    """
    Load a model from a pickle file.
    
    Parameters:
        model_path (str): File path to the model.
        
    Returns:
        Loaded model.
    """
    return joblib.load(model_path)

def predict_with_threshold(model, X, threshold=0.5):
    """
    Predict probabilities for class 1 using the model and convert them to binary predictions
    based on the given threshold.
    
    Parameters:
        model: Trained model.
        X (DataFrame): Input features.
        threshold (float): Threshold to convert probabilities into binary predictions.
    
    Returns:
        probs (ndarray): Predicted probabilities for class 1.
        preds (ndarray): Binary predictions.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)
    return probs, preds

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using precision score, classification report, and confusion matrix.
    
    Parameters:
        y_true (Series or array-like): True target values.
        y_pred (Series or array-like): Predicted target values.
    
    Returns:
        metrics (dict): Dictionary containing precision, classification report, and confusion matrix.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"precision": precision, "classification_report": report, "confusion_matrix": cm}

def walk_forward_backtest(data, model, features, target="Target", start=2500, step=250, threshold=0.5, model_path=None):
    """
    Perform a walk-forward backtest on time series data.
    
    For each iteration, the model is retrained on all data up to the current index,
    then predictions are made on the next 'step' rows.
    
    Parameters:
        data (DataFrame): The complete dataset with features and target.
        model: A trained model instance (used if model_path is not provided).
        features (list): List of feature column names.
        target (str): Name of the target column.
        start (int): Number of initial rows to use as training data.
        step (int): Number of rows to move forward at each iteration.
        threshold (float): Threshold for binary prediction.
        model_path (str): If provided, a new model will be loaded from this path in each iteration.
    
    Returns:
        backtest_df (DataFrame): DataFrame containing actual and predicted values along with probabilities.
    """
    all_predictions = []
    
    for i in range(start, len(data), step):
        train = data.iloc[:i]
        test = data.iloc[i:i+step]
        
        # Use a fresh copy of the model if model_path is provided
        if model_path:
            current_model = load_model(model_path)
        else:
            current_model = model
        
        # Retrain the model on training data
        current_model.fit(train[features], train[target])
        
        # Predict probabilities and apply threshold
        probs = current_model.predict_proba(test[features])[:, 1]
        preds = (probs > threshold).astype(int)
        
        temp_df = test.copy()
        temp_df["Prediction"] = preds
        temp_df["Prob_Up"] = probs
        all_predictions.append(temp_df)
    
    return pd.concat(all_predictions)

# Example usage when running this module directly
if __name__ == "__main__":
    # For demonstration purposes, use a small sample dataset (Iris example) 
    # This block is for testing only; in practice, use your stock market data.
    from sklearn.datasets import load_iris
    data = load_iris()
    X_sample = pd.DataFrame(data.data, columns=data.feature_names)
    y_sample = (data.target == 2).astype(int)
    
    # Define a simple model configuration for testing
    from sklearn.ensemble import RandomForestClassifier
    model_configs = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "params": {
                "n_estimators": [50, 100],
                "min_samples_split": [2, 10]
            }
        }
    }
    
    best_models, results_df = train_models(X_sample, y_sample, model_configs)
    print("Test Results:")
    print(results_df)
