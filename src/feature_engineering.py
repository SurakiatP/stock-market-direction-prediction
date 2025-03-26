# feature_engineering.py

import pandas as pd
import numpy as np
import os
from datetime import datetime
import ta

def engineer_features(
    df: pd.DataFrame,
    start_date: datetime = datetime(1990, 1, 1),
) -> pd.DataFrame:
    """
    Main function to create target column, rolling averages, technical indicators, 
    and strategy-driven features. Returns a DataFrame with all features.

    :param df: Raw DataFrame containing at least 'Close', 'Volume', etc.
    :param start_date: Earliest date to keep in the data
    :return: DataFrame with engineered features
    """

    # 1) Filter by start date
    df = df[df.index.notnull()]
    df = df[df.index >= start_date].copy()

    # 2) Create target column
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    # 3) Rolling averages and trends
    horizons = [2, 5, 20, 60, 120, 250, 1000]
    for h in horizons:
        df[f"Close_Ratio_{h}"] = df["Close"] / df["Close"].rolling(h).mean()
        df[f"Trend_{h}"] = df["Target"].shift(1).rolling(h).sum()

    # 4) Technical indicators (using ta)
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_diff"] = df["MACD"] - df["MACD_Signal"]

    df["EMA_5"] = ta.trend.EMAIndicator(close=df["Close"], window=5).ema_indicator()
    df["EMA_20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()

    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    df["Momentum_10"] = ta.momentum.ROCIndicator(close=df["Close"], window=10).roc()
    df["ATR_14"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"]
    ).on_balance_volume()

    # 5) Lag features
    df["Lag_Close_1"] = df["Close"].shift(1)
    df["Lag_Close_2"] = df["Close"].shift(2)
    df["Lag_Volume_1"] = df["Volume"].shift(1)

    # 6) Strategy-driven features
    # Crossover: EMA 5 > EMA 20
    df["EMA_Cross"] = (df["EMA_5"] > df["EMA_20"]).astype(int)

    # Price distance from EMA 20
    df["Close_EMA20_Pct"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"]

    # Bollinger Band width & volatility %
    df["Bollinger_Width"] = df["BB_High"] - df["BB_Low"]
    df["Volatility_Pct"] = df["Bollinger_Width"] / df["Close"]

    # Momentum acceleration
    df["Momentum_Change"] = df["Momentum_10"].diff()

    # Relative volume
    df["Volume_Ratio_5"] = df["Volume"] / df["Volume"].rolling(5).mean()

    # Volume Z-Score
    df["Volume_ZScore"] = (
        df["Volume"] - df["Volume"].rolling(20).mean()
    ) / df["Volume"].rolling(20).std()

    # Price up streak (past 5 days)
    df["Price_Up_Streak"] = df["Target"].rolling(5).sum()

    # MACD crossover
    df["MACD_Cross"] = (df["MACD"] > df["MACD_Signal"]).astype(int)

    # Strong momentum
    df["Momentum_StrongUp"] = (df["Momentum_10"] > 5).astype(int)
    df["Momentum_StrongDown"] = (df["Momentum_10"] < -5).astype(int)

    # Drop missing data from rolling windows
    df.dropna(inplace=True)

    return df


def save_engineered_data(
    df: pd.DataFrame,
    output_full="../data/processed/sp500_cleaned.csv",
    output_model="../data/processed/sp500_cleaned_model_ready.csv",
):
    """
    Save full version with all features, 
    and model-ready version (exclude 'Tomorrow'/'Target' as needed).
    """
    # Save full version
    os.makedirs(os.path.dirname(output_full), exist_ok=True)
    df.to_csv(output_full)

    # Save model-ready version
    features_for_modeling = [c for c in df.columns if c not in ["Tomorrow", "Target"]]
    df_model = df[features_for_modeling + ["Target"]]
    df_model.to_csv(output_model)
    print(f"Saved model-ready dataset: shape = {df_model.shape}")


def run_feature_engineering_pipeline(
    raw_path="../data/raw/sp500_raw.csv",
    start_date=datetime(1990, 1, 1),
    output_full="../data/processed/sp500_cleaned.csv",
    output_model="../data/processed/sp500_cleaned_model_ready.csv",
):
    """
    High-level pipeline function that:
    1) Loads raw data from CSV
    2) Creates all features
    3) Saves full & model-ready CSV

    :return: The final DataFrame (with all features)
    """
    df = pd.read_csv(raw_path, index_col=0)
    # Convert index to datetime
    df.index = pd.to_datetime(df.index, errors="coerce").map(
        lambda x: x.tz_convert(None) if x and x.tzinfo else x
    )
    df.dropna(subset=[df.index.name], inplace=True)

    # Engineer features
    df = engineer_features(df, start_date=start_date)

    # Save results
    save_engineered_data(df, output_full, output_model)

    return df
