import pandas as pd
import numpy as np

def create_features(df):
    """
    Apply feature engineering to the raw dataset based on feature_requirements.md
    """
    df = df.copy()

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 1. Calendar Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 2. Lag Features
    df['lag_7'] = df['demand'].shift(7 * 24)
    df['lag_14'] = df['demand'].shift(14 * 24)
    df['lag_30'] = df['demand'].shift(30 * 24)

    # 3. Rolling Statistics
    df['rolling_mean_7'] = df['demand'].shift(1).rolling(window=7*24, min_periods=1).mean()
    df['rolling_std_7'] = df['demand'].shift(1).rolling(window=7*24, min_periods=1).std()
    df['rolling_mean_30'] = df['demand'].shift(1).rolling(window=30*24, min_periods=1).mean()

    # Handle NaNs from shifting/rolling
    df = df.bfill()

    return df
