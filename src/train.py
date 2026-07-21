import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import os

from feature_engineering import create_features

def train_model():
    print("Loading and preparing data...")
    df = pd.read_csv('data/bike_sharing.csv')
    df = create_features(df)

    # Define features and target
    features = [
        'hour', 'temp', 'is_holiday', 'day_of_week', 'month', 'quarter',
        'is_weekend', 'lag_7', 'lag_14', 'lag_30',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30'
    ]
    target = 'demand'

    X = df[features]
    y = df[target]

    # Train test split (temporal)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)

    print(f"Model Evaluation -> MAPE: {mape:.4f}, RMSE: {rmse:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/lgbm_forecast.pkl')
    print("Model saved to models/lgbm_forecast.pkl")

if __name__ == "__main__":
    train_model()
