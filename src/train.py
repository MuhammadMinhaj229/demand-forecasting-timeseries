import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, mean_absolute_error
import os
import mlflow

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

    print("Evaluating using TimeSeriesSplit (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)

    mapes = []
    rmses = []
    maes = []

    # We will log hyperparameters and metrics to mlflow
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("bike_sharing_demand")

    with mlflow.start_run():
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 7,
            'random_state': 42,
            'verbose': -1
        }
        mlflow.log_params(params)

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mapes.append(mean_absolute_percentage_error(y_test, preds))
            rmses.append(root_mean_squared_error(y_test, preds))
            maes.append(mean_absolute_error(y_test, preds))

        final_mape = sum(mapes)/len(mapes)
        final_rmse = sum(rmses)/len(rmses)
        final_mae = sum(maes)/len(maes)

        mlflow.log_metric("cv_mape", final_mape)
        mlflow.log_metric("cv_rmse", final_rmse)
        mlflow.log_metric("cv_mae", final_mae)

        print(f"Model Evaluation -> MAPE: {final_mape:.4f}, RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")

        # Train on full dataset before saving
        model.fit(X, y)

        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/lgbm_forecast.pkl')
        print("Model saved to models/lgbm_forecast.pkl")

if __name__ == "__main__":
    train_model()
