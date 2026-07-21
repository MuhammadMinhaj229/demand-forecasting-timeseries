# Feature Engineering Requirements

Based on the initial EDA and business goals, the following features are required for the forecasting models (especially LightGBM):

## 1. Lag Features
Captures historical demand values.
- `lag_7`: Demand 7 days ago.
- `lag_14`: Demand 14 days ago.
- `lag_30`: Demand 30 days ago.

## 2. Rolling Statistics
Smooths out noise and captures short-term trends.
- `rolling_mean_7`: 7-day rolling mean of demand.
- `rolling_std_7`: 7-day rolling standard deviation of demand.
- `rolling_mean_30`: 30-day rolling mean of demand.

## 3. Calendar & Temporal Features
Extracts seasonality patterns.
- `hour`: Hour of the day (0-23).
- `day_of_week`: Day of the week (0-6).
- `month`: Month of the year (1-12).
- `quarter`: Quarter of the year (1-4).
- `is_weekend`: Boolean flag (1 if Saturday/Sunday, 0 otherwise).

## 4. Event Features
Captures anomalies and external events.
- `is_holiday`: Boolean flag (1 if holiday, 0 otherwise).
