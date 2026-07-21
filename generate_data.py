import pandas as pd
import numpy as np

# Generate synthetic data for 1 year, hourly
dates = pd.date_range(start="2023-01-01", end="2023-12-31 23:00:00", freq="h")
df = pd.DataFrame({'date': dates})
df['hour'] = df['date'].dt.hour

# Base demand with some noise, daily seasonality (higher in day), and yearly trend
np.random.seed(42)
base_demand = 100
daily_seasonality = np.sin((df['hour'] - 6) * np.pi / 12) * 50
trend = np.linspace(0, 50, len(df))
noise = np.random.normal(0, 10, len(df))

df['demand'] = np.clip(base_demand + daily_seasonality + trend + noise, 0, None).astype(int)
df['temp'] = np.clip(20 + np.sin((df['hour'] - 8) * np.pi / 12) * 10 + np.random.normal(0, 2, len(df)), -10, 40)
df['is_holiday'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])

df.to_csv('data/bike_sharing.csv', index=False)
