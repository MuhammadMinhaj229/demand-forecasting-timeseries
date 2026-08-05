# Demand Forecasting & Time Series Analysis

This repository contains a demand forecasting pipeline built on a time-series dataset. It demonstrates how to explore the data, engineer features, and train multiple time-series and machine learning models to forecast future demand.

## Problem / Motivation

The goal of this project is to explore the predictability of the bike sharing demand time series by comparing statistical approaches (ARIMA/SARIMA) with an additive model (Prophet) and a gradient boosting approach (LightGBM).

## Approach

1. **Exploratory Data Analysis (EDA):** `notebooks/01_eda_timeseries.ipynb` loads the data and performs basic exploratory analysis.
2. **Statistical Baselines (ARIMA/SARIMA):** `notebooks/02_arima_sarima.ipynb` resamples the data to a daily frequency and fits `pmdarima` auto-ARIMA models to establish a baseline.
3. **Prophet Additive Model:** `notebooks/03_prophet_model.ipynb` fits a Prophet model that incorporates yearly, weekly, and daily seasonality, along with public holidays.
4. **LightGBM:** `notebooks/04_lightgbm_model.ipynb` applies a gradient boosting regressor on engineered features including lags, rolling statistics, and calendar attributes.
5. **Experiment Tracking:** The training script `src/train.py` uses `MLflow` to log LightGBM parameters and evaluation metrics.
6. **API and Dashboard:** The repository includes a FastAPI endpoint (`src/api.py`) for serving predictions and a Streamlit dashboard (`app/streamlit_app.py`) for visualization.

## Results

Models were evaluated on a temporal train-test split (80/20). For LightGBM, a 5-fold TimeSeriesSplit was used to cross-validate the performance. The following metrics were recorded (see outputs in respective notebooks and `src/train.py`):

| Model | MAPE | RMSE | MAE | Source |
|-------|------|------|-----|--------|
| ARIMA (Daily) | ~1.17% | 49.14 | 40.28 | `notebooks/02_arima_sarima.ipynb` |
| SARIMA (Daily, m=7) | ~1.10% | 46.36 | 37.86 | `notebooks/02_arima_sarima.ipynb` |
| Prophet (Hourly) | ~9.02% | 14.94 | 11.98 | `notebooks/03_prophet_model.ipynb` |
| LightGBM (Hourly) | ~8.53% | 12.60 | 10.11 | `src/train.py` |

*(Note: ARIMA/SARIMA were evaluated on daily aggregated demand for feasibility, while Prophet and LightGBM were evaluated on the original hourly frequency. MAPE percentages for ARIMA/SARIMA correspond to 0.0117 and 0.0110 in the raw output).*

## Tech Stack

- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `lightgbm`, `prophet`, `pmdarima`
- **Experiment Tracking:** `mlflow`
- **API:** `fastapi`, `uvicorn`, `pydantic`
- **Dashboard:** `streamlit`, `plotly`

## Project Structure

```
demand-forecasting-timeseries/
├── app/
│   └── streamlit_app.py           # Streamlit dashboard
├── data/
│   └── bike_sharing.csv           # Raw dataset
├── models/
│   └── lgbm_forecast.pkl          # Serialized LightGBM model
├── notebooks/
│   ├── 01_eda_timeseries.ipynb
│   ├── 02_arima_sarima.ipynb
│   ├── 03_prophet_model.ipynb
│   └── 04_lightgbm_model.ipynb
├── src/
│   ├── api.py                     # FastAPI application
│   ├── feature_engineering.py     # Feature engineering script
│   └── train.py                   # MLflow training script
├── tests/
│   └── test_api.py                # Pytest cases for the API
├── README.md
└── requirements.txt
```

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You may also need to install `jupyter`, `pmdarima`, `prophet`, and `pytest` depending on your environment.*

3. **Train the model and track with MLflow:**
   ```bash
   python src/train.py
   ```
   You can view the MLflow UI by running `mlflow ui` (it logs to a local `mlruns.db`).

4. **Start the API:**
   ```bash
   uvicorn src.api:app --reload
   ```

5. **Start the Dashboard:**
   In a separate terminal, run:
   ```bash
   streamlit run app/streamlit_app.py
   ```

6. **Run Tests:**
   ```bash
   python -m pytest tests/
   ```

## Limitations & Future Work

- **Dataset:** The provided data is static and represents a point-in-time snapshot of demand. Real-world applications require a pipeline to handle streaming or continually appended data.
- **Comparison Baseline:** The statistical models (ARIMA/SARIMA) were applied to daily aggregated data, while Prophet and LightGBM were trained on hourly data. A direct comparison requires harmonizing the time scale.
- **Model Deployment:** The current setup loads a local `.pkl` file into a local FastAPI instance. In a production scenario, the model would likely be served from a model registry or robust cloud storage.
- **Dashboard Integration:** The Streamlit dashboard currently loads the local model directly rather than querying the FastAPI endpoint, meaning the API is largely decoupled from the frontend in this configuration.

---

### UNVERIFIED CLAIMS — NOT INCLUDED

The previous version of this README contained several marketing claims and tables that could not be verified by the code in this repository. These have been removed:

- Any ROI, cost savings, or revenue numbers (e.g., "₹40–80 Lakhs annually", "₹2,000 Crores annually", "18–25% reduction").
- Business impact tables claiming specific reductions in stockout incidents or planning cycle time.
- Market sizing or applicability tables ("Industries Where This Applies").
- Subjective marketing adjectives like "production-grade", "enterprise", or "cutting-edge".
- Any mention of the FastAPI serving real-time predictions to the dashboard (the dashboard currently loads the model from disk directly).

## Author / Links

*Built by Mohammed Minhaj Mahmood*
*[LinkedIn](https://linkedin.com/in/muhammadminhaj229) · [GitHub](https://github.com/MuhammadMinhaj229) · Hyderabad, India*
