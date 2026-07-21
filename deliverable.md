# End-to-End Implementation Plan & Analysis Summary: Demand Forecasting System

## 1. Business Analysis Summary

### Stakeholder Needs

- **Procurement & Operations:** Need accurate, real-time predictions of future demand to prevent overstocking and understocking.
- **Finance:** Require cost reduction through minimized inventory holding costs (targeting 18-25% reduction) and reduced lost sales from stockouts.
- **Data Science/IT:** Need a reproducible, automated ML pipeline capable of comparing multiple models and serving the best one.

### Success Metrics (KPIs)

- **Forecast Accuracy (MAPE):** Target < 10% (Currently achieved 6.8% with LightGBM).
- **Inventory Costs:** 18-25% reduction in inventory holding costs.
- **Stockout Incidents:** > 40% reduction in stockout occurrences.
- **Overstock Incidents:** > 30% reduction in overstock occurrences.
- **System Latency:** API response time for real-time forecast serving < 200ms.

### Assumptions, Dependencies & Blockers

- **Assumptions:** Historical data provided (`bike_sharing.csv`) is representative of current market conditions. The chosen ML metrics (MAPE) align perfectly with business cost reduction goals.
- **Dependencies:** The pipeline relies on external weather/macro data sources if those features are included in LightGBM.
- **Blockers for Clarification:** We need clarification on the exact definition of "Overstock Incidents" (e.g., threshold of inventory holding days) from the Finance team before finalizing the monitoring dashboard.

### Risk Mitigation

- **Data Quality:** Poor historical data can lead to inaccurate forecasts. *Mitigation:* Implement strict data validation and automated data cleaning in the ETL pipeline.
- **Model Drift:** The forecasting model may degrade over time as consumer behavior changes. *Mitigation:* Implement continuous monitoring and automated retraining pipelines when performance (MAPE) drops below the 10% threshold.
- **System Downtime:** API unavailability affecting operations. *Mitigation:* Multi-instance deployment with auto-scaling and fallback mechanisms.

---

## 2. Data Analysis Plan

### Data Sources & Requirements

- **Primary Data:** Historical daily and hourly sales/demand data (e.g., `bike_sharing.csv`).
- **External Data (Optional/Future):** Weather data, macroeconomic indicators, marketing spend.

### Transformation & ETL Steps

1. **Data Cleaning:** Handle missing values, outliers, and correct data types (especially datetime parsing).
2. **Feature Engineering (`src/feature_engineering.py`):**
   - **Lag Features:** Demand from past 7, 14, 30 days.
   - **Rolling Stats:** 7-day and 30-day rolling means and standard deviations.
   - **Temporal/Calendar Features:** Hour of day, day of week, month, quarter, is_weekend.
   - **Event Features:** Holidays (using Prophet's holiday effects or boolean flags for LightGBM).

### Analytics Methods & Business Tie-in

- **Exploratory Data Analysis (EDA):** Identify stationarity, trend, and seasonality (ACF/PACF) to inform model selection.
- **Time Series Decomposition:** Understand underlying patterns (e.g., 17% YoY growth, weekday vs. weekend demand).
- **Model Evaluation (TimeSeriesSplit):** Use proper temporal cross-validation (RMSE, MAE, MAPE) to ensure the chosen model genuinely reduces forecasting error, directly driving down inventory costs.

---

## 3. Implementation Plan & System Architecture

### Technology Stack

- **Data & ML:** Python 3.10, Pandas, NumPy, scikit-learn, LightGBM, XGBoost, Prophet, statsmodels.
- **Experiment Tracking:** MLflow.
- **Backend/API:** FastAPI, Uvicorn.
- **Frontend/Dashboard:** Streamlit, Plotly, Matplotlib.

### System Architecture & Component Responsibilities

- **Data Pipeline:** Scheduled cron jobs/Airflow to extract data, apply `src/feature_engineering.py`, and prepare for inference.
- **Model Training & Tracking (`src/model_utils.py`):** MLflow server records experiments, hyperparameters, and metrics. The best model (`models/lgbm_forecast.pkl`) is registered.
- **Backend API (`src/api.py`):** FastAPI application loads the registered model and exposes a REST endpoint (`/predict`) for real-time forecasting.
- **Frontend Dashboard (`app/streamlit_app.py`):** Streamlit application consumes the FastAPI endpoint to visualize historical data and future forecasts for business stakeholders.

### Integration Points

- Dashboard -> API: REST over HTTP.
- API -> Model Artifact: Local file load or MLflow Model Registry fetch.
- Training Pipeline -> MLflow: MLflow tracking URI.

---

## 4. Code Organization Guidelines

To ensure maintainability and scalability, follow this structure:

```text
demand-forecasting-timeseries/
├── data/                      # Raw and processed datasets (ignored in git)
├── notebooks/                 # Exploratory data analysis & model prototyping (01 to 04)
├── src/                       # Production source code
│   ├── __init__.py
│   ├── feature_engineering.py # ETL and feature creation logic
│   ├── model_utils.py         # Training, evaluation, and MLflow logging functions
│   └── api.py                 # FastAPI application and routing
├── app/
│   └── streamlit_app.py       # Streamlit UI application
├── models/                    # Serialized model artifacts (e.g., lgbm_forecast.pkl)
├── tests/                     # Unit and integration tests
│   ├── test_api.py
│   └── test_features.py
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Containerization instructions
├── .gitignore
└── README.md
```

- **Guidelines:** Keep notebooks for prototyping only. All production logic must reside in `src/`. Ensure modularity (separate data processing from model training).

---

## 5. Testing & Deployment Strategy

### Testing Strategy

- **Unit Testing (pytest):** Validate individual functions in `src/feature_engineering.py` (e.g., correct lag calculation).
- **Integration Testing:** Ensure the FastAPI endpoint returns expected payload formats and handles edge cases (e.g., missing data fields).
- **Validation (Temporal):** Strictly use `TimeSeriesSplit` for model evaluation to prevent data leakage.
- **Load Testing:** Use Locust or Artillery to ensure the API handles concurrent requests within latency requirements.

### Deployment Plan

- **Environment Setup:** Containerize the API and Dashboard using Docker.
- **CI/CD Pipeline (GitHub Actions / GitLab CI):**
  - **CI:** On pull request -> Run linting (flake8/black), unit tests (pytest), and build Docker image.
  - **CD:** On merge to `main` -> Deploy FastAPI to a scalable backend service (e.g., Render Web Service, AWS ECS) and Streamlit to a frontend service (Render, Streamlit Community Cloud).
- **Monitoring:** Monitor API health/latency (Datadog/New Relic) and Model Drift (monitor MAPE on newly incoming actuals).
- **Rollback Strategy:** Version model artifacts via MLflow and Docker images via a container registry. If a new deployment degrades performance, immediately revert to the previous container image tag.

---

## 6. Step-by-Step Development Roadmap

**Phase 1: Business & Data Analysis (Weeks 1-2)** *(Roles: Business Analyst, Data Analyst)*

- Milestone 1: Finalize KPI definitions and risk mitigation plans with stakeholders.
- Milestone 2: Complete EDA (`01_eda_timeseries.ipynb`) and finalize feature engineering requirements.

**Phase 2: Model Development & Experimentation (Weeks 3-4)** *(Roles: Data Scientist)*

- Milestone 3: Implement baseline statistical models (`02_arima_sarima.ipynb`).
- Milestone 4: Implement advanced models (Prophet, LightGBM) and track experiments via MLflow (`03_prophet_model.ipynb`, `04_lightgbm_model.ipynb`). Select the best performing model.

**Phase 3: Productionization & Backend Development (Week 5)** *(Roles: ML Engineer, Backend Developer)*

- Milestone 5: Refactor notebook code into `src/feature_engineering.py` and `src/model_utils.py`.
- Milestone 6: Develop and unit-test the FastAPI endpoint (`src/api.py`).

**Phase 4: Frontend Development & Integration (Week 6)** *(Roles: Frontend Developer)*

- Milestone 7: Build the Streamlit dashboard (`app/streamlit_app.py`).
- Milestone 8: Integrate the dashboard with the FastAPI backend and perform end-to-end testing.

**Phase 5: Deployment & Handoff (Week 7)** *(Roles: DevOps, Project Manager)*

- Milestone 9: Dockerize applications and configure the CI/CD pipeline.
- Milestone 10: Deploy to production (e.g., Render), set up monitoring, and hand off to stakeholders.
