import streamlit as st
import httpx
import pandas as pd
import datetime

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

st.title("📈 Demand Forecasting & Time Series Analysis")
st.markdown("Interact with the real-time forecasting API powered by LightGBM.")

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.sidebar.header("Input Parameters")

# Mock inputs for the ML model
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.number_input("Temperature (°C)", -10.0, 40.0, 20.0)
is_holiday = st.sidebar.selectbox("Is Holiday?", [0, 1])
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
month = st.sidebar.slider("Month", 1, 12, 6)
quarter = (month - 1) // 3 + 1
is_weekend = 1 if day_of_week >= 5 else 0

st.sidebar.markdown("### Historical Demand Features")
lag_7 = st.sidebar.number_input("Demand 7 Days Ago", 0, 500, 100)
lag_14 = st.sidebar.number_input("Demand 14 Days Ago", 0, 500, 110)
lag_30 = st.sidebar.number_input("Demand 30 Days Ago", 0, 500, 105)

st.sidebar.markdown("### Rolling Statistics")
rolling_mean_7 = st.sidebar.number_input("7-Day Rolling Mean", 0.0, 500.0, 105.0)
rolling_std_7 = st.sidebar.number_input("7-Day Rolling Std Dev", 0.0, 100.0, 15.0)
rolling_mean_30 = st.sidebar.number_input("30-Day Rolling Mean", 0.0, 500.0, 102.0)

if st.button("Generate Forecast"):
    payload = {
        "hour": hour,
        "temp": temp,
        "is_holiday": is_holiday,
        "day_of_week": day_of_week,
        "month": month,
        "quarter": quarter,
        "is_weekend": is_weekend,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_30": lag_30,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7,
        "rolling_mean_30": rolling_mean_30
    }

    try:
        with st.spinner("Calling API..."):
            response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)

        if response.status_code == 200:
            data = response.json()
            pred = data['predicted_demand']

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Demand", f"{pred:.2f} units")
            col2.metric("Timestamp", data['timestamp'].split('T')[1][:8])
            col3.metric("Model", "LightGBM")

            st.success("Forecast generated successfully!")

        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to backend API: {e}. Ensure FastAPI is running on {API_URL}.")
