import streamlit as st
import httpx
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Demand Forecasting HQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Header Section
st.title("📈 Demand Forecasting Intelligence")
st.markdown("Predictive analytics powered by LightGBM to optimize inventory, reduce stockouts, and streamline operations.")
st.divider()

# Layout: Sidebar for inputs, Main for Visualization
with st.sidebar:
    st.header("⚙️ Forecast Parameters")

    st.subheader("Time & Calendar")
    hour = st.slider("Hour of Day", 0, 23, 12, help="Current time of day")
    day_of_week = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    month = st.slider("Month", 1, 12, 6)

    st.subheader("Environmental & Events")
    temp = st.number_input("Temperature (°C)", -10.0, 40.0, 25.0)
    is_holiday = st.checkbox("Is Public Holiday?", value=False)

    st.subheader("Historical Context (Lags)")
    lag_7 = st.number_input("Demand (T-7 days)", 0.0, 1000.0, 120.0)
    lag_14 = st.number_input("Demand (T-14 days)", 0.0, 1000.0, 115.0)
    lag_30 = st.number_input("Demand (T-30 days)", 0.0, 1000.0, 110.0)

    st.subheader("Smoothing (Rolling Stats)")
    rolling_mean_7 = st.number_input("7-Day Avg Demand", 0.0, 1000.0, 118.0)
    rolling_std_7 = st.number_input("7-Day Volatility (StdDev)", 0.0, 100.0, 12.0)
    rolling_mean_30 = st.number_input("30-Day Avg Demand", 0.0, 1000.0, 105.0)

    quarter = (month - 1) // 3 + 1
    is_weekend = 1 if day_of_week >= 5 else 0
    holiday_val = 1 if is_holiday else 0

    submit_btn = st.button("🚀 Generate Real-Time Forecast")

# Main content tabs
tab1, tab2 = st.tabs(["📊 Real-Time Forecast", "🧠 Model Insights"])

with tab1:
    if submit_btn:
        payload = {
            "hour": hour,
            "temp": temp,
            "is_holiday": holiday_val,
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
            with st.spinner("Analyzing parameters and inferencing model..."):
                response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)

            if response.status_code == 200:
                data = response.json()
                pred = data['predicted_demand']

                # KPI Metrics Row
                cols = st.columns(3)

                with cols[0]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>🔮 Predicted Demand</h3>
                            <h1 style="color: #4CAF50;">{pred:.0f}</h1>
                            <p>Units needed</p>
                        </div>
                    """, unsafe_allow_html=True)

                with cols[1]:
                    # Mocking previous period for comparison metric
                    delta = pred - rolling_mean_7
                    color = "green" if delta > 0 else "red"
                    arrow = "↑" if delta > 0 else "↓"
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>⚖️ vs 7-Day Avg</h3>
                            <h1 style="color: {color};">{arrow} {abs(delta):.0f}</h1>
                            <p>Variance</p>
                        </div>
                    """, unsafe_allow_html=True)

                with cols[2]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>⏱️ API Latency</h3>
                            <h1 style="color: #2196F3;">{response.elapsed.total_seconds() * 1000:.0f} ms</h1>
                            <p>Response Time</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.write("") # Spacer

                # Visualization: Current state vs historical context
                st.subheader("📈 Demand Context Visualization")

                chart_data = pd.DataFrame({
                    "Period": ["30 Days Ago", "14 Days Ago", "7 Days Ago", "Predicted (Now)"],
                    "Demand": [lag_30, lag_14, lag_7, pred]
                })

                fig = px.line(chart_data, x="Period", y="Demand", markers=True,
                              title="Historical Demand Trajectory vs Prediction")
                fig.update_traces(line=dict(color="#FF5722", width=3), marker=dict(size=10))

                # Highlight prediction
                fig.add_scatter(x=["Predicted (Now)"], y=[pred], mode="markers",
                                marker=dict(color="#4CAF50", size=14, symbol="star"),
                                name="Forecast")

                fig.update_layout(plot_bgcolor='white', xaxis_title="", yaxis_title="Units")
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"⚠️ API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Failed to connect to backend API: {e}. Ensure FastAPI is running on {API_URL}.")
    else:
        st.info("👈 Adjust parameters in the sidebar and click **Generate Real-Time Forecast** to begin.")

        # Display an empty placeholder state chart
        fig = go.Figure()
        fig.add_annotation(text="No Forecast Generated Yet", x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="gray"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="#f8f9fa", height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Technical Details")
    st.markdown("""
    - **Model Architecture:** LightGBM Regressor (Gradient Boosting)
    - **Optimization Objective:** RMSE
    - **Target Metric:** MAPE < 10%
    - **Feature Importance (Top 3):**
      1. `lag_7` (Highest correlation to current demand)
      2. `rolling_mean_30` (Captures macro seasonal trend)
      3. `hour` (Captures intra-day cyclic behavior)
    """)
    st.info("Continuous integration and MLflow tracking ensure this model automatically retraining when MAPE degrades beyond threshold.")
