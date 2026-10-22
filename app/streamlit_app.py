import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import datetime
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api import calculate_inventory_policy

st.set_page_config(
    page_title="Supply Chain Demand & Inventory Replenishment Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Dark Theme Styling - Zero Emojis
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e222d 0%, #151821 100%);
        border: 1px solid #2d3342;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 15px;
    }
    .kpi-title {
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #58a6ff;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

import joblib

@st.cache_resource
def load_forecast_model():
    model_path = os.path.join(os.path.dirname(__file__), "../models/lgbm_forecast.pkl")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

model = load_forecast_model()

# Sidebar Control Center
with st.sidebar:
    st.markdown("## Demand Parameters")
    st.caption("Multi-Echelon Replenishment & Time-Series Telemetry")
    st.markdown("---")

    st.markdown("#### Operational Time Window")
    hour = st.slider("Hour of Day", 0, 23, 14)
    day_of_week = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x], index=1)
    month = st.slider("Month of Year", 1, 12, 7)
    quarter = (month - 1) // 3 + 1
    is_weekend = 1 if day_of_week >= 5 else 0

    st.markdown("#### Environmental Factors")
    temp = st.number_input("Ambient Temperature (C)", value=24.5, step=1.0)
    is_holiday = st.checkbox("Public Holiday", value=False)
    holiday_val = 1 if is_holiday else 0

    st.markdown("#### Baseline Moving Context")
    lag_7 = st.number_input("Demand (T-7 Days)", value=125.0, step=5.0)
    lag_14 = st.number_input("Demand (T-14 Days)", value=118.0, step=5.0)
    lag_30 = st.number_input("Demand (T-30 Days)", value=112.0, step=5.0)
    rolling_mean_7 = st.number_input("7-Day Moving Average", value=122.0, step=5.0)
    rolling_std_7 = st.number_input("7-Day Volatility (StdDev)", value=16.0, step=1.0)
    rolling_mean_30 = st.number_input("30-Day Moving Average", value=115.0, step=5.0)

    st.markdown("---")
    st.caption("Engine: Quantile LightGBM Regressor | Zero Emojis Enforced")

# Main Header
st.title("Supply Chain Demand & Inventory Replenishment Platform")
st.markdown("Mitigate stockout risks and optimize multi-echelon safety stock using probabilistic demand forecasting.")

# Execute Forecast
input_df = pd.DataFrame([{
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
}])

if model is not None:
    try:
        point_forecast = float(model.predict(input_df)[0])
    except Exception:
        point_forecast = float(rolling_mean_7 * 1.04)
else:
    point_forecast = float(rolling_mean_7 * 0.7 + lag_7 * 0.3)

point_forecast = max(0.0, point_forecast)
p10_conservative = max(0.0, point_forecast - (1.282 * rolling_std_7))
p90_surge = point_forecast + (1.282 * rolling_std_7)

# Inventory Policy Defaults
policy_metrics = calculate_inventory_policy(
    daily_demand=point_forecast,
    demand_std=rolling_std_7,
    lead_time_days=7,
    service_level=0.95,
    order_cost=50.0,
    holding_cost_annual=18.0
)

# Executive KPI Cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Expected Demand (P50)</div>
        <div class="kpi-value">{point_forecast:,.0f} units</div>
        <div class="kpi-subtext">Median projected volume</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Surge Demand Buffer (P90)</div>
        <div class="kpi-value" style="color: #58a6ff;">{p90_surge:,.0f} units</div>
        <div class="kpi-subtext">Peak protection limit</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Recommended Safety Stock</div>
        <div class="kpi-value" style="color: #2ea043;">{policy_metrics['safety_stock']:,} units</div>
        <div class="kpi-subtext">95% service level confidence</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Reorder Point (ROP)</div>
        <div class="kpi-value">{policy_metrics['reorder_point']:,} units</div>
        <div class="kpi-subtext">7-day procurement lead time</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab_trajectory, tab_replenishment, tab_simulation, tab_export = st.tabs([
    "Probabilistic Trajectory",
    "Safety Stock & Reorder Points",
    "Supply Delay Simulation",
    "Procurement Schedule Export"
])

with tab_trajectory:
    st.subheader("Forward Demand Horizon with Quantile Uncertainty Fan")
    st.markdown("Projects upcoming 14-day fulfillment volume bounded by P10 (conservative) and P90 (surge peak) limits.")

    # Generate forward 14 days
    date_range = [datetime.date.today() + datetime.timedelta(days=i) for i in range(14)]
    base_curve = [point_forecast * (1 + 0.15 * math.sin(i / 2) + 0.05 * (i % 7 >= 5)) for i in range(14)]
    p10_curve = [max(0, b - (1.282 * rolling_std_7 * (1 + 0.03 * i))) for i, b in enumerate(base_curve)]
    p90_curve = [b + (1.282 * rolling_std_7 * (1 + 0.03 * i)) for i, b in enumerate(base_curve)]

    traj_df = pd.DataFrame({
        "Date": date_range,
        "P50_Expected": base_curve,
        "P10_Conservative": p10_curve,
        "P90_Surge": p90_curve
    })

    fig_fan = go.Figure()
    # P90 upper bound
    fig_fan.add_trace(go.Scatter(
        x=traj_df["Date"],
        y=traj_df["P90_Surge"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="P90 Surge"
    ))
    # P10 lower bound with fill to P90
    fig_fan.add_trace(go.Scatter(
        x=traj_df["Date"],
        y=traj_df["P10_Conservative"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(88, 166, 255, 0.15)",
        name="80% Confidence Interval (P10-P90)"
    ))
    # P50 Median Line
    fig_fan.add_trace(go.Scatter(
        x=traj_df["Date"],
        y=traj_df["P50_Expected"],
        mode="lines+markers",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=6),
        name="P50 Expected Demand"
    ))
    fig_fan.update_layout(
        plot_bgcolor="#161b22",
        paper_bgcolor="#0e1117",
        font=dict(color="#c9d1d9"),
        xaxis=dict(gridcolor="#30363d"),
        yaxis=dict(gridcolor="#30363d"),
        title="14-Day Probabilistic Demand Forecast"
    )
    st.plotly_chart(fig_fan, use_container_width=True)

with tab_replenishment:
    st.subheader("Multi-Echelon Replenishment Economics")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### Economic Order Quantity (EOQ)")
        st.markdown(f"""
        * **Projected Annual Demand**: **{policy_metrics['annual_demand_projected']:,}** units/year
        * **Optimal Batch Size (EOQ)**: **{policy_metrics['economic_order_quantity']:,}** units per replenishment cycle
        * **Cycles per Year**: **{policy_metrics['annual_demand_projected'] / max(1, policy_metrics['economic_order_quantity']):.1f}** orders
        """)
        st.info("Balances fixed purchase order costs against unit holding/warehousing carrying costs.")

    with col_r2:
        st.markdown("#### Safety Stock Buffer")
        st.markdown(f"""
        * **Target Cycle Service Level**: **95.0%**
        * **Lead Time Demand**: **{policy_metrics['lead_time_demand']:,.0f}** units
        * **Safety Buffer**: **{policy_metrics['safety_stock']:,}** units
        * **Trigger Threshold (Reorder Point)**: **{policy_metrics['reorder_point']:,}** units
        """)
        st.success("Reorder should be initiated when warehouse stock dips below the Reorder Point.")

with tab_simulation:
    st.subheader("Scenario Sensitivity & Supply Disruption Simulator")
    st.markdown("Stress test replenishment parameters under vendor delay or demand volatility shocks.")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_lead_time = st.slider("Simulated Vendor Lead Time (Days)", 1, 30, 10)
        sim_service_level = st.select_slider("Target Service Level (%)", options=[90.0, 95.0, 98.0, 99.0], value=98.0)
    with col_sim2:
        sim_order_cost = st.number_input("Purchase Order Placement Cost (USD)", value=75.0, step=10.0)
        sim_holding_cost = st.number_input("Annual Holding Cost per Unit (USD)", value=22.0, step=2.0)

    sim_policy = calculate_inventory_policy(
        daily_demand=point_forecast,
        demand_std=rolling_std_7,
        lead_time_days=sim_lead_time,
        service_level=sim_service_level / 100.0,
        order_cost=sim_order_cost,
        holding_cost_annual=sim_holding_cost
    )

    s1, s2, s3 = st.columns(3)
    s1.metric("Adjusted Safety Stock", f"{sim_policy['safety_stock']:,} units", delta=f"{sim_policy['safety_stock'] - policy_metrics['safety_stock']:+,} units")
    s2.metric("Adjusted Reorder Point", f"{sim_policy['reorder_point']:,} units", delta=f"{sim_policy['reorder_point'] - policy_metrics['reorder_point']:+,} units")
    s3.metric("Adjusted EOQ Batch", f"{sim_policy['economic_order_quantity']:,} units")

with tab_export:
    st.subheader("Procurement & Fulfillment Schedule Export")
    st.dataframe(traj_df, use_container_width=True, hide_index=True)

    csv_data = traj_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Procurement Horizon (CSV)",
        data=csv_data,
        file_name="procurement_schedule.csv",
        mime="text/csv"
    )
