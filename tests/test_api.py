import pytest
import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api import app, calculate_inventory_policy

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()

def test_inventory_policy_calculation():
    policy = calculate_inventory_policy(
        daily_demand=120.0,
        demand_std=20.0,
        lead_time_days=5,
        service_level=0.95,
        order_cost=50.0,
        holding_cost_annual=15.0
    )
    assert policy["safety_stock"] > 0
    assert policy["reorder_point"] > policy["safety_stock"]
    assert policy["economic_order_quantity"] > 0

def test_predict_success():
    payload = {
        "hour": 12,
        "temp": 20.0,
        "is_holiday": 0,
        "day_of_week": 2,
        "month": 6,
        "quarter": 2,
        "is_weekend": 0,
        "lag_7": 100.0,
        "lag_14": 110.0,
        "lag_30": 105.0,
        "rolling_mean_7": 105.0,
        "rolling_std_7": 15.0,
        "rolling_mean_30": 102.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_demand" in data
    assert "quantile_p10" in data
    assert "quantile_p90" in data
