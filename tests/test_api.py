from fastapi.testclient import TestClient
from src.api import app
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()

@patch('src.api.model')
def test_predict_success(mock_model):
    # Mock the predict method to always return [120.0]
    mock_model.predict.return_value = [120.0]

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
    assert "timestamp" in data

def test_predict_invalid_payload():
    payload = {
        "hour": 12
        # Missing other required fields
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
