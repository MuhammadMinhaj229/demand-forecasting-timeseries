from fastapi.testclient import TestClient
from src.api import app
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    # we don't strictly assert model_loaded is True here, as it might fail if model isn't built yet locally
    assert "model_loaded" in response.json()

@patch('src.api.model')
def test_predict_success(mock_model):
    # Mock the predict method to always return [120.0]
    # Handle the case where model is None (e.g. before train.py is run)
    if mock_model is not None:
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

    # We patch the prediction inside the route if model is None to avoid 500 error during tests
    with patch('src.api.model') as m:
        m.predict.return_value = [120.0]
        # We need to temporarily set the model to not None in the app module scope
        import src.api
        original_model = src.api.model
        src.api.model = m

        response = client.post("/predict", json=payload)

        # Restore original
        src.api.model = original_model

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
