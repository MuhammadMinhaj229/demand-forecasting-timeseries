from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Demand Forecasting API")

# Load trained model
try:
    model = joblib.load('models/lgbm_forecast.pkl')
except Exception as e:
    model = None
    print(f"Warning: Model not found. {e}")

class ForecastRequest(BaseModel):
    hour: int
    temp: float
    is_holiday: int
    day_of_week: int
    month: int
    quarter: int
    is_weekend: int
    lag_7: float
    lag_14: float
    lag_30: float
    rolling_mean_7: float
    rolling_std_7: float
    rolling_mean_30: float

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_demand(req: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Create DataFrame from request
        df = pd.DataFrame([req.model_dump()])

        # Make prediction
        prediction = model.predict(df)[0]

        return {
            "timestamp": datetime.now().isoformat(),
            "predicted_demand": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
