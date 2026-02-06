#FastAPI inference service loading the MLflow model

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import mlflow
import mlflow.pyfunc

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field



# ============================================================
# Config 
# ============================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://84.247.128.24:5000")
RUN_ID = os.getenv("RUN_ID", "").strip()
#alternative:
# MODEL_URI = os.getenv("MODEL_URI", f"runs:/{RUN_ID}/model")


FEATURE_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "pickup_hour",
    "pickup_weekday",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
    "payment_type",
    "VendorID",
    "store_and_fwd_flag",
]


# ============================================================
# Request Schema
# ============================================================
class TaxiFeatures(BaseModel):
    passenger_count: int = Field(..., ge=0, le=8)
    trip_distance: float = Field(..., gt=0)
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_weekday: int = Field(..., ge=0, le=6)
    PULocationID: int = Field(..., ge=1)
    DOLocationID: int = Field(..., ge=1)
    RatecodeID: int = Field(..., ge=1)
    payment_type: int = Field(..., ge=1)
    VendorID: int = Field(..., ge=1)
    store_and_fwd_flag: str = Field(..., min_length=1, max_length=1)  


# ============================================================
# App + Model loading
# ============================================================
app = FastAPI(title="Yellow Taxi Time Estimation API", version="1.0.0")

model: Any | None = None


@app.on_event("startup")
def load_model() -> None:
    """
    Lädt das Modell einmal beim Start, hält es dann im RAM.
    """
    global model

    if not RUN_ID and "runs:/" in MODEL_URI:
        # Wenn MODEL_URI runs:/... verwendet, brauchst du eine RUN_ID
        raise RuntimeError(
            "RUN_ID ist nicht gesetzt. Setze RUN_ID oder setze MODEL_URI explizit."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception as e:
        raise RuntimeError(
            f"Model load failed. tracking_uri={MLFLOW_TRACKING_URI}, model_uri={MODEL_URI}. Error: {e}"
        ) from e


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict")
def predict(payload: TaxiFeatures) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Pydantic v1/v2 kompatibel
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()

    # DataFrame 
    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

    try:
        y_pred = model.predict(df)
        pred_sec = float(y_pred[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "duration_sec": pred_sec,
        "duration_min": pred_sec / 60.0,
    }
