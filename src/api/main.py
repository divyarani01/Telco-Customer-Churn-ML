"""FastAPI inference service for Customer Churn prediction.

Start with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

POST /predict  ->  {"prediction": 0, "probability": 0.23, ...}
GET  /health   ->  {"status": "ok"}
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.config import MLFLOW_URI, REGISTRY_NAME
from pipelines.inference_pipeline import InferencePipeline
from utils.logger import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(application: FastAPI):
    load_pipeline()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a Telco customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """Raw customer features (pre-preprocessing)."""
    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35, "TotalCharges": 844.20,
            }
        }
    }

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    churn_label: str
    threshold: float
    model_version: str
    timestamp: str


# ---------------------------------------------------------------------------
# Startup: load pipeline once
# ---------------------------------------------------------------------------

_pipeline: InferencePipeline | None = None
_model_version: str = "unknown"


def load_pipeline() -> None:
    global _pipeline, _model_version

    models_dir = Path(__file__).resolve().parents[2] / "models"
    model_path = models_dir / "best_model.pkl"
    pipe_path  = models_dir / "preprocessing_pipe.pkl"
    thresh_path = models_dir / "threshold.txt"

    # Load threshold if saved, otherwise default
    threshold = 0.5
    if thresh_path.exists():
        threshold = float(thresh_path.read_text().strip())

    # Prefer disk (works in Docker without MLflow artifact store access)
    if model_path.exists() and pipe_path.exists():
        _pipeline = InferencePipeline.from_disk(
            model_path=model_path,
            pipe_path=pipe_path,
            threshold=threshold,
        )
        _model_version = "v1-disk"
        logger.info("Loaded model from disk (threshold=%.2f)", threshold)
        return

    # Fallback: MLflow registry (works locally, not in Docker)
    try:
        _pipeline = InferencePipeline.from_mlflow(
            registry_name=REGISTRY_NAME,
            stage="Staging",
            tracking_uri=MLFLOW_URI,
            threshold=threshold,
        )
        _model_version = "Staging"
        logger.info("Loaded model from MLflow registry")
    except Exception as exc:
        logger.error("No model found (%s). /predict will return 503.", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status"       : "ok",
        "model_loaded" : _pipeline is not None,
        "model_version": _model_version,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    raw = features.model_dump()
    try:
        result = _pipeline.predict(raw)
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictionResponse(
        prediction   = result["prediction"],
        probability  = result["probability"],
        churn_label  = result["churn_label"],
        threshold    = result["threshold"],
        model_version= _model_version,
        timestamp    = datetime.now(timezone.utc).isoformat(),
    )
