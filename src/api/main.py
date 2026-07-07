"""FastAPI inference service for Customer Churn prediction.

Start with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health                        - Service health + loaded model count
GET  /models                        - List all trained models with metrics
GET  /models/{model_name}           - Get one model's details
POST /predict?model_name=CatBoost   - Predict churn for a customer
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from utils.logger import get_logger
from utils.model_store import list_models, load_model_by_name, load_registry
from models.predict import predict_single

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
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
    model_name: str
    prediction: int
    probability: float
    churn_label: str
    threshold: float
    timestamp: str


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    models = list_models()
    if models:
        logger.info("Model store loaded — %d model(s) available", len(models))
        for m in models:
            logger.info("  %-20s  ROC-AUC=%.4f  threshold=%.2f",
                        m["model_name"], m["metrics"]["roc_auc"], m["threshold"])
    else:
        logger.warning("No trained models found in models/. Run the training pipeline first.")
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Serve predictions from any trained churn model.\n\n"
        "1. Train all models via the notebook or `scripts/train.py`\n"
        "2. `GET /models` to see what's available\n"
        "3. `POST /predict?model_name=CatBoost` to get a prediction"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health() -> dict:
    models = list_models()
    return {
        "status"        : "ok",
        "models_available": len(models),
        "model_names"   : [m["model_name"] for m in models],
    }


@app.get("/models", summary="List all trained models")
def get_models() -> list[dict]:
    """Returns all models sorted by Recall -> F1 -> ROC-AUC descending."""
    models = list_models()
    if not models:
        raise HTTPException(
            status_code=404,
            detail="No trained models found. Run the training pipeline first.",
        )
    # Strip internal path fields
    return [
        {k: v for k, v in m.items() if k not in ("model_path", "pipe_path")}
        for m in models
    ]


@app.get("/models/{model_name}", summary="Get a specific model's details")
def get_model(model_name: str) -> dict:
    registry = load_registry()
    if model_name not in registry:
        available = list(registry.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {available}",
        )
    entry = registry[model_name]
    return {k: v for k, v in entry.items() if k not in ("model_path", "pipe_path")}


@app.post("/predict", response_model=PredictionResponse, summary="Predict churn")
def predict(
    features: CustomerFeatures,
    model_name: Optional[str] = Query(
        default=None,
        description="Name of the model to use (e.g. 'CatBoost'). "
                    "Defaults to the best model by ROC-AUC. "
                    "Call GET /models to see all options.",
    ),
) -> PredictionResponse:
    # Default to best model if not specified
    if model_name is None:
        models = list_models()
        if not models:
            raise HTTPException(
                status_code=503,
                detail="No trained models available. Run the training pipeline first.",
            )
        model_name = models[0]["model_name"]
        logger.debug("No model_name specified, defaulting to '%s'", model_name)

    try:
        model, preprocessing_pipe, threshold = load_model_by_name(model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        result = predict_single(
            model=model,
            preprocessing_pipeline=preprocessing_pipe,
            raw_features=features.model_dump(),
            threshold=threshold,
        )
    except Exception as exc:
        logger.exception("Prediction error for model '%s': %s", model_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictionResponse(
        model_name  = model_name,
        prediction  = result["prediction"],
        probability = result["probability"],
        churn_label = result["churn_label"],
        threshold   = result["threshold"],
        timestamp   = datetime.now(timezone.utc).isoformat(),
    )
