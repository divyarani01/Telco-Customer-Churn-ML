"""Tests for the FastAPI prediction endpoint (uses TestClient, no server needed)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VALID_PAYLOAD = {
    "gender"          : "Male",
    "SeniorCitizen"   : 0,
    "Partner"         : "Yes",
    "Dependents"      : "No",
    "tenure"          : 12,
    "PhoneService"    : "Yes",
    "MultipleLines"   : "No",
    "InternetService" : "Fiber optic",
    "OnlineSecurity"  : "No",
    "OnlineBackup"    : "Yes",
    "DeviceProtection": "No",
    "TechSupport"     : "No",
    "StreamingTV"     : "Yes",
    "StreamingMovies" : "Yes",
    "Contract"        : "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod"   : "Electronic check",
    "MonthlyCharges"  : 70.35,
    "TotalCharges"    : 844.20,
}


@pytest.fixture(scope="module")
def client():
    """Return a TestClient with the model pipeline mocked out."""
    from fastapi.testclient import TestClient

    mock_result = {
        "prediction" : 1,
        "probability": 0.72,
        "churn_label": "Yes",
        "threshold"  : 0.5,
    }
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = mock_result

    # Patch before importing the app so startup uses the mock
    with patch("api.main._pipeline", mock_pipeline), \
         patch("api.main._model_version", "test-v1"):
        from api.main import app
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_valid_payload_returns_200(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        body = r.json()
        for field in ("prediction", "probability", "churn_label",
                      "threshold", "model_version", "timestamp"):
            assert field in body, f"Missing field: {field}"

    def test_prediction_is_binary(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.json()["prediction"] in (0, 1)

    def test_probability_in_range(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        prob = r.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_missing_field_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422
