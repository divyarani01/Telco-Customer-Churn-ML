"""Tests for the FastAPI prediction endpoint (uses TestClient, no server needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VALID_PAYLOAD = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 844.20,
}

MOCK_REGISTRY = {
    "CatBoost": {
        "safe_name"  : "CatBoost",
        "model_type" : "tree",
        "model_class": "CatBoostClassifier",
        "model_path" : "/fake/CatBoost.pkl",
        "pipe_path"  : "/fake/CatBoost_pipe.pkl",
        "threshold"  : 0.52,
        "metrics"    : {"accuracy": 0.82, "precision": 0.68, "recall": 0.76,
                        "f1": 0.72, "roc_auc": 0.87, "pr_auc": 0.74},
        "train_time_s": 12.3,
    }
}

MOCK_PREDICT_RESULT = {
    "prediction": 1, "probability": 0.72,
    "churn_label": "Yes", "threshold": 0.52,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    mock_model = MagicMock()
    mock_pipe  = MagicMock()

    with patch("utils.model_store.load_registry", return_value=MOCK_REGISTRY), \
         patch("utils.model_store.load_model_by_name",
               return_value=(mock_model, mock_pipe, 0.52)), \
         patch("models.predict.predict_single", return_value=MOCK_PREDICT_RESULT):
        from api.main import app
        yield TestClient(app)


class TestHealthEndpoint:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_lists_available_models(self, client):
        r = client.get("/health")
        assert "model_names" in r.json()


class TestModelsEndpoint:
    def test_get_models_200(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_model_has_metrics(self, client):
        r = client.get("/models")
        first = r.json()[0]
        assert "metrics" in first
        assert "roc_auc" in first["metrics"]

    def test_paths_not_exposed(self, client):
        r = client.get("/models")
        first = r.json()[0]
        assert "model_path" not in first
        assert "pipe_path" not in first

    def test_get_single_model(self, client):
        r = client.get("/models/CatBoost")
        assert r.status_code == 200
        assert r.json()["model_class"] == "CatBoostClassifier"

    def test_get_unknown_model_404(self, client):
        r = client.get("/models/NonExistentModel")
        assert r.status_code == 404


class TestPredictEndpoint:
    def test_valid_payload_200(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_fields(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        body = r.json()
        for field in ("model_name", "prediction", "probability",
                      "churn_label", "threshold", "timestamp"):
            assert field in body

    def test_prediction_is_binary(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.json()["prediction"] in (0, 1)

    def test_probability_in_range(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert 0.0 <= r.json()["probability"] <= 1.0

    def test_model_name_in_response(self, client):
        r = client.post("/predict?model_name=CatBoost", json=VALID_PAYLOAD)
        assert r.json()["model_name"] == "CatBoost"

    def test_unknown_model_404(self, client):
        with patch("api.main.load_model_by_name",
                   side_effect=KeyError("Model 'Foo' not found")):
            r = client.post("/predict?model_name=Foo", json=VALID_PAYLOAD)
        assert r.status_code == 404

    def test_missing_field_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422
