"""Tests for evaluate_model, threshold_optimization, and run_model_workflow."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def synthetic_data():
    X, y = make_classification(
        n_samples=300, n_features=10, random_state=42, class_sep=1.0
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_tr), X_te := scaler.transform(X_te),
        y_tr, y_te,
    )


@pytest.fixture(scope="module")
def fitted_lr(synthetic_data):
    X_tr, _, y_tr, _ = synthetic_data
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_tr, y_tr)
    return model


class TestEvaluateModel:
    def test_returns_all_keys(self, fitted_lr, synthetic_data):
        from models.evaluate import evaluate_model
        _, X_te, _, y_te = synthetic_data
        result = evaluate_model(fitted_lr, X_te, y_te)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc",
                    "pr_auc", "threshold", "inf_time", "y_proba", "y_pred"):
            assert key in result

    def test_roc_auc_range(self, fitted_lr, synthetic_data):
        from models.evaluate import evaluate_model
        _, X_te, _, y_te = synthetic_data
        result = evaluate_model(fitted_lr, X_te, y_te)
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_custom_threshold(self, fitted_lr, synthetic_data):
        from models.evaluate import evaluate_model
        _, X_te, _, y_te = synthetic_data
        r_low  = evaluate_model(fitted_lr, X_te, y_te, threshold=0.3)
        r_high = evaluate_model(fitted_lr, X_te, y_te, threshold=0.7)
        # Lower threshold -> higher recall
        assert r_low["recall"] >= r_high["recall"]


class TestThresholdOptimization:
    def test_returns_float_in_range(self, fitted_lr, synthetic_data):
        from models.evaluate import threshold_optimization
        _, X_te, _, y_te = synthetic_data
        proba = fitted_lr.predict_proba(X_te)[:, 1]
        best_t = threshold_optimization(y_te, proba)
        assert 0.10 <= best_t <= 0.90

    def test_improves_f1(self, fitted_lr, synthetic_data):
        from models.evaluate import evaluate_model, threshold_optimization
        _, X_te, _, y_te = synthetic_data
        proba  = fitted_lr.predict_proba(X_te)[:, 1]
        best_t = threshold_optimization(y_te, proba)
        r_default = evaluate_model(fitted_lr, X_te, y_te, threshold=0.5)
        r_optimal  = evaluate_model(fitted_lr, X_te, y_te, threshold=best_t)
        assert r_optimal["f1"] >= r_default["f1"] - 0.01  # at least as good


class TestBuildLeaderboard:
    def test_sorted_by_recall(self):
        from models.evaluate import build_leaderboard
        rows = [
            {"Model": "A", "Accuracy": 0.8, "Precision": 0.7, "Recall": 0.6,
             "F1": 0.65, "ROC-AUC": 0.85, "PR-AUC": 0.70,
             "Best Threshold": 0.5, "Train Time (s)": 1.0, "Inf Time (s)": 0.001},
            {"Model": "B", "Accuracy": 0.9, "Precision": 0.8, "Recall": 0.9,
             "F1": 0.85, "ROC-AUC": 0.80, "PR-AUC": 0.75,
             "Best Threshold": 0.5, "Train Time (s)": 2.0, "Inf Time (s)": 0.001},
        ]
        lb = build_leaderboard(rows)
        assert lb.iloc[0]["Model"] == "B"  # higher recall -> rank 1
