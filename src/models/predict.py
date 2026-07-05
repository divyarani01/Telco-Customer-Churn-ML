"""Prediction helpers used by both the training pipeline and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def predict_proba(model, X) -> np.ndarray:
    """Return churn probability for each row in X."""
    return model.predict_proba(X)[:, 1]


def predict(model, X, threshold: float = 0.5) -> np.ndarray:
    """Return binary churn prediction at *threshold*."""
    proba = predict_proba(model, X)
    return (proba >= threshold).astype(int)


def predict_single(
    model,
    preprocessing_pipeline,
    raw_features: dict,
    threshold: float = 0.5,
) -> dict:
    """End-to-end prediction for a single customer (used by FastAPI).

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
    preprocessing_pipeline : fitted preprocessing Pipeline
    raw_features : dict of raw column values (as received from the API)
    threshold : decision threshold

    Returns
    -------
    dict with keys: prediction, probability, threshold
    """
    df = pd.DataFrame([raw_features])
    X_proc = preprocessing_pipeline.transform(df)
    proba = float(predict_proba(model, X_proc)[0])
    pred  = int(proba >= threshold)
    logger.debug("predict_single: proba=%.4f, pred=%d, threshold=%.2f", proba, pred, threshold)
    return {
        "prediction" : pred,
        "probability": round(proba, 4),
        "threshold"  : threshold,
        "churn_label": "Yes" if pred == 1 else "No",
    }
