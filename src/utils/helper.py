"""Miscellaneous utilities."""
from __future__ import annotations

import pickle
from pathlib import Path

import shap
import matplotlib.pyplot as plt

from utils.logger import get_logger

logger = get_logger(__name__)


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved to %s", path)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_shap(model, X_test, feature_names: list[str], max_display: int = 20) -> None:
    """Generate SHAP beeswarm + bar plots for the best model.

    Uses TreeExplainer for tree/boosting models, LinearExplainer for LR.
    Mirrors notebook cell 83 exactly.
    """
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        X_display = X_test
    else:
        explainer = shap.LinearExplainer(
            model, X_test, feature_perturbation="interventional"
        )
        sv = explainer.shap_values(X_test)
        X_display = X_test

    plt.figure()
    shap.summary_plot(sv, X_display, feature_names=feature_names,
                      max_display=max_display, show=True)

    plt.figure()
    shap.summary_plot(sv, X_display, feature_names=feature_names,
                      plot_type="bar", max_display=max_display, show=True)
