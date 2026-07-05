"""evaluate_model(), threshold_optimization(), and plot helpers — notebook cell 57."""
from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_te, y_te, threshold: float = 0.5) -> dict[str, Any]:
    """Compute all classification metrics at *threshold*.

    Returns a dict with: accuracy, precision, recall, f1, roc_auc, pr_auc,
    threshold, inf_time, y_proba, y_pred.
    """
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    t0 = time.perf_counter()
    model.predict(X_te)
    inf_time = round(time.perf_counter() - t0, 6)

    return {
        "accuracy" : round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_te, y_pred), 4),
        "f1"       : round(f1_score(y_te, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_te, y_proba), 4),
        "pr_auc"   : round(average_precision_score(y_te, y_proba), 4),
        "threshold": threshold,
        "inf_time" : inf_time,
        "y_proba"  : y_proba,
        "y_pred"   : y_pred,
    }


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def threshold_optimization(
    y_true,
    y_proba,
    metric: str = "f1",
) -> float:
    """Sweep thresholds 0.10-0.90 and return the one that maximises F1.

    Mirrors notebook cell 57 exactly.
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = [
        f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_t = round(float(thresholds[int(np.argmax(scores))]), 2)
    logger.debug("threshold_optimization: best_t=%.2f, best_f1=%.4f", best_t, max(scores))
    return best_t


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    ax: plt.Axes | None = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {model_name}")
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_roc_curve(
    y_true,
    y_proba,
    model_name: str,
    ax: plt.Axes | None = None,
    color=None,
) -> tuple:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(5, 4))
    kw: dict = {"label": f"{model_name} (AUC={auc_val:.3f})"}
    if color is not None:
        kw["color"] = color
    ax.plot(fpr, tpr, **kw)
    if standalone:
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve"); ax.legend()
        plt.tight_layout(); plt.show()
    return fpr, tpr, auc_val


def plot_pr_curve(
    y_true,
    y_proba,
    model_name: str,
    ax: plt.Axes | None = None,
    color=None,
) -> tuple:
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(5, 4))
    kw: dict = {"label": f"{model_name} (AP={ap:.3f})"}
    if color is not None:
        kw["color"] = color
    ax.plot(rec, prec, **kw)
    if standalone:
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve"); ax.legend()
        plt.tight_layout(); plt.show()
    return prec, rec, ap


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard(results: list[dict]) -> pd.DataFrame:
    """Build ranked leaderboard from results_list, sorted by Recall/F1/ROC-AUC."""
    cols = [
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "ROC-AUC", "PR-AUC", "Best Threshold", "Train Time (s)", "Inf Time (s)",
    ]
    lb = (
        pd.DataFrame(results)[cols]
        .sort_values(
            by=["Recall", "F1", "ROC-AUC"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    lb.index += 1
    return lb
