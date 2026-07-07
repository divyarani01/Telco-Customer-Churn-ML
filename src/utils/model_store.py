"""Disk-based model store.

Saves / loads models, preprocessing pipelines, and a JSON registry
so the API can serve any trained model without touching MLflow artifacts.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from config.config import MODELS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

REGISTRY_PATH = MODELS_DIR / "registry.json"


def _safe_name(model_name: str) -> str:
    """Convert e.g. 'Ridge LR (L2)' -> 'Ridge_LR_L2' for use as a filename."""
    return re.sub(r"[^\w]", "_", model_name).strip("_")


def save_model(
    model,
    preprocessing_pipe,
    result: dict[str, Any],
    model_type: str,
) -> None:
    """Persist a trained model, its pipeline, threshold, and registry entry."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = result["Model"]
    safe = _safe_name(name)

    # Model
    model_path = MODELS_DIR / f"{safe}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Preprocessing pipeline
    pipe_path = MODELS_DIR / f"{safe}_pipe.pkl"
    with open(pipe_path, "wb") as f:
        pickle.dump(preprocessing_pipe, f)

    # Update registry — store filenames only so paths resolve correctly on any OS
    registry = load_registry()
    registry[name] = {
        "safe_name"      : safe,
        "model_type"     : model_type,
        "model_class"    : type(model).__name__,
        "model_path"     : model_path.name,
        "pipe_path"      : pipe_path.name,
        "threshold"      : result["Best Threshold"],
        "metrics": {
            "accuracy" : result["Accuracy"],
            "precision": result["Precision"],
            "recall"   : result["Recall"],
            "f1"       : result["F1"],
            "roc_auc"  : result["ROC-AUC"],
            "pr_auc"   : result["PR-AUC"],
        },
        "train_time_s"   : result["Train Time (s)"],
    }
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("Saved '%s' -> %s", name, model_path)


def load_model_by_name(model_name: str):
    """Return (model, preprocessing_pipe, threshold) for the given model name."""
    registry = load_registry()
    if model_name not in registry:
        available = list(registry.keys())
        raise KeyError(f"Model '{model_name}' not found. Available: {available}")

    entry = registry[model_name]
    with open(MODELS_DIR / entry["model_path"], "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / entry["pipe_path"], "rb") as f:
        pipe = pickle.load(f)
    threshold = float(entry["threshold"])
    return model, pipe, threshold


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def list_models() -> list[dict]:
    """Return all registry entries sorted by Recall -> F1 -> ROC-AUC descending.

    Recall is the primary sort key because missing a churner (false negative)
    is more costly than a false alarm.  Consistent with the training leaderboard.
    """
    registry = load_registry()
    rows = [{"model_name": name, **entry} for name, entry in registry.items()]
    return sorted(
        rows,
        key=lambda r: (
            r["metrics"]["recall"],
            r["metrics"]["f1"],
            r["metrics"]["roc_auc"],
        ),
        reverse=True,
    )
