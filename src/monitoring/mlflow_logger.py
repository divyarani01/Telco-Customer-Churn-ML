"""MLflow experiment logging — mirrors notebook cells 57 (log_mlflow_run) & 85-87."""
from __future__ import annotations

import os
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

from config.config import EXPERIMENT_NAME, MLFLOW_URI, RANDOM_STATE, REGISTRY_NAME
from models.evaluate import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from utils.logger import get_logger

logger = get_logger(__name__)


def setup_mlflow(tracking_uri: str = MLFLOW_URI) -> None:
    """Configure MLflow tracking URI and create experiment if needed."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow URI: %s | Experiment: %s", tracking_uri, EXPERIMENT_NAME)


def log_mlflow_run(
    result: dict[str, Any],
    X_te,
    y_te,
    feature_names: list[str],
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Log one model run to MLflow. Returns the run_id."""
    model      = result["_model"]
    model_name = result["Model"]

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        # --- Parameters ---
        params = model.get_params() if hasattr(model, "get_params") else {}
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("best_threshold", result["Best Threshold"])

        # --- Metrics ---
        mlflow.log_metrics({
            "accuracy"       : result["Accuracy"],
            "precision"      : result["Precision"],
            "recall"         : result["Recall"],
            "f1"             : result["F1"],
            "roc_auc"        : result["ROC-AUC"],
            "pr_auc"         : result["PR-AUC"],
            "train_time_s"   : result["Train Time (s)"],
            "inf_time_s"     : result["Inf Time (s)"],
        })

        with tempfile.TemporaryDirectory() as tmp:
            # --- Confusion Matrix ---
            fig, ax = plt.subplots(figsize=(4, 3))
            plot_confusion_matrix(y_te, result["_y_pred"], model_name, ax=ax)
            plt.tight_layout()
            cm_path = os.path.join(tmp, "confusion_matrix.png")
            fig.savefig(cm_path, dpi=100)
            plt.close(fig)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # --- ROC Curve ---
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_roc_curve(y_te, result["_y_proba"], model_name, ax=ax)
            ax.plot([0, 1], [0, 1], "k--")
            ax.legend(); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("ROC Curve")
            plt.tight_layout()
            roc_path = os.path.join(tmp, "roc_curve.png")
            fig.savefig(roc_path, dpi=100)
            plt.close(fig)
            mlflow.log_artifact(roc_path, artifact_path="plots")

            # --- PR Curve ---
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_pr_curve(y_te, result["_y_proba"], model_name, ax=ax)
            ax.legend(); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            plt.tight_layout()
            pr_path = os.path.join(tmp, "pr_curve.png")
            fig.savefig(pr_path, dpi=100)
            plt.close(fig)
            mlflow.log_artifact(pr_path, artifact_path="plots")

            # --- Feature Importance (tree/boosting models) ---
            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=feature_names)
                fig, ax = plt.subplots(figsize=(6, 5))
                fi.sort_values(ascending=True).tail(15).plot(kind="barh", ax=ax)
                ax.set_title(model_name + " -- Top 15 Features")
                ax.set_xlabel("Importance")
                plt.tight_layout()
                fi_path = os.path.join(tmp, "feature_importance.png")
                fig.savefig(fi_path, dpi=100)
                plt.close(fig)
                mlflow.log_artifact(fi_path, artifact_path="plots")

            # --- Model artifact ---
            _log_model_artifact(model, model_name)

        run_id = run.info.run_id
        logger.info("Logged '%s' -> run_id=%s", model_name, run_id)
        return run_id


def _log_model_artifact(model, model_name: str) -> None:
    """Choose the right mlflow flavour for the model type."""
    try:
        from catboost import CatBoostClassifier as _CB  # noqa: PLC0415
        if isinstance(model, _CB):
            import mlflow.catboost  # noqa: PLC0415
            mlflow.catboost.log_model(model, name="model")
            return
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier as _XGB  # noqa: PLC0415
        if isinstance(model, _XGB):
            mlflow.xgboost.log_model(model, name="model")
            return
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier as _LGBM  # noqa: PLC0415
        if isinstance(model, _LGBM):
            mlflow.lightgbm.log_model(model, name="model")
            return
    except ImportError:
        pass

    # Default: sklearn flavour with skops trusted types
    trusted = [type(model).__module__ + "." + type(model).__qualname__]
    mlflow.sklearn.log_model(
        model, name="model",
        metadata={"model_name": model_name},
    )


def register_best_model(
    run_id: str,
    model_name: str,
    registry_name: str = REGISTRY_NAME,
) -> None:
    """Register the best run in the MLflow Model Registry and tag as Staging."""
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)
    logger.info("Registered '%s' v%s from run %s", registry_name, mv.version, run_id)

    from mlflow.tracking import MlflowClient  # noqa: PLC0415
    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=registry_name, version=mv.version, stage="Staging",
        )
        logger.info("%s v%s -> Staging", registry_name, mv.version)
    except Exception:
        client.set_model_version_tag(registry_name, mv.version, "stage", "Staging")
        logger.info("%s v%s tagged as Staging (MLflow 3.x)", registry_name, mv.version)
