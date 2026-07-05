"""Load a trained model + preprocessing pipeline and serve predictions."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from config.config import MLFLOW_URI, MODELS_DIR, REGISTRY_NAME
from models.predict import predict_single
from pipelines.preprocessing_pipeline import build_preprocessing_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """Loads preprocessing + model and exposes a predict() method.

    The pipeline can be loaded from:
      - A local pickle file (models/preprocessing_pipe.pkl + models/<name>.pkl)
      - The MLflow Model Registry (default)
    """

    def __init__(
        self,
        preprocessing_pipe=None,
        model=None,
        threshold: float = 0.5,
        model_type: str = "tree",
    ) -> None:
        self.preprocessing_pipe = preprocessing_pipe
        self.model = model
        self.threshold = threshold
        self.model_type = model_type

    @classmethod
    def from_mlflow(
        cls,
        registry_name: str = REGISTRY_NAME,
        stage: str = "Staging",
        tracking_uri: str = MLFLOW_URI,
        threshold: float = 0.5,
        model_type: str = "tree",
    ) -> "InferencePipeline":
        """Load the registered model from MLflow and a saved preprocessing pipe."""
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{registry_name}/{stage}"
        logger.info("Loading model from MLflow: %s", model_uri)
        model = mlflow.pyfunc.load_model(model_uri)

        pipe_path = MODELS_DIR / "preprocessing_pipe.pkl"
        if not pipe_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline not found at {pipe_path}. "
                "Run the training pipeline first."
            )
        with open(pipe_path, "rb") as f:
            preprocessing_pipe = pickle.load(f)
        logger.info("Loaded preprocessing pipeline from %s", pipe_path)

        return cls(
            preprocessing_pipe=preprocessing_pipe,
            model=model,
            threshold=threshold,
            model_type=model_type,
        )

    @classmethod
    def from_disk(
        cls,
        model_path: str | Path,
        pipe_path: str | Path = MODELS_DIR / "preprocessing_pipe.pkl",
        threshold: float = 0.5,
    ) -> "InferencePipeline":
        """Load from local pickle files."""
        with open(pipe_path, "rb") as f:
            preprocessing_pipe = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model from %s", model_path)
        return cls(preprocessing_pipe=preprocessing_pipe, model=model, threshold=threshold)

    def predict(self, raw_features: dict) -> dict[str, Any]:
        """Run end-to-end prediction for a single customer."""
        return predict_single(
            model=self.model,
            preprocessing_pipeline=self.preprocessing_pipe,
            raw_features=raw_features,
            threshold=self.threshold,
        )
