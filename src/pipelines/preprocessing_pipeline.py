"""Build the preprocessing sklearn Pipeline.

Two variants:
  - "linear"  -> CleaningTransformer | BinaryEncoder | NominalEncoder
                  | RedundantDummyCollapser | StandardScaler
  - "tree"    -> same pipeline WITHOUT StandardScaler

Usage
-----
    pipe = build_preprocessing_pipeline("linear")
    X_train_proc = pipe.fit_transform(X_train)   # fit + transform on train
    X_test_proc  = pipe.transform(X_test)         # transform-only on test
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing.cleaning import CleaningTransformer
from preprocessing.encoding import BinaryEncoder, NominalEncoder
from preprocessing.transformers import RedundantDummyCollapser
from utils.logger import get_logger

logger = get_logger(__name__)

_LINEAR = "linear"
_TREE   = "tree"


def build_preprocessing_pipeline(model_type: str = _TREE) -> Pipeline:
    """Return a fitted-ready sklearn Pipeline for the given model family.

    Parameters
    ----------
    model_type : {"tree", "linear"}
        "tree"   -> no StandardScaler (Decision Tree, Random Forest,
                    XGBoost, LightGBM, CatBoost)
        "linear" -> includes StandardScaler (Logistic Regression variants)
    """
    if model_type not in {_LINEAR, _TREE}:
        raise ValueError(f"model_type must be 'linear' or 'tree', got {model_type!r}")

    steps: list[tuple] = [
        ("cleaner",     CleaningTransformer()),
        ("binary_enc",  BinaryEncoder()),
        ("nominal_enc", NominalEncoder()),
        ("feat_eng",    RedundantDummyCollapser()),
    ]

    if model_type == _LINEAR:
        steps.append(("scaler", StandardScaler()))
        logger.debug("Preprocessing pipeline: LINEAR (with StandardScaler)")
    else:
        logger.debug("Preprocessing pipeline: TREE (no StandardScaler)")

    return Pipeline(steps)
