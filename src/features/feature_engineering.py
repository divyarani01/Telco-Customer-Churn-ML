"""VIF analysis utilities (exploratory — not part of the inference pipeline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for every column in X.

    Mirrors notebook cells 34 & 45.
    Removes rows with NaN/Inf before calculation.
    """
    df = X.copy().replace([np.inf, -np.inf], np.nan).dropna()
    # Cast any bool columns
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    vif = pd.DataFrame({
        "Feature": df.columns,
        "VIF": [
            variance_inflation_factor(df.values, i)
            for i in range(df.shape[1])
        ],
    })
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)


def interpret_vif(vif: float) -> str:
    """Return Low / Moderate / High label for a VIF value."""
    if vif < 5:
        return "Low"
    elif vif < 10:
        return "Moderate"
    return "High"
