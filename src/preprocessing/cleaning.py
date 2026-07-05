"""Stateless cleaning transformer — mirrors notebook cells 5 & 12."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from utils.logger import get_logger

logger = get_logger(__name__)


class CleaningTransformer(BaseEstimator, TransformerMixin):
    """Drop customerID and impute TotalCharges nulls with 0.

    This replicates exactly:
      - telco_cust_df["TotalCharges"] = pd.to_numeric(..., errors="coerce")
      - telco_cust_df["TotalCharges"].fillna(0)
      - telco_cust_df.drop("customerID", axis=1)
    """

    def fit(self, X: pd.DataFrame, y=None) -> "CleaningTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()

        # Cast TotalCharges (may already be float if load_data was called)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        nulls = int(df["TotalCharges"].isna().sum())
        if nulls:
            logger.debug("Imputing %d TotalCharges nulls with 0", nulls)
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

        # Drop ID column if present
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        return df
