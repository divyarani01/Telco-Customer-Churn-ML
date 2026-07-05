"""Encoding transformers — mirrors notebook cells 27 & 38–39."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.config import BINARY_COLUMNS, BINARY_MAPPING, NOMINAL_COLUMNS
from utils.logger import get_logger

logger = get_logger(__name__)


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Map Yes/No/Male/Female to 1/0 for binary columns.

    Mirrors:
      binary_mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
      for col in binary_columns:
          telco_cust_df[col] = telco_cust_df[col].map(binary_mapping)
    """

    def __init__(
        self,
        columns: list[str] = BINARY_COLUMNS,
        mapping: dict[str, int] = BINARY_MAPPING,
    ) -> None:
        self.columns = columns
        self.mapping = mapping

    def fit(self, X: pd.DataFrame, y=None) -> "BinaryEncoder":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = df[col].map(self.mapping)
        # SeniorCitizen is already int; ensure it stays that way
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
        return df


class NominalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode nominal (multi-category) columns with drop_first=True.

    Learns the category list during fit() so transform() on test data
    produces exactly the same columns — unseen categories get all-zero rows.

    Mirrors:
      pd.get_dummies(telco_cust_df, columns=nominal_columns,
                     drop_first=True, dtype=int)
    """

    def __init__(self, columns: list[str] = NOMINAL_COLUMNS) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "NominalEncoder":
        df = X.copy()
        # Run get_dummies on training data to learn resulting column layout
        encoded = pd.get_dummies(
            df, columns=[c for c in self.columns if c in df.columns],
            drop_first=True, dtype=int,
        )
        # Convert any stray bool columns
        bool_cols = encoded.select_dtypes(include="bool").columns
        encoded[bool_cols] = encoded[bool_cols].astype(int)
        self.feature_names_out_: list[str] = encoded.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        encoded = pd.get_dummies(
            df, columns=[c for c in self.columns if c in df.columns],
            drop_first=True, dtype=int,
        )
        bool_cols = encoded.select_dtypes(include="bool").columns
        encoded[bool_cols] = encoded[bool_cols].astype(int)

        # Align to training columns: add missing cols as 0, drop unseen
        encoded = encoded.reindex(
            columns=self.feature_names_out_, fill_value=0
        )
        return encoded

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
