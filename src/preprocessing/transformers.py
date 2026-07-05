"""Feature-engineering transformer — mirrors notebook cells 43–44."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from utils.logger import get_logger

logger = get_logger(__name__)

_NIS = "No internet service"
_NPS = "No phone service"


class RedundantDummyCollapser(BaseEstimator, TransformerMixin):
    """Collapse redundant OHE sentinel columns into single indicator flags.

    After get_dummies, each service column (OnlineSecurity, OnlineBackup …)
    gets a '…_No internet service' dummy.  These are all perfectly correlated
    with InternetService_No, so we collapse them into one 'No_internet_service'
    flag and drop the originals.

    Likewise 'MultipleLines_No phone service' -> 'No_phone_service'.

    Mirrors notebook cells 43-44 exactly.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "RedundantDummyCollapser":
        # Record which columns we expect to collapse so transform is consistent
        self.nis_cols_: list[str] = [
            c for c in X.columns if _NIS in c
        ]
        self.nps_col_: str | None = (
            "MultipleLines_No phone service"
            if "MultipleLines_No phone service" in X.columns
            else None
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()

        # Collapse "No internet service" sentinels
        nis_cols = [c for c in self.nis_cols_ if c in df.columns]
        if nis_cols:
            df["No_internet_service"] = df[nis_cols].any(axis=1).astype(int)
            df = df.drop(columns=nis_cols)
            logger.debug("Collapsed %d 'No internet service' columns", len(nis_cols))

        # Collapse "No phone service" sentinel
        if self.nps_col_ and self.nps_col_ in df.columns:
            df["No_phone_service"] = df[self.nps_col_].astype(int)
            df = df.drop(columns=[self.nps_col_])
            logger.debug("Collapsed 'MultipleLines_No phone service'")

        return df
