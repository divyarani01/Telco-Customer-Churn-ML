"""Load raw Telco Customer Churn CSV into a DataFrame."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from config.config import RAW_DATA_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and cast TotalCharges to float.

    This mirrors the first two steps of the notebook exactly:
      - pd.read_csv
      - pd.to_numeric(errors='coerce') on TotalCharges
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.info("Loaded %d rows x %d columns", *df.shape)
    return df
