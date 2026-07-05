"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Minimal synthetic raw DataFrame matching the Telco schema."""
    return pd.DataFrame({
        "customerID"      : ["1234-ABCD", "5678-EFGH"],
        "gender"          : ["Male", "Female"],
        "SeniorCitizen"   : [0, 1],
        "Partner"         : ["Yes", "No"],
        "Dependents"      : ["No", "Yes"],
        "tenure"          : [12, 36],
        "PhoneService"    : ["Yes", "Yes"],
        "MultipleLines"   : ["No", "Yes"],
        "InternetService" : ["Fiber optic", "DSL"],
        "OnlineSecurity"  : ["No", "Yes"],
        "OnlineBackup"    : ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport"     : ["No", "No"],
        "StreamingTV"     : ["Yes", "No"],
        "StreamingMovies" : ["Yes", "No"],
        "Contract"        : ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod"   : ["Electronic check", "Mailed check"],
        "MonthlyCharges"  : [70.35, 50.10],
        "TotalCharges"    : ["844.20", "1803.60"],   # strings, like raw CSV
        "Churn"           : ["Yes", "No"],
    })


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    from preprocessing.cleaning import CleaningTransformer
    from preprocessing.encoding import BinaryEncoder, NominalEncoder
    from preprocessing.transformers import RedundantDummyCollapser

    df = CleaningTransformer().fit_transform(raw_df)
    df = BinaryEncoder().fit_transform(df)
    # Encode target separately (not in X)
    return df
