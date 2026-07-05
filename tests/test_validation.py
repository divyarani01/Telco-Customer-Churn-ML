"""Tests for Great Expectations validation rules."""
from __future__ import annotations

import pandas as pd
import pytest


def _make_valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID"      : ["1234-ABCD"],
        "gender"          : ["Male"],
        "SeniorCitizen"   : [0],
        "Partner"         : ["Yes"],
        "Dependents"      : ["No"],
        "tenure"          : [12],
        "PhoneService"    : ["Yes"],
        "MultipleLines"   : ["No"],
        "InternetService" : ["Fiber optic"],
        "OnlineSecurity"  : ["No"],
        "OnlineBackup"    : ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport"     : ["No"],
        "StreamingTV"     : ["Yes"],
        "StreamingMovies" : ["Yes"],
        "Contract"        : ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod"   : ["Electronic check"],
        "MonthlyCharges"  : [70.35],
        "TotalCharges"    : [844.20],
        "Churn"           : ["Yes"],
    })


class TestValidation:
    def test_valid_dataframe_passes(self):
        from validation.gx_validation import validate
        df = _make_valid_df()
        assert validate(df) is True

    def test_bad_gender_fails(self):
        from validation.gx_validation import validate
        df = _make_valid_df()
        df["gender"] = "Unknown"
        assert validate(df) is False

    def test_negative_tenure_fails(self):
        from validation.gx_validation import validate
        df = _make_valid_df()
        df["tenure"] = -1
        assert validate(df) is False

    def test_null_customer_id_fails(self):
        from validation.gx_validation import validate
        df = _make_valid_df()
        df["customerID"] = None
        assert validate(df) is False
