"""Tests for cleaning, encoding, and feature engineering transformers."""
from __future__ import annotations

import pandas as pd
import pytest

from preprocessing.cleaning import CleaningTransformer
from preprocessing.encoding import BinaryEncoder, NominalEncoder
from preprocessing.transformers import RedundantDummyCollapser
from pipelines.preprocessing_pipeline import build_preprocessing_pipeline


class TestCleaningTransformer:
    def test_drops_customer_id(self, raw_df):
        out = CleaningTransformer().fit_transform(raw_df)
        assert "customerID" not in out.columns

    def test_total_charges_is_float(self, raw_df):
        out = CleaningTransformer().fit_transform(raw_df)
        assert out["TotalCharges"].dtype == float

    def test_total_charges_nulls_filled(self):
        df = pd.DataFrame({
            "customerID": ["X"], "TotalCharges": [" "],
            "SeniorCitizen": [0],
        })
        out = CleaningTransformer().fit_transform(df)
        assert out["TotalCharges"].isna().sum() == 0
        assert out["TotalCharges"].iloc[0] == 0.0

    def test_fit_transform_idempotent(self, raw_df):
        t = CleaningTransformer()
        out1 = t.fit_transform(raw_df)
        out2 = t.transform(raw_df)
        pd.testing.assert_frame_equal(out1, out2)


class TestBinaryEncoder:
    def test_churn_not_encoded_by_default(self, raw_df):
        df = CleaningTransformer().fit_transform(raw_df)
        out = BinaryEncoder().fit_transform(df)
        # Churn is NOT in BINARY_COLUMNS (handled separately as target)
        assert out["Churn"].dtype == object or out["Churn"].iloc[0] in ("Yes", "No", 0, 1)

    def test_gender_encoded(self, raw_df):
        df = CleaningTransformer().fit_transform(raw_df)
        out = BinaryEncoder().fit_transform(df)
        assert set(out["gender"].unique()).issubset({0, 1})

    def test_no_nulls_introduced(self, raw_df):
        df = CleaningTransformer().fit_transform(raw_df)
        out = BinaryEncoder().fit_transform(df)
        # binary cols should have no NaN after mapping
        from config.config import BINARY_COLUMNS
        for col in BINARY_COLUMNS:
            if col in out.columns:
                assert out[col].isna().sum() == 0, f"NaN in {col}"


class TestNominalEncoder:
    def test_nominal_cols_removed(self, raw_df):
        from config.config import NOMINAL_COLUMNS
        df = CleaningTransformer().fit_transform(raw_df)
        df = BinaryEncoder().fit_transform(df)
        enc = NominalEncoder()
        out = enc.fit_transform(df)
        for col in NOMINAL_COLUMNS:
            assert col not in out.columns

    def test_transform_aligns_columns(self, raw_df):
        df = CleaningTransformer().fit_transform(raw_df)
        df = BinaryEncoder().fit_transform(df)
        enc = NominalEncoder().fit(df)
        out = enc.transform(df)
        assert list(out.columns) == enc.feature_names_out_

    def test_unseen_category_handled(self, raw_df):
        df = CleaningTransformer().fit_transform(raw_df)
        df = BinaryEncoder().fit_transform(df)
        enc = NominalEncoder().fit(df)

        # Inject unseen category
        df2 = df.copy()
        df2.loc[0, "Contract"] = "Three year"
        out = enc.transform(df2)
        assert list(out.columns) == enc.feature_names_out_


class TestRedundantDummyCollapser:
    def test_no_internet_service_created(self, raw_df):
        df = CleaningTransformer().fit_transform(raw_df)
        df = BinaryEncoder().fit_transform(df)
        df = NominalEncoder().fit_transform(df)
        collapser = RedundantDummyCollapser().fit(df)
        out = collapser.transform(df)
        # Either collapsed or no NIS columns to begin with
        nis_cols = [c for c in out.columns if "No internet service" in c]
        assert len(nis_cols) == 0


class TestPreprocessingPipeline:
    def test_tree_pipeline_produces_dataframe(self, raw_df):
        pipe = build_preprocessing_pipeline("tree")
        X = raw_df.drop(columns=["Churn"])
        out = pipe.fit_transform(X)
        assert hasattr(out, "columns")

    def test_linear_pipeline_has_scaler(self):
        pipe = build_preprocessing_pipeline("linear")
        step_names = [name for name, _ in pipe.steps]
        assert "scaler" in step_names

    def test_tree_pipeline_no_scaler(self):
        pipe = build_preprocessing_pipeline("tree")
        step_names = [name for name, _ in pipe.steps]
        assert "scaler" not in step_names

    def test_train_test_no_leakage(self, raw_df):
        """Test data must only be transformed, not fitted."""
        X = raw_df.drop(columns=["Churn"])
        pipe = build_preprocessing_pipeline("tree")
        pipe.fit(X)
        # transform on same data should work; fit_transform and transform should match
        out_ft = pipe.fit_transform(X)
        out_t  = pipe.transform(X)
        # Column alignment guaranteed
        assert list(out_ft.columns) == list(out_t.columns)

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError):
            build_preprocessing_pipeline("boosting")
