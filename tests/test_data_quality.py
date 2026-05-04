# tests/test_data_quality.py
import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import EXPECTED_COLUMNS, EXPECTED_SCHEMA, _validate


class TestDataQuality:

    def test_expected_columns_present(self, raw_train_df):
        """All required columns must exist in the dataset."""
        missing = [c for c in EXPECTED_COLUMNS if c not in raw_train_df.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_no_empty_dataframe(self, raw_train_df):
        """Dataset must not be empty."""
        assert len(raw_train_df) > 0, "DataFrame is empty"

    def test_target_column_binary(self, raw_train_df):
        """is_fraud must only contain 0 and 1."""
        unique_vals = set(raw_train_df["is_fraud"].unique())
        assert unique_vals.issubset({0, 1}), f"Unexpected values in is_fraud: {unique_vals}"

    def test_amount_positive(self, raw_train_df):
        """Transaction amounts must be positive."""
        assert (raw_train_df["amt"] > 0).all(), "Found non-positive transaction amounts"

    def test_amt_dtype(self, raw_train_df):
        """Amount column must be float."""
        assert raw_train_df["amt"].dtype == "float64", \
            f"amt dtype is {raw_train_df['amt'].dtype}, expected float64"

    def test_validation_passes_on_good_data(self, raw_train_df):
        """_validate should not raise on valid data."""
        try:
            _validate(raw_train_df, path="mock_path.csv")
        except ValueError as e:
            pytest.fail(f"_validate raised ValueError on good data: {e}")

    def test_validation_catches_missing_columns(self):
        """_validate should raise ValueError when columns are missing."""
        bad_df = pd.DataFrame({"amt": [1.0, 2.0], "is_fraud": [0, 1]})
        with pytest.raises(ValueError, match="Missing columns"):
            _validate(bad_df, path="bad.csv")

    def test_validation_catches_empty_dataframe(self):
        """_validate should raise ValueError for empty DataFrames."""
        empty_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        with pytest.raises(ValueError, match="empty"):
            _validate(empty_df, path="empty.csv")

    def test_fraud_rate_reasonable(self, raw_train_df, data_source):
        """Fraud rate should be between 0.1% and 10%."""
        fraud_rate = raw_train_df["is_fraud"].mean()
        assert 0.001 <= fraud_rate <= 0.10, \
            f"Fraud rate {fraud_rate:.4f} outside expected range (data source: {data_source})"

    def test_no_null_in_critical_columns(self, raw_train_df):
        """Critical columns must have no nulls."""
        critical = ["amt", "is_fraud", "category", "merchant", "gender"]
        for col in critical:
            null_count = raw_train_df[col].isna().sum()
            assert null_count == 0, f"Column '{col}' has {null_count} null values"
