# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import (
    extract_time_features,
    extract_age,
    extract_amount_features,
    extract_city_pop,
    drop_identity_columns,
)


@pytest.fixture(scope="module")
def preprocessed_pair(raw_train_df, raw_test_df):
    """Run core feature engineering steps and return (train, test)."""
    train = raw_train_df.copy()
    test  = raw_test_df.copy()

    train, test = drop_identity_columns(train, test)
    train, test = extract_time_features(train, test)
    train, test = extract_age(train, test)
    train, test, _ = extract_amount_features(train, test)
    train, test = extract_city_pop(train, test)

    return train, test


class TestTimeFeatures:

    def test_hour_range(self, preprocessed_pair):
        """Hour must be between 0 and 23."""
        train, _ = preprocessed_pair
        assert train["hour"].between(0, 23).all(), "hour out of range [0, 23]"

    def test_day_of_week_range(self, preprocessed_pair):
        """day_of_week must be between 0 and 6."""
        train, _ = preprocessed_pair
        assert train["day_of_week"].between(0, 6).all(), "day_of_week out of range [0, 6]"

    def test_month_range(self, preprocessed_pair):
        """month must be between 1 and 12."""
        train, _ = preprocessed_pair
        assert train["month"].between(1, 12).all(), "month out of range [1, 12]"

    def test_is_weekend_binary(self, preprocessed_pair):
        """is_weekend must be 0 or 1."""
        train, _ = preprocessed_pair
        assert set(train["is_weekend"].unique()).issubset({0, 1})

    def test_is_night_binary(self, preprocessed_pair):
        """is_night must be 0 or 1."""
        train, _ = preprocessed_pair
        assert set(train["is_night"].unique()).issubset({0, 1})


class TestAmountFeatures:

    def test_amt_log_non_negative(self, preprocessed_pair):
        """Log-transformed amount must be non-negative."""
        train, _ = preprocessed_pair
        assert (train["amt_log"] >= 0).all(), "amt_log contains negative values"

    def test_amt_zscore_exists(self, preprocessed_pair):
        """amt_zscore_by_category must be present."""
        train, _ = preprocessed_pair
        assert "amt_zscore_by_category" in train.columns

    def test_no_nan_in_amt_log(self, preprocessed_pair):
        """amt_log must have no NaN values."""
        train, _ = preprocessed_pair
        assert train["amt_log"].isna().sum() == 0, "amt_log contains NaN"


class TestAgeFeature:

    def test_age_positive(self, preprocessed_pair):
        """Age must be positive."""
        train, _ = preprocessed_pair
        assert (train["age"] > 0).all(), "age contains non-positive values"

    def test_age_reasonable(self, preprocessed_pair):
        """Age must be between 18 and 100."""
        train, _ = preprocessed_pair
        assert train["age"].between(0, 110).all(), "age outside reasonable range"


class TestCityPopFeature:

    def test_city_pop_log_non_negative(self, preprocessed_pair):
        """Log city population must be non-negative."""
        train, _ = preprocessed_pair
        assert (train["city_pop_log"] >= 0).all(), "city_pop_log contains negative values"

    def test_no_nan_city_pop_log(self, preprocessed_pair):
        """city_pop_log must have no NaN values."""
        train, _ = preprocessed_pair
        assert train["city_pop_log"].isna().sum() == 0
