# tests/conftest.py
import os
import pytest
import numpy as np
import pandas as pd

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

TRAIN_PATH = os.path.join(DATA_DIR, "fraudTrain.csv")
TEST_PATH  = os.path.join(DATA_DIR, "fraudTest.csv")

HAS_REAL_DATA   = os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)
HAS_REAL_MODELS = os.path.exists(os.path.join(MODELS_DIR, "model.txt"))


def make_mock_df(n=100) -> pd.DataFrame:
    """
    Minimal mock DataFrame that matches the real schema.
    Used when real data files aren't available (e.g. GitHub Actions).
    """
    np.random.seed(42)
    base_time = pd.Timestamp("2020-06-21 12:00:00")
    return pd.DataFrame({
        "trans_date_trans_time": [str(base_time + pd.Timedelta(minutes=i)) for i in range(n)],
        "cc_num":      np.random.randint(1e15, 9e15, n),
        "merchant":    np.random.choice(["fraud_Shop", "fraud_Mart", "fraud_Store"], n),
        "category":    np.random.choice(["grocery_pos", "misc_net", "shopping_net"], n),
        "amt":         np.random.uniform(1.0, 1000.0, n).round(2),
        "first":       ["John"] * n,
        "last":        ["Doe"] * n,
        "gender":      np.random.choice(["M", "F"], n),
        "street":      ["123 Main St"] * n,
        "city":        ["Atlanta"] * n,
        "state":       np.random.choice(["GA", "CA", "TX"], n),
        "zip":         np.random.randint(10000, 99999, n),
        "lat":         np.random.uniform(30.0, 45.0, n).round(4),
        "long":        np.random.uniform(-100.0, -70.0, n).round(4),
        "city_pop":    np.random.randint(100, 1000000, n),
        "job":         np.random.choice(["Engineer", "Teacher", "Doctor"], n),
        "dob":         ["1985-03-22"] * n,
        "trans_num":   [f"tx_{i:06d}" for i in range(n)],
        "unix_time":   np.random.randint(1370000000, 1610000000, n),
        "merch_lat":   np.random.uniform(30.0, 45.0, n).round(4),
        "merch_long":  np.random.uniform(-100.0, -70.0, n).round(4),
        "is_fraud":    np.random.choice([0, 1], n, p=[0.994, 0.006]),
    })


@pytest.fixture(scope="session")
def raw_train_df():
    """Real train data if available, otherwise mock."""
    if HAS_REAL_DATA:
        return pd.read_csv(TRAIN_PATH, nrows=5000)
    return make_mock_df(500)


@pytest.fixture(scope="session")
def raw_test_df():
    """Real test data if available, otherwise mock."""
    if HAS_REAL_DATA:
        return pd.read_csv(TEST_PATH, nrows=1000)
    return make_mock_df(100)


@pytest.fixture(scope="session")
def data_source():
    """Tell tests whether they're using real or mock data."""
    return "real" if HAS_REAL_DATA else "mock"


@pytest.fixture(scope="session")
def models_available():
    return HAS_REAL_MODELS
