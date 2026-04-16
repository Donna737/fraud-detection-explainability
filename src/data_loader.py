# src/data_loader.py
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category",
    "amt", "first", "last", "gender", "street", "city", "state",
    "zip", "lat", "long", "city_pop", "job", "dob",
    "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud"
]

def load_data(train_path: str, test_path: str):
    """
    Load raw train and test CSVs from disk.
    Returns two DataFrames: (train_df, test_df)
    """
    logger.info("Loading training data...")
    train_df = _load_single(train_path)

    logger.info("Loading test data...")
    test_df = _load_single(test_path)

    logger.info(f"Train: {train_df.shape} | Test: {test_df.shape}")
    return train_df, test_df


def _load_single(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    _validate(df, path)
    return df


def _validate(df: pd.DataFrame, path: str):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    if df.empty:
        raise ValueError(f"File is empty: {path}")
    logger.info(f"Validation passed for {path}")


if __name__ == "__main__":
    train_df, test_df = load_data(
        train_path="data/fraudTrain.csv",
        test_path="data/fraudTest.csv"
    )
    print(train_df.head())