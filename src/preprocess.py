# src/preprocess.py
import logging
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# columns that are pure identifiers — no predictive value, would cause leakage
DROP_IDENTITY_COLS = [
    "Unnamed: 0", "cc_num", "first", "last",
    "street", "trans_num", "unix_time"
]

# columns we drop after feature engineering (replaced by better features)
DROP_AFTER_ENGINEERING = [
    "trans_date_trans_time", "dob",
    "lat", "long", "merch_lat", "merch_long",  # replaced by distance_km
    "zip",                                       # too granular, state covers it
    "cat_amt_mean", "cat_amt_std",               # intermediate, not a feature
    "city_pop",                                  # replaced by city_pop_log
    "city",                                      # too many unique values
]

# columns we encode (keep originals for explaination)
CAT_COLS = ["merchant", "category", "gender", "state", "job"]

# target + columns to exclude from feature matrix
TARGET = "is_fraud"


# ── Individual steps ───────────────────────────────────────────────────────────

def drop_identity_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Drop columns that are identifiers or would cause data leakage."""
    logger.info("Dropping identity columns...")
    train_df = train_df.drop(columns=DROP_IDENTITY_COLS, errors="ignore")
    test_df = test_df.drop(columns=DROP_IDENTITY_COLS, errors="ignore")
    return train_df, test_df


def save_original_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save readable string versions of categorical columns before encoding.
    We'll need these later for the explanation prompts.
    """
    logger.info("Saving original categorical columns...")
    for df in [train_df, test_df]:
        for col in CAT_COLS:
            df[f"{col}_original"] = df[col]
    return train_df, test_df


def extract_time_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Parse datetime and extract hour, day, month, weekend, night flags."""
    logger.info("Extracting time features...")
    for df in [train_df, test_df]:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["hour"]        = df["trans_date_trans_time"].dt.hour
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # 0=Mon, 6=Sun
        df["month"]       = df["trans_date_trans_time"].dt.month
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    return train_df, test_df


def extract_age(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Compute cardholder age at time of transaction."""
    logger.info("Extracting age feature...")
    for df in [train_df, test_df]:
        df["dob"] = pd.to_datetime(df["dob"])
        df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    return train_df, test_df


def _compute_distance(row) -> float:
    """Geodesic distance in km between cardholder home and merchant location."""
    try:
        return geodesic(
            (row["lat"], row["long"]),
            (row["merch_lat"], row["merch_long"])
        ).km
    except Exception:
        return np.nan


def extract_distance(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Distance between cardholder home and merchant.
    One of the strongest fraud signals — a transaction 800km from home is suspicious.
    Note: this is slow (~few minutes) because it runs row by row.
    """
    logger.info("Computing distance features (this takes a few minutes)...")
    train_df["distance_km"] = train_df.apply(_compute_distance, axis=1)
    test_df["distance_km"]  = test_df.apply(_compute_distance, axis=1)
    logger.info("Distance feature extracted.")
    return train_df, test_df


def extract_amount_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Log-transform amount and compute z-score within merchant category.
    A $500 grocery charge vs $500 electronics charge have very different
    fraud implications — the z-score captures this.
    """
    logger.info("Extracting amount features...")

    for df in [train_df, test_df]:
        df["amt_log"] = np.log1p(df["amt"])

    # fit category stats on TRAIN only — no leakage
    category_stats = (
        train_df.groupby("category")["amt"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "cat_amt_mean", "std": "cat_amt_std"})
    )

    train_df = train_df.merge(category_stats, on="category", how="left")
    test_df  = test_df.merge(category_stats, on="category", how="left")

    for df in [train_df, test_df]:
        df["amt_zscore_by_category"] = (
            (df["amt"] - df["cat_amt_mean"]) / (df["cat_amt_std"] + 1e-8)
        )

    return train_df, test_df


def extract_city_pop(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Log-transform city population — ranges wildly from hundreds to millions."""
    logger.info("Extracting city population feature...")
    for df in [train_df, test_df]:
        df["city_pop_log"] = np.log1p(df["city_pop"])
    return train_df, test_df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Label-encode categorical columns.
    Fit encoders on train only to avoid data leakage.
    Returns encoders dict so we can reuse them in the API.
    """
    logger.info("Encoding categorical columns...")
    encoders = {}

    for col in CAT_COLS:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])

        # handle unseen labels in test set gracefully
        test_df[col] = test_df[col].map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le

    logger.info(f"Encoded columns: {CAT_COLS}")
    return train_df, test_df, encoders


def drop_intermediate_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Drop columns replaced by engineered features."""
    logger.info("Dropping intermediate columns...")
    train_df = train_df.drop(columns=DROP_AFTER_ENGINEERING, errors="ignore")
    test_df  = test_df.drop(columns=DROP_AFTER_ENGINEERING, errors="ignore")
    return train_df, test_df


def build_feature_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Split into X (features) and y (target).
    Excludes target, raw amount (replaced by amt_log), and original string columns.
    """
    logger.info("Building feature matrix...")

    exclude = [TARGET, "amt"] + [f"{col}_original" for col in CAT_COLS]
    features = [col for col in train_df.columns if col not in exclude]

    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_test  = test_df[features]
    y_test  = test_df[TARGET]

    logger.info(f"Features ({len(features)}): {features}")
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Fraud rate — train: {y_train.mean():.4f} | test: {y_test.mean():.4f}")

    return X_train, y_train, X_test, y_test, features


# ── Main pipeline function ─────────────────────────────────────────────────────

def run_preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Run the full preprocessing pipeline.

    Returns:
        X_train, y_train, X_test, y_test  — feature matrices and targets
        encoders                           — fitted LabelEncoders (needed for API)
        test_df                            — test df with original string columns 
        features                           — list of feature names
    """
    logger.info("=== Starting preprocessing pipeline ===")

    train_df, test_df = drop_identity_columns(train_df, test_df)
    train_df, test_df = save_original_categoricals(train_df, test_df)
    train_df, test_df = extract_time_features(train_df, test_df)
    train_df, test_df = extract_age(train_df, test_df)
    train_df, test_df = extract_distance(train_df, test_df)
    train_df, test_df = extract_amount_features(train_df, test_df)
    train_df, test_df = extract_city_pop(train_df, test_df)
    train_df, test_df, encoders = encode_categoricals(train_df, test_df)
    train_df, test_df = drop_intermediate_columns(train_df, test_df)

    X_train, y_train, X_test, y_test, features = build_feature_matrix(train_df, test_df)

    logger.info("=== Preprocessing complete ===")
    return X_train, y_train, X_test, y_test, encoders, test_df, features


# ── Run standalone ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_data

    train_df, test_df = load_data(
        train_path="data/fraudTrain.csv",
        test_path="data/fraudTest.csv"
    )

    X_train, y_train, X_test, y_test, encoders, test_df, features = run_preprocessing(
        train_df, test_df
    )

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"\nFeatures: {features}")