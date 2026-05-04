# scripts/get_test_cases.py
# Run from project root: python scripts/get_test_cases.py
# Prints real flagged transactions you can paste into the API docs

import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.append("src")

from data_loader import load_data

# ── load artifacts (no preprocessing needed) ──────────────────────────────────

model          = joblib.load("models/model.pkl")
threshold      = joblib.load("models/threshold.pkl")
train_columns  = joblib.load("models/train_columns.pkl")
encoders       = joblib.load("models/encoders.pkl")
category_stats = joblib.load("models/category_stats.pkl")

print(f"Threshold: {threshold:.4f}")

# ── load raw test data only ────────────────────────────────────────────────────

print("Loading raw test data...")
_, raw_test_df = load_data(
    train_path="data/fraudTrain.csv",
    test_path="data/fraudTest.csv"
)

# ── preprocess using saved artifacts (fast — no fitting) ──────────────────────

print("Preprocessing with saved artifacts...")
df = raw_test_df.copy()

# time features
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"]        = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df["month"]       = df["trans_date_trans_time"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

# age
df["dob"] = pd.to_datetime(df["dob"])
df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

# distance
from geopy.distance import geodesic
print("Computing distances (takes a minute)...")
df["distance_km"] = df.apply(
    lambda r: geodesic(
        (r["lat"], r["long"]),
        (r["merch_lat"], r["merch_long"])
    ).km, axis=1
)

# amount features
df["amt_log"] = np.log1p(df["amt"])
df = df.merge(category_stats, on="category", how="left")
df["amt_zscore_by_category"] = (
    (df["amt"] - df["cat_amt_mean"]) / (df["cat_amt_std"] + 1e-8)
)

# city population
df["city_pop_log"] = np.log1p(df["city_pop"])

# encode categoricals
CAT_COLS = ["merchant", "category", "gender", "state", "job"]
for col in CAT_COLS:
    le = encoders[col]
    df[col] = df[col].map(
        lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
    )

# align to training columns
X_test = df.reindex(columns=train_columns, fill_value=0)

# ── find flagged true positives ───────────────────────────────────────────────

print("Running predictions...")
probs = model.predict_proba(X_test)[:, 1]
mask  = (probs >= threshold) & (raw_test_df["is_fraud"].values == 1)

flagged_df = raw_test_df[mask].copy()
flagged_df["predicted_prob"] = probs[mask]

print(f"\nTrue positives found: {len(flagged_df)}")
print(f"Showing top 3 by confidence:\n")

top3 = flagged_df.nlargest(3, "predicted_prob")

for _, row in top3.iterrows():
    transaction = {
        "trans_date_trans_time": str(row["trans_date_trans_time"]),
        "amt":        round(float(row["amt"]), 2),
        "category":   str(row["category"]),
        "merchant":   str(row["merchant"]),
        "gender":     str(row["gender"]),
        "state":      str(row["state"]),
        "job":        str(row["job"]),
        "dob":        str(row["dob"])[:10],
        "lat":        float(row["lat"]),
        "long":       float(row["long"]),
        "merch_lat":  float(row["merch_lat"]),
        "merch_long": float(row["merch_long"]),
        "city_pop":   int(row["city_pop"]),
    }
    print(f"── Predicted fraud probability: {row['predicted_prob']:.2%} ──")
    print(json.dumps(transaction, indent=2))
    print()