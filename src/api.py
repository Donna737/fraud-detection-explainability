# src/api.py
import logging
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with SHAP explanations.",
    version="1.0.0",
)

# ── Load artifacts at startup ──────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


model = lgb.Booster(model_file=f"{MODELS_DIR}/model.txt")
threshold      = joblib.load(f"{MODELS_DIR}/threshold.pkl")
encoders       = joblib.load(f"{MODELS_DIR}/encoders.pkl")
category_stats = joblib.load(f"{MODELS_DIR}/category_stats.pkl")
train_columns  = joblib.load(f"{MODELS_DIR}/train_columns.pkl")
explainer      = shap.TreeExplainer(model)

logger.info("All artifacts loaded.")

# ── Request / Response schemas ─────────────────────────────────────────────────

class Transaction(BaseModel):
    trans_date_trans_time: str   = Field(..., example="2020-10-09 22:52:14")
    amt:                   float = Field(..., example=845.70)
    category:              str   = Field(..., example="misc_net")
    merchant:              str   = Field(..., example="fraud_Jaskolski-Vandervort")
    gender:                str   = Field(..., example="M")
    state:                 str   = Field(..., example="GA")
    job:                   str   = Field(..., example="Pensions consultant")
    dob:                   str   = Field(..., example="1944-05-14")
    lat:                   float = Field(..., example=34.9298)
    long:                  float = Field(..., example=-84.9885)
    merch_lat:             float = Field(..., example=35.423928)
    merch_long:            float = Field(..., example=-84.891228)
    city_pop:              int   = Field(..., example=74)


class FeatureContribution(BaseModel):
    feature:     str
    contribution: float
    direction:   str


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    top_features:      list[FeatureContribution]


# ── Preprocessing (mirrors preprocess.py — single transaction) ─────────────────

def preprocess_transaction(t: Transaction) -> pd.DataFrame:
    df = pd.DataFrame([t.model_dump()])

    # ── time features ──
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"]       = df["trans_date_trans_time"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # ── age ──
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # ── distance ──
    from geopy.distance import geodesic
    df["distance_km"] = df.apply(
        lambda r: geodesic(
            (r["lat"], r["long"]),
            (r["merch_lat"], r["merch_long"])
        ).km, axis=1
    )

    # ── amount features ──
    df["amt_log"] = np.log1p(df["amt"])
    df = df.merge(category_stats, on="category", how="left")
    df["amt_zscore_by_category"] = (
        (df["amt"] - df["cat_amt_mean"]) / (df["cat_amt_std"] + 1e-8)
    )

    # ── city population ──
    df["city_pop_log"] = np.log1p(df["city_pop"])

    # ── encode categoricals ──
    CAT_COLS = ["merchant", "category", "gender", "state", "job"]
    for col in CAT_COLS:
        le = encoders[col]
        df[col] = df[col].map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
        )

    # ── align to training columns ──
    df = df.reindex(columns=train_columns, fill_value=0)

    return df


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        X = preprocess_transaction(transaction)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # ── prediction ──
    prob      = float(model.predict(X)[0])
    is_fraud = prob >= threshold

    # ── SHAP explanation ──
    shap_values = explainer.shap_values(X)

    # handle LightGBM returning list [class0, class1]
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    if len(sv.shape) > 1:
        sv = sv[0]

    # normalize to percentage contribution
    sv_abs_total = sum(abs(v) for v in sv) or 1
    feature_contributions = sorted(
        [
            FeatureContribution(
                feature     = col,
                contribution= round(float(abs(val) / sv_abs_total), 4),
                direction   = "toward fraud" if val > 0 else "away from fraud"
            )
            for col, val in zip(train_columns, sv)
        ],
        key=lambda x: x.contribution,
        reverse=True
    )

    return PredictionResponse(
        fraud_probability = round(prob, 4),
        is_fraud          = bool(is_fraud),
        top_features      = feature_contributions[:5],
    )