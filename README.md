# fraud-detection-explainability

# Fraud Detection & Explainability

A production-grade ML pipeline that detects credit card fraud in real time and explains every prediction using SHAP — so fraud analysts know not just *what* was flagged, but *why*.

**[🚀 Live Demo](https://fraud-detection-explainability-kbmrtpbv2xf27ffxwpqmjm.streamlit.app)** &nbsp;|&nbsp; **[GitHub](https://github.com/Donna737/fraud-detection-explainability)**

---

## Project Overview

**The problem.** Credit card fraud costs the financial industry billions annually. Most deployed models return a binary flag — fraud or not fraud — with no explanation. That's not enough for a fraud analyst who needs to decide whether to block a transaction, call the cardholder, or let it through. A model that can't explain itself creates friction and erodes trust.

**The end user.** Fraud analysts at financial institutions who review flagged transactions in real time. The output of this pipeline feeds directly into their review queue.

**The data.** The [Sparkov synthetic dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — 1.85M credit card transactions with a 0.58% fraud rate. Synthetic data was deliberately chosen over real-world datasets: real fraud datasets anonymize feature values (e.g. V1–V28 in the popular Kaggle dataset) to protect cardholder privacy, which makes SHAP explanations meaningless — "V3 pushed the score up by 0.4" tells an analyst nothing actionable. Sparkov preserves real feature semantics (amount, category, location, time) so every SHAP value maps to something a human can act on.

**The output.** For every transaction: a fraud probability, a binary flag at the chosen threshold, and a ranked list of the features that drove the prediction — in human-readable terms.

**Key design decision.** Threshold selection is a business decision, not a purely technical one. The default 0.5 threshold optimizes accuracy — the wrong metric when fraud is 0.58% of transactions. This pipeline tunes the threshold on a held-out validation set to maximize precision while guaranteeing recall ≥ 90%. The cost of missing fraud (financial loss, liability) far exceeds the cost of a false alarm (a friction call to the cardholder).

---

## Architecture

```
Raw CSVs (fraudTrain.csv, fraudTest.csv)
         │
         ▼
┌─────────────────────┐
│   data_loader.py    │  Schema validation, column checks
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   preprocess.py     │  17 engineered features
│                     │  ├── Time: hour, day, weekend, night
│                     │  ├── Age: from DOB + transaction date
│                     │  ├── Distance: cardholder home → merchant (geodesic)
│                     │  ├── Amount: log + z-score within category
│                     │  ├── City population: log-transform
│                     │  └── Categoricals: LabelEncoder (train only)
│                     │
│                     │  Saves: encoders.pkl, category_stats.pkl,
│                     │         train_columns.pkl
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│     train.py        │  LightGBM + scale_pos_weight
│                     │  Train → Val → threshold selection
│                     │  Test set touched once
│                     │  MLflow experiment tracking
│                     │  Saves: model.txt, threshold.pkl
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    explain.py       │  SHAP TreeExplainer
│                     │  Global: beeswarm + bar charts
│                     │  Per-case: waterfall plots
│                     │  Fraud alert cards (terminal)
└─────────────────────┘
         │
         ├──────────────────────────┐
         ▼                          ▼
┌─────────────────┐      ┌──────────────────────┐
│    api.py       │      │  streamlit_app.py    │
│  FastAPI        │      │  4-page portfolio    │
│  POST /predict  │      │  app with live       │
│  + SHAP         │      │  predictions         │
│  Docker ready   │      │  Deployed publicly   │
└─────────────────┘      └──────────────────────┘
```

---

## Results

### Threshold Comparison (Test Set)

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| Default (0.50) | ~0.75 | ~0.92 | ~0.83 |
| Tuned (0.90) | ~0.91 | ~0.90 | ~0.90 |

### How the threshold was selected

The threshold is tuned on the **validation set** (20% of training data, stratified) — never on the test set, which is touched exactly once at final evaluation. The selection algorithm:

1. Compute the full precision-recall curve on validation predictions
2. Filter to all thresholds where recall ≥ 90% (the business floor)
3. Among those, select the threshold with highest precision

This gives the best precision the model can achieve while guaranteeing it catches at least 9 in 10 fraud cases. The 90% floor is a business parameter — it can be adjusted without retraining.

### Why LightGBM

- Handles severe class imbalance natively via `scale_pos_weight`
- `scale_pos_weight = len(negatives) / len(positives)` ≈ 170× for this dataset
- Fast training on 1.85M rows
- First-class SHAP support via `TreeExplainer` (exact, not approximate)
- No preprocessing required for ordinal features

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **pandas / numpy** | Data loading, feature engineering |
| **geopy** | Geodesic distance between cardholder home and merchant |
| **scikit-learn** | LabelEncoder, train/val/test split, metrics |
| **LightGBM** | Gradient boosting classifier |
| **SHAP** | Model explainability — global and per-transaction |
| **MLflow** | Experiment tracking — params, metrics, artifacts |
| **FastAPI** | REST API for real-time predictions |
| **Pydantic** | Request/response schema validation |
| **uvicorn** | ASGI server for FastAPI |
| **Streamlit** | Interactive portfolio app |
| **Plotly** | Interactive EDA charts |
| **Docker** | Containerization for reproducible deployment |
| **pytest** | Unit and integration tests |
| **ruff** | Linting and code quality |
| **GitHub Actions** | CI/CD — runs tests and lint on every push |

---

## Feature Engineering

| Feature | Source | Rationale |
|---------|--------|-----------|
| `hour` | Transaction datetime | Fraud spikes 10pm–5am |
| `day_of_week` | Transaction datetime | Weekend patterns differ |
| `month` | Transaction datetime | Seasonal fraud patterns |
| `is_weekend` | day_of_week ≥ 5 | Binary flag for weekend |
| `is_night` | hour ≥ 22 or ≤ 5 | Binary flag for night transactions |
| `age` | Transaction date − DOB | Older cardholders have different risk profiles |
| `distance_km` | Geodesic(home, merchant) | Strongest single signal — 800km from home is suspicious |
| `amt_log` | log(1 + amount) | Reduces skew from extreme outliers |
| `amt_zscore_by_category` | (amt − cat_mean) / cat_std | Is this amount unusual *for this category*? $500 grocery ≠ $500 electronics |
| `city_pop_log` | log(1 + city_pop) | Population ranges from hundreds to millions |
| `merchant` | LabelEncoded | Certain merchants have elevated fraud rates |
| `category` | LabelEncoded | Online categories (misc_net, shopping_net) are higher risk |
| `gender` | LabelEncoded | Demographic signal |
| `state` | LabelEncoded | Geographic fraud patterns |
| `job` | LabelEncoded | Occupation correlates with transaction patterns |

**Why z-score within category, not raw amount?**
A $500 transaction at a grocery store is anomalous. A $500 transaction at an electronics store is routine. Raw amount conflates these. The z-score captures how unusual the amount is *relative to the merchant category*, which is what actually matters for fraud detection.

**Why geodesic distance?**
Haversine gives the straight-line distance on the Earth's surface — fast to compute and sufficient for this signal. A transaction 500km from the cardholder's registered home address is suspicious regardless of the exact route.

---

## Setup & Installation

**Prerequisites:** Python 3.11, pip

```bash
# clone the repo
git clone https://github.com/Donna737/fraud-detection-explainability.git
cd fraud-detection-explainability

# create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# download data from Kaggle
# https://www.kaggle.com/datasets/kartik2112/fraud-detection
# place fraudTrain.csv and fraudTest.csv in data/
```

---

## How to Run

### Full training pipeline

```bash
python src/data_loader.py   # validate data
python src/preprocess.py    # feature engineering (~15 mins for distance)
python src/train.py         # train model + MLflow tracking
python src/explain.py       # SHAP values + waterfall plots
```

### Streamlit app (local)

```bash
streamlit run app/streamlit_app.py
# opens http://localhost:8501
```

### FastAPI (local)

```bash
uvicorn src.api:app --reload
# interactive docs at http://localhost:8000/docs
```

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2020-10-09 22:52:14",
    "amt": 845.70,
    "category": "misc_net",
    "merchant": "fraud_Jaskolski-Vandervort",
    "gender": "M",
    "state": "GA",
    "job": "Pensions consultant",
    "dob": "1944-05-14",
    "lat": 34.9298,
    "long": -84.9885,
    "merch_lat": 35.4239,
    "merch_long": -84.8912,
    "city_pop": 74
  }'
```

### Docker

```bash
docker compose up
# Streamlit app at http://localhost:8501
```

### Tests

```bash
pytest tests/ -v
# 31 tests across data quality, feature engineering, and model loading
# model tests use real artifacts; data tests fall back to mock data
# safe to run without the full dataset
```

---

## Key Decisions & Lessons

**1. Synthetic data enables meaningful explainability.**
Real fraud datasets anonymize feature values to protect privacy — SHAP on V1–V28 is uninterpretable. Sparkov's named features (amount, category, distance, time) make every SHAP value actionable. The tradeoff is important to state explicitly: **synthetic data produces artificially high metrics**. The model achieves ~90% precision and recall partly because Sparkov's fraud patterns are consistent and learnable by design — real-world fraud is messier, more adversarial, and constantly evolving. These numbers demonstrate the pipeline works correctly on this dataset. The value of this project is the end-to-end architecture — feature engineering, threshold selection, explainability, serving, and testing — not the benchmark numbers themselves.

**2. Threshold selection belongs on the validation set, not the test set.**
Searching for the optimal threshold across a grid of values is a form of optimization. Doing it on the test set inflates reported metrics. The validation set (20% of training data, stratified) is used exclusively for this — the test set is touched exactly once.

**3. The model is stored in LightGBM's native text format, not pickle.**
Pickle files embed the Python environment they were created in — loading a Mac-trained `.pkl` on Linux Docker failed with `libgomp.so.1: cannot open shared object file`. LightGBM's native `.txt` format is platform-independent and loads cleanly anywhere. This cost several hours to debug and is worth documenting explicitly.

**4. Three correlated time features outperform one.**
`hour`, `is_night`, and `is_weekend` are correlated but capture different granularities of the time signal. SHAP analysis confirmed all three appear in the top features — removing any one of them degraded recall. Correlated features are not always redundant.

**5. Distance computation is the pipeline bottleneck.**
`geopy.geodesic` applied row-by-row across 1.85M rows takes ~15 minutes. The fix was caching the computed distances to parquet after the first run. In production this would be vectorized or precomputed at ingestion time.

---

## File Structure

```
fraud-detection-explainability/
│
├── src/
│   ├── data_loader.py       # load + validate raw CSVs
│   ├── preprocess.py        # feature engineering pipeline
│   ├── train.py             # LightGBM training + threshold selection
│   ├── explain.py           # SHAP global + per-case explanations
│   └── api.py               # FastAPI /predict endpoint
│
├── app/
│   └── streamlit_app.py     # 4-page interactive portfolio app
│
├── tests/
│   ├── conftest.py          # shared fixtures with real/mock fallback
│   ├── test_data_quality.py # schema, nulls, fraud rate checks
│   ├── test_features.py     # feature range and correctness checks
│   └── test_model.py        # model loading and prediction checks
│
├── models/                  # saved artifacts (committed to repo)
│   ├── model.txt            # LightGBM native format
│   ├── threshold.pkl        # chosen operating threshold
│   ├── encoders.pkl         # fitted LabelEncoders
│   ├── category_stats.pkl   # per-category amount stats
│   └── train_columns.pkl    # feature column order
│
├── outputs/                 # generated plots (committed to repo)
│   ├── pr_curve.png
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── waterfall_case_1-4.png
│
├── notebooks/
│   └── fraud_detection.ipynb  # end-to-end walkthrough with outputs
│
├── data/
│   └── sample.csv           # 5000-row sample for Streamlit Cloud
│                            # full data: kaggle.com/datasets/kartik2112/fraud-detection
│
├── .github/
│   └── workflows/
│       └── ci.yml           # pytest + ruff on every push
│
├── Dockerfile               # python:3.11-slim + libgomp1
├── docker-compose.yml       # mounts data/ and models/ as volumes
├── requirements.txt         # pinned dependencies
├── runtime.txt              # Python 3.11 for Streamlit Cloud
└── README.md
```

---

## Live Demo

**[Open the app →](https://fraud-detection-explainability-kbmrtpbv2xf27ffxwpqmjm.streamlit.app)**

The default transaction on page 4 is a real fraud case from the test set (100% model confidence). Try changing the hour from 22 to 10, or the amount from $845 to $20, to see the score drop and the feature contributions shift in real time.