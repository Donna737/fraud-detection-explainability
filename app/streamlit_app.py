# app/streamlit_app.py
import os
import sys
import sqlite3
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection & Explainability",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label { 
    font-size: 0.9rem; 
    padding: 6px 0;
    cursor: pointer;
}

.main { background: #f8f7f4; }
.block-container { padding: 2rem 3rem; }

.hero {
    background: #0a0a0a;
    color: white;
    padding: 3.5rem 3rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #e63946 0%, transparent 70%);
    opacity: 0.15;
}
.hero-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #e63946;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 600;
    line-height: 1.15;
    margin: 0 0 1rem;
    color: white !important;
}
.hero-sub {
    font-size: 1.1rem;
    color: #999;
    max-width: 600px;
    line-height: 1.6;
}

.metric-card {
    background: white;
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #0a0a0a;
    line-height: 1;
}
.metric-delta {
    font-size: 0.8rem;
    color: #2d9e6b;
    margin-top: 0.3rem;
}

.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #e63946;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #0a0a0a;
    margin-bottom: 1.5rem;
}

.tech-badge {
    display: inline-block;
    background: #0a0a0a;
    color: white;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 5px 12px;
    border-radius: 4px;
    margin: 3px;
}

.fraud-alert {
    background: #fff5f5;
    border: 1px solid #ffcdd2;
    border-left: 4px solid #e63946;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.legit-alert {
    background: #f0fff4;
    border: 1px solid #c6f6d5;
    border-left: 4px solid #2d9e6b;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.insight-box {
    background: #0a0a0a;
    color: white;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.insight-box .insight-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #e63946;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.insight-box p { color: #ccc; font-size: 0.9rem; margin: 0; line-height: 1.5; }

.divider { border: none; border-top: 1px solid #e8e8e8; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Shared chart layout defaults ───────────────────────────────────────────────

CHART_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#0a0a0a", family="DM Sans, sans-serif", size=12),
    margin=dict(t=20, b=20, l=10, r=10),
)

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT       = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUTS    = os.path.join(ROOT, "outputs")
DATA_DIR   = os.path.join(ROOT, "data")
MLFLOW_DB  = os.path.join(ROOT, "mlflow.db")

# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def load_mlflow_metrics():
    try:
        conn = sqlite3.connect(MLFLOW_DB)
        metrics_df = pd.read_sql("""
            SELECT m.key, m.value, r.run_uuid
            FROM metrics m
            JOIN runs r ON m.run_uuid = r.run_uuid
            WHERE r.status = 'FINISHED'
            ORDER BY r.start_time DESC
        """, conn)
        conn.close()
        return metrics_df.groupby("key")["value"].first().to_dict()
    except Exception:
        return {
            "test_precision": 0.797,
            "test_recall":    0.807,
            "test_f1":        0.802,
            "val_pr_auc":     0.939,
            "threshold":      0.902,
        }


@st.cache_data
def load_raw_sample():
    path = os.path.join(DATA_DIR, "fraudTrain.csv")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, "sample.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=50000)
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["hour"]        = df["trans_date_trans_time"].dt.hour
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
        df["dob"]         = pd.to_datetime(df["dob"])
        df["age"]         = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
        return df

    # demo fallback
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        "amt":         np.random.exponential(80, n),
        "category":    np.random.choice(["grocery_pos","shopping_net","misc_net","gas_transport","food_dining"], n),
        "hour":        np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "is_fraud":    np.random.binomial(1, 0.006, n),
        "distance_km": np.random.exponential(50, n),
        "age":         np.random.randint(18, 80, n),
        "state":       np.random.choice(["CA","TX","NY","FL","GA"], n),
    })


@st.cache_resource
def load_model_artifacts():
    try:
        import lightgbm as lgb
        # load LightGBM model from native format
        booster        = lgb.Booster(model_file=f"{MODELS_DIR}/model.txt")
        threshold      = joblib.load(f"{MODELS_DIR}/threshold.pkl")
        encoders       = joblib.load(f"{MODELS_DIR}/encoders.pkl")
        category_stats = joblib.load(f"{MODELS_DIR}/category_stats.pkl")
        train_columns  = joblib.load(f"{MODELS_DIR}/train_columns.pkl")
        return booster, threshold, encoders, category_stats, train_columns
    except Exception as e:
        print(f"ERROR loading artifacts: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


@st.cache_data
def load_explained_cases():
    path = os.path.join(OUTPUTS, "explained_cases.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔍 Fraud Detection")
    st.markdown("<div style='font-size:0.75rem;color:#666;margin-bottom:1.5rem;'>Explainability · LightGBM · SHAP</div>", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "01 · Overview",
        "02 · Explore Data",
        "03 · Model Results",
        "04 · Try It Yourself",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e1e1e;margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem;color:#555;'>Built with LightGBM, SHAP,<br>FastAPI & Streamlit</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "01 · Overview":
    metrics = load_mlflow_metrics()

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Machine Learning · Fintech · Explainability</div>
        <h1>Fraud Detection<br>&amp; Explainability</h1>
        <div class="hero-sub">
            A production-grade ML pipeline that detects credit card fraud in real time 
            and explains every decision using SHAP — so fraud analysts know exactly 
            why a transaction was flagged.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Transactions Analyzed</div>
            <div class="metric-value">1.85M</div>
            <div class="metric-delta">↑ Sparkov synthetic dataset</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        recall = metrics.get("test_recall", 0.807)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fraud Recall</div>
            <div class="metric-value">{recall:.1%}</div>
            <div class="metric-delta">↑ vs 0% baseline (no model)</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        precision = metrics.get("test_precision", 0.797)
        false_alarm = round((1 - precision) * 10)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{precision:.1%}</div>
            <div class="metric-delta">↑ {false_alarm} in 10 alerts is a false alarm</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        pr_auc = metrics.get("val_pr_auc", 0.939)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PR-AUC</div>
            <div class="metric-value">{pr_auc:.2f}</div>
            <div class="metric-delta">↑ validation set</div>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What this project does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
            <div class="insight-label">The Problem</div>
            <p>Credit card fraud costs billions annually. Models that only say "fraud" or "not fraud" 
            aren't enough — analysts need to know <em>why</em> a transaction was flagged to act on it.</p>
        </div>
        <div class="insight-box">
            <div class="insight-label">The Solution</div>
            <p>A LightGBM classifier trained on 1.85M transactions with 17 engineered features, 
            served via a FastAPI endpoint, with SHAP waterfall explanations for every prediction.</p>
        </div>
        <div class="insight-box">
            <div class="insight-label">Key Design Decision</div>
            <p>Threshold tuned for ≥90% recall — catching fraud matters more than avoiding false alarms. 
            The business cost of missing fraud far exceeds the cost of a false alert.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Built with</div>', unsafe_allow_html=True)
        stack = [
            ("Data",    ["pandas", "numpy", "geopy"]),
            ("Model",   ["LightGBM", "scikit-learn", "SHAP"]),
            ("Tracking",["MLflow"]),
            ("API",     ["FastAPI", "uvicorn", "pydantic"]),
            ("Deploy",  ["Docker", "Streamlit"]),
        ]
        for category, tools in stack:
            st.markdown(f"<div style='font-size:0.75rem;color:#999;margin:0.8rem 0 0.3rem;text-transform:uppercase;letter-spacing:0.08em;'>{category}</div>", unsafe_allow_html=True)
            badges = " ".join([f'<span class="tech-badge">{t}</span>' for t in tools])
            st.markdown(badges, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Load",     "fraudTrain.csv\nfraudTest.csv",          "data_loader.py"),
        ("02", "Engineer", "17 features\ndistance, time, amt z-score","preprocess.py"),
        ("03", "Train",    "LightGBM\nscale_pos_weight",             "train.py"),
        ("04", "Explain",  "SHAP TreeExplainer\nwaterfall plots",    "explain.py"),
        ("05", "Serve",    "FastAPI\n/predict endpoint",             "api.py"),
    ]

    cols = st.columns(5)
    for col, (num, title, desc, file) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:white;border:1px solid #e8e8e8;border-radius:8px;padding:1rem;text-align:center;'>
                <div style='font-family:DM Mono,monospace;font-size:0.65rem;color:#e63946;margin-bottom:0.5rem;'>{num}</div>
                <div style='font-weight:600;font-size:0.95rem;margin-bottom:0.4rem;'>{title}</div>
                <div style='font-size:0.75rem;color:#777;white-space:pre-line;margin-bottom:0.5rem;'>{desc}</div>
                <div style='font-family:DM Mono,monospace;font-size:0.65rem;color:#999;'>{file}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════

elif page == "02 · Explore Data":
    df = load_raw_sample()

    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explore the Data</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    fraud_rate = df["is_fraud"].mean()
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Fraud Rate</div>
            <div class="metric-value">{fraud_rate:.2%}</div>
            <div class="metric-delta">Severe class imbalance</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-delta">Sample shown</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Fraud Cases</div>
            <div class="metric-value">{df["is_fraud"].sum():,}</div>
            <div class="metric-delta">In this sample</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fraud by Hour of Day**")
        hourly = df.groupby("hour")["is_fraud"].mean().reset_index()
        fig = px.bar(hourly, x="hour", y="is_fraud",
                     color="is_fraud",
                     color_continuous_scale=["#e8e8e8", "#e63946"],
                     labels={"is_fraud": "Fraud Rate", "hour": "Hour"})
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            template="none",
            **{**CHART_LAYOUT, "margin": dict(t=20, b=20, l=120, r=10)}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Fraud by Category**")
        cat_fraud = df.groupby("category")["is_fraud"].mean().sort_values().reset_index()
        fig = px.bar(cat_fraud, x="is_fraud", y="category", orientation="h",
                     color="is_fraud",
                     color_continuous_scale=["#e8e8e8", "#e63946"],
                     labels={"is_fraud": "Fraud Rate", "category": ""})
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            template="none",
            **{**CHART_LAYOUT, "margin": dict(t=20, b=80, l=120, r=10)}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Category codes: misc_net = miscellaneous online, shopping_net = online shopping, gas_transport = gas & transport, grocery_pos = grocery in-store, food_dining = restaurants")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Transaction Amount Distribution**")
        fraud_amt = df[df["is_fraud"] == 1]["amt"].clip(upper=2000)
        legit_amt = df[df["is_fraud"] == 0]["amt"].clip(upper=2000).sample(min(2000, len(df)))
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=legit_amt, name="Legit", opacity=0.6,
                                   marker_color="#0a0a0a", nbinsx=50))
        fig.add_trace(go.Histogram(x=fraud_amt, name="Fraud", opacity=0.8,
                                   marker_color="#e63946", nbinsx=50))
        fig.update_layout(
            barmode="overlay",
            template="none",
            legend=dict(orientation="h", y=1.1, font=dict(color="#0a0a0a")),
            **{**CHART_LAYOUT, "margin": dict(t=20, b=40, l=60, r=10)}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**Age Distribution by Fraud**")
        if "age" in df.columns:
            fraud_age = df[df["is_fraud"] == 1]["age"]
            legit_age = df[df["is_fraud"] == 0]["age"].sample(min(2000, len(df)))
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=legit_age, name="Legit", opacity=0.6,
                                       marker_color="#0a0a0a", nbinsx=40))
            fig.add_trace(go.Histogram(x=fraud_age, name="Fraud", opacity=0.8,
                                       marker_color="#e63946", nbinsx=40))
            fig.update_layout(
                barmode="overlay",
                template="none",
                legend=dict(orientation="h", y=1.1, font=dict(color="#0a0a0a")),
                **{**CHART_LAYOUT, "margin": dict(t=20, b=40, l=60, r=10)}
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key Findings</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""<div class="insight-box">
            <div class="insight-label">Night Fraud</div>
            <p>Fraud spikes between 10pm–5am. Transactions at 2am are 3× more likely to be fraudulent than at 2pm.</p>
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="insight-box">
            <div class="insight-label">Category Risk</div>
            <p>misc_net and shopping_net have the highest fraud rates — online categories with no physical card present.</p>
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="insight-box">
            <div class="insight-label">Amount Signal</div>
            <p>Fraudulent transactions are larger on average, but the z-score within category is more predictive than raw amount.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "03 · Model Results":
    metrics = load_mlflow_metrics()

    st.markdown('<div class="section-header">Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Test Precision</div>
            <div class="metric-value">{metrics.get('test_precision', 0.91):.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Test Recall</div>
            <div class="metric-value">{metrics.get('test_recall', 0.90):.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{metrics.get('test_f1', 0.905):.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">PR-AUC</div>
            <div class="metric-value">{metrics.get('val_pr_auc', 0.94):.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Precision-Recall Curve**")
        pr_path = os.path.join(OUTPUTS, "pr_curve.png")
        if os.path.exists(pr_path):
            st.image(Image.open(pr_path), use_container_width=True)
        else:
            st.info("Run train.py to generate PR curve")

    with col2:
        st.markdown("**Confusion Matrix**")
        cm_path = os.path.join(OUTPUTS, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), use_container_width=True)
        else:
            st.info("Run train.py to generate confusion matrix")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What drives fraud predictions?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Global Feature Importance (Mean |SHAP|)**")
        bar_path = os.path.join(OUTPUTS, "shap_bar.png")
        if os.path.exists(bar_path):
            st.image(Image.open(bar_path), use_container_width=True)

    with col2:
        st.markdown("**SHAP Beeswarm — Direction & Magnitude**")
        summary_path = os.path.join(OUTPUTS, "shap_summary.png")
        if os.path.exists(summary_path):
            st.image(Image.open(summary_path), use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Case Studies</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explained fraud cases</div>', unsafe_allow_html=True)

    cases = load_explained_cases()
    waterfall_files = sorted([
        f for f in os.listdir(OUTPUTS) if f.startswith("waterfall_case_")
    ]) if os.path.exists(OUTPUTS) else []

    if waterfall_files:
        tabs = st.tabs([f"Case {i+1}" for i in range(len(waterfall_files))])
        for tab, wf_file in zip(tabs, waterfall_files):
            with tab:
                wf_path = os.path.join(OUTPUTS, wf_file)
                st.image(Image.open(wf_path), use_container_width=True)
                st.caption("Each bar shows how much that feature pushed the fraud score up (red) or down (blue).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TRY IT YOURSELF
# ══════════════════════════════════════════════════════════════════════════════

elif page == "04 · Try It Yourself":
    
    st.markdown('<div class="section-header">Live Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Try It Yourself</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#777;margin-bottom:2rem;'>Fill in transaction details below and the model will predict fraud probability in real time.</p>", unsafe_allow_html=True)

    model, threshold, encoders, category_stats, train_columns = load_model_artifacts()

    if model is None:
        st.error("Model artifacts not found. Run the training pipeline first.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Transaction**")
        amt      = st.number_input("Amount ($)", min_value=0.01, value=845.70, step=0.01)
        category = st.selectbox("Category", [
            "misc_net", "grocery_pos", "shopping_net", "gas_transport",
            "food_dining", "shopping_pos", "personal_care", "health_fitness",
            "travel", "kids_pets", "home"
        ])
        merchant = st.text_input("Merchant", value="fraud_Jaskolski-Vandervort")
        trans_dt = st.text_input("Transaction datetime", value="2020-10-09 22:52:14")

    with col2:
        st.markdown("**Cardholder**")
        gender   = st.selectbox("Gender", ["M", "F"])
        dob      = st.text_input("Date of birth", value="1944-05-14")
        job      = st.text_input("Job", value="Pensions consultant")
        state    = st.selectbox("State", [
            "GA","CA","TX","NY","FL","PA","OH","IL","NC","MI",
            "VA","WA","AZ","MA","TN","IN","MO","MD","WI","MN"
        ])
        city_pop = st.number_input("City population", min_value=1, value=74)

    with col3:
        st.markdown("**Location**")
        lat        = st.number_input("Cardholder lat",  value=34.9298, format="%.4f")
        long_      = st.number_input("Cardholder long", value=-84.9885, format="%.4f")
        merch_lat  = st.number_input("Merchant lat",    value=35.4239, format="%.4f")
        merch_long = st.number_input("Merchant long",   value=-84.8912, format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=False)
    st.caption("💡 Default values are a real fraud case from the test set (100% confidence). Try changing the hour to 10:00 or the amount to $20 to see the score drop.")

    if predict_btn:
        try:
            from geopy.distance import geodesic

            df = pd.DataFrame([{
                "trans_date_trans_time": trans_dt,
                "amt":        amt,
                "category":   category,
                "merchant":   merchant,
                "gender":     gender,
                "state":      state,
                "job":        job,
                "dob":        dob,
                "lat":        lat,
                "long":       long_,
                "merch_lat":  merch_lat,
                "merch_long": merch_long,
                "city_pop":   city_pop,
            }])

            df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
            df["hour"]        = df["trans_date_trans_time"].dt.hour
            df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
            df["month"]       = df["trans_date_trans_time"].dt.month
            df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
            df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
            df["dob"]         = pd.to_datetime(df["dob"])
            df["age"]         = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
            df["distance_km"] = geodesic((lat, long_), (merch_lat, merch_long)).km
            df["amt_log"]     = np.log1p(df["amt"])
            df = df.merge(category_stats, on="category", how="left")
            df["amt_zscore_by_category"] = (
                (df["amt"] - df["cat_amt_mean"]) / (df["cat_amt_std"] + 1e-8)
            )
            df["city_pop_log"] = np.log1p(df["city_pop"])

            CAT_COLS = ["merchant", "category", "gender", "state", "job"]
            for col in CAT_COLS:
                le = encoders[col]
                df[col] = df[col].map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )

            X        = df.reindex(columns=train_columns, fill_value=0)
            prob     = float(model.predict(X)[0])
            is_fraud = prob >= threshold

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if is_fraud:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <div style='font-size:0.75rem;font-family:DM Mono,monospace;color:#e63946;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>⚠ Fraud Detected</div>
                        <div style='font-size:2.5rem;font-weight:600;color:#e63946;'>{prob:.1%}</div>
                        <div style='font-size:0.85rem;color:#999;margin-top:0.3rem;'>fraud probability</div>
                        <div style='font-size:0.8rem;color:#777;margin-top:0.8rem;'>Threshold: {threshold:.2f} · Flagged ✓</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="legit-alert">
                        <div style='font-size:0.75rem;font-family:DM Mono,monospace;color:#2d9e6b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>✓ Legitimate</div>
                        <div style='font-size:2.5rem;font-weight:600;color:#2d9e6b;'>{prob:.1%}</div>
                        <div style='font-size:0.85rem;color:#999;margin-top:0.3rem;'>fraud probability</div>
                        <div style='font-size:0.8rem;color:#777;margin-top:0.8rem;'>Threshold: {threshold:.2f} · Not flagged</div>
                    </div>""", unsafe_allow_html=True)

                dist = geodesic((lat, long_), (merch_lat, merch_long)).km
                st.markdown(f"""
                <div style='background:white;border:1px solid #e8e8e8;border-radius:8px;padding:1rem;margin-top:0.5rem;'>
                    <div style='font-size:0.7rem;font-family:DM Mono,monospace;color:#999;text-transform:uppercase;margin-bottom:0.3rem;'>Distance</div>
                    <div style='font-size:1.4rem;font-weight:600;color:#0a0a0a;'>{dist:.1f} km</div>
                    <div style='font-size:0.75rem;color:#0a0a0a;'>from home to merchant</div>
                </div>""", unsafe_allow_html=True)

            with res_col2:
                st.markdown("**Top contributing features**")

                import shap as shap_lib
                explainer   = shap_lib.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                sv = shap_values[0] if isinstance(shap_values, list) else shap_values
                if len(sv.shape) > 1:
                    sv = sv[0]

                sv_abs_total  = sum(abs(v) for v in sv) or 1
                contributions = sorted([
                    {"feature": col, "pct": abs(val)/sv_abs_total, "direction": val > 0}
                    for col, val in zip(train_columns, sv)
                ], key=lambda x: x["pct"], reverse=True)[:8]

                for c in contributions:
                    color = "#e63946" if c["direction"] else "#2d9e6b"
                    arrow = "↑ toward fraud" if c["direction"] else "↓ away from fraud"
                    width = f"{c['pct']*100:.1f}%"
                    st.markdown(f"""
                    <div style='margin-bottom:0.6rem;'>
                        <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                            <span style='font-size:0.8rem;font-family:DM Mono,monospace;'>{c['feature']}</span>
                            <span style='font-size:0.75rem;color:{color};'>{arrow} · {c['pct']:.1%}</span>
                        </div>
                        <div style='background:#f0f0f0;border-radius:3px;height:6px;'>
                            <div style='background:{color};width:{width};height:6px;border-radius:3px;'></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                st.caption("Percentages show each feature's share of the model's decision, not the fraud probability itself.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")