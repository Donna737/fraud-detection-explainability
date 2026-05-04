# src/explain.py
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

OUTPUTS_DIR = "outputs"
MODELS_DIR  = "models"
DAYS        = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Maps encoded model feature → readable original column in test_df
READABLE_MAP = {
    "category": "category_original",
    "merchant": "merchant_original",
    "gender":   "gender_original",
    "state":    "state_original",
    "job":      "job_original",
}


# ── Readable display data ──────────────────────────────────────────────────────

def make_readable_explanation(
    explanation,
    X_test: pd.DataFrame,
    test_df: pd.DataFrame
):
    """
    Replace encoded integer values in the SHAP Explanation object
    with human-readable strings, so waterfall plots show:

        category  = grocery_pos    instead of   category  = 11
        gender    = M              instead of   gender    = 0
        merchant  = Shop-Mart      instead of   merchant  = 47
        distance  = 847.3 km       instead of   distance  = 847.3
        hour      = 2:00           instead of   hour      = 2

    How it works:
        SHAP Explanation objects have two data fields:
        - data          — the raw numeric values used for SHAP computation (unchanged)
        - display_data  — the values shown as labels in plots (we override this)

        So the SHAP values themselves never change — we only change what
        gets printed next to each bar in the waterfall.
    """
    display_df = X_test.copy().astype(object)

    # ── categorical columns → original string labels ──
    for encoded_col, original_col in READABLE_MAP.items():
        if encoded_col in display_df.columns and original_col in test_df.columns:
            display_df[encoded_col] = test_df.loc[display_df.index, original_col].values

    # ── numeric columns → formatted strings ──
    if "distance_km" in display_df.columns:
        display_df["distance_km"] = (
            X_test["distance_km"].round(1).astype(str) + " km"
        )
    if "amt" in display_df.columns:
        display_df["amt"] = "$" + X_test["amt"].round(2).astype(str)

    if "amt_log" in display_df.columns:
        display_df["amt_log"] = "$" + np.expm1(X_test["amt_log"]).round(2).astype(str)

    if "amt_zscore_by_category" in display_df.columns:
        display_df["amt_zscore_by_category"] = (
            X_test["amt_zscore_by_category"].round(2).astype(str) + " σ"
        )
    if "hour" in display_df.columns:
        display_df["hour"] = X_test["hour"].astype(int).astype(str) + ":00"

    if "age" in display_df.columns:
        display_df["age"] = X_test["age"].astype(int).astype(str) + " yrs"

    if "day_of_week" in display_df.columns:
        display_df["day_of_week"] = X_test["day_of_week"].astype(int).map(
            lambda d: DAYS[d]
        )
    if "is_night" in display_df.columns:
        display_df["is_night"] = X_test["is_night"].map({1: "yes", 0: "no"})

    if "is_weekend" in display_df.columns:
        display_df["is_weekend"] = X_test["is_weekend"].map({1: "yes", 0: "no"})

    if "city_pop_log" in display_df.columns:
        display_df["city_pop_log"] = (
            np.expm1(X_test["city_pop_log"]).round(0).astype(int).astype(str)
        )

    # ── inject display_data into Explanation object ──
    readable_explanation = shap.Explanation(
        values        = explanation.values,
        base_values   = explanation.base_values,
        data          = explanation.data,         # raw numeric — SHAP computation unchanged
        display_data  = display_df.values,        # readable strings — plot labels only
        feature_names = explanation.feature_names
    )

    return readable_explanation


# ── SHAP explainer + values ────────────────────────────────────────────────────

def build_explainer(
    model,
    X_test: pd.DataFrame,
    test_df: pd.DataFrame
):
    logger.info("Building SHAP TreeExplainer (raw output)...")

    explainer = shap.TreeExplainer(model)  # default raw output — always works

    # ── raw values (cached) ──
    shap_path = f"{MODELS_DIR}/shap_values.npy"
    if os.path.exists(shap_path):
        logger.info(f"Loading cached SHAP values from {shap_path}...")
        shap_values = np.load(shap_path)
    else:
        logger.info("Computing SHAP values (this takes a few minutes)...")
        shap_values = explainer.shap_values(X_test)
        os.makedirs(MODELS_DIR, exist_ok=True)
        np.save(shap_path, shap_values)
        logger.info(f"SHAP values cached to {shap_path}")

    # ── Explanation object ──
    logger.info("Building SHAP Explanation object...")
    explanation = explainer(X_test, check_additivity=False)

    # ── convert base_values and values from log-odds → probability ──
    # sigmoid(log-odds) = probability
    # We scale the shap values proportionally so they still sum correctly
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    raw_base    = explanation.base_values          # shape (n_samples,)
    raw_output  = raw_base + explanation.values.sum(axis=1)

    prob_base   = sigmoid(raw_base)                # baseline in probability
    prob_output = sigmoid(raw_output)              # final score in probability

    # scale factor — redistributes the probability difference
    # across features proportionally to their raw SHAP contribution
    scale = np.where(
        explanation.values.sum(axis=1) != 0,
        (prob_output - prob_base) / explanation.values.sum(axis=1),
        1.0
    )

    prob_shap_values = explanation.values * scale[:, None]

    prob_explanation = shap.Explanation(
        values        = prob_shap_values,
        base_values   = prob_base,
        data          = explanation.data,
        display_data  = None,              # readable labels added next
        feature_names = explanation.feature_names
    )

    prob_explanation = make_readable_explanation(prob_explanation, X_test, test_df)

    return explainer, shap_values, prob_explanation

# ── Global plots ───────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    save_path: str = None
):
    """
    Beeswarm summary plot — global view across all test transactions.
    Shows which features matter most and in which direction.
        Red   = high feature value pushed prediction toward fraud
        Blue  = low feature value pushed prediction toward fraud
    Best plot for your README — most information dense.
    """
    logger.info("Plotting SHAP beeswarm summary...")
    plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=15, show=False)
    plt.title("Global Feature Importance — SHAP Beeswarm", fontsize=13, pad=12)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_shap_bar(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    save_path: str = None
):
    """
    Bar plot of mean absolute SHAP values.
    Simpler than beeswarm — good for presentations and README intro.
    Pair with beeswarm: bar shows ranking, beeswarm shows direction.
    """
    logger.info("Plotting SHAP bar chart...")
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
    plt.title("Global Feature Importance — Mean |SHAP|", fontsize=13, pad=12)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


# ── Case selection ─────────────────────────────────────────────────────────────

def build_test_copy(
    X_test: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    threshold: float
) -> pd.DataFrame:
    """
    Enrich X_test with predictions and readable original string columns.
    The originals were dropped before training but we need them
    for the fraud alert cards printed to the terminal.
    """
    logger.info("Building enriched test copy...")

    X_test_copy = X_test.copy()
    X_test_copy["predicted_prob"]  = y_prob
    X_test_copy["predicted_class"] = (y_prob >= threshold).astype(int)
    X_test_copy["actual"]          = y_test.values

    readable_cols = [
        "amt", "category_original", "merchant_original",
        "job_original", "state_original", "gender_original",
        "distance_km"
    ]
    for col in readable_cols:
        if col in test_df.columns:
            X_test_copy[col] = test_df.loc[X_test_copy.index, col].values

    n_correct = (
        (X_test_copy["actual"] == 1) &
        (X_test_copy["predicted_class"] == 1)
    ).sum()
    logger.info(f"Correctly predicted fraud cases: {n_correct}")

    return X_test_copy


def select_interesting_cases(
    X_test_copy: pd.DataFrame,
    n: int = 4,
    random_state: int = 20
) -> pd.DataFrame:
    """
    Sample n true positive fraud cases for deep explanation.
    True positives = model flagged it AND it was actually fraud.
    These are the most useful cases for demonstrating explainability
    because the explanation is grounded in a real fraud event.
    """
    correct_fraud = X_test_copy[
        (X_test_copy["actual"] == 1) &
        (X_test_copy["predicted_class"] == 1)
    ].copy()

    logger.info(
        f"Selecting {n} cases from "
        f"{len(correct_fraud)} correctly predicted fraud cases..."
    )
    return correct_fraud.sample(n, random_state=random_state)


# ── Per-case waterfall plots ───────────────────────────────────────────────────

def plot_waterfall(
    explanation,
    X_test: pd.DataFrame,
    cases: pd.DataFrame,
    save_dir: str = OUTPUTS_DIR
) -> list:
    """
    Generate one SHAP waterfall plot per selected fraud case.

    Each waterfall shows:
        - The model baseline (average prediction across all transactions)
        - Each feature's contribution: how much it pushed the score
          up (red, toward fraud) or down (blue, away from fraud)
        - The final fraud probability at the top
        - Human-readable feature values on the right
          (e.g. category=grocery_pos, gender=M, distance=847.3 km)

    This replaces template-based text explanations entirely:
        - Zero feature-specific code — works for any feature automatically
        - Shows exact magnitude of each contribution
        - Scales to any number of features without changes
        - Produces screenshot-ready images for your README
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = []

    for i, (idx, row) in enumerate(cases.iterrows(), start=1):
        pos      = X_test.index.get_loc(idx)
        merchant = str(row.get("merchant_original", "unknown")).replace("fraud_", "")
        category = str(row.get("category_original", "unknown")).replace("_", " ")

        plt.figure()
        shap.plots.waterfall(explanation[pos], max_display=12, show=False)
        ax = plt.gca()
        ax.set_xlabel(
            "Fraud probability contribution\n"
            "◀  reduces fraud score          increases fraud score  ▶",
            fontsize=8, labelpad=6
        )
        ax.set_ylabel("Features (sorted by impact)", fontsize=8, labelpad=6)
        plt.title(
            f"Case {i}  |  {row['predicted_prob']:.0%} fraud probability\n"
            f"${row['amt']:.2f} at {merchant} ({category})",
            fontsize=10,
            pad=12
        )

        path = f"{save_dir}/waterfall_case_{i}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Waterfall saved: {path}")
        paths.append(path)

    return paths


# ── Fraud alert card ───────────────────────────────────────────────────────────

def print_fraud_alert(
    cases: pd.DataFrame,
    waterfall_paths: list
):
    """
    Print a structured fraud alert card per case to the terminal.
    Covers transaction context and cardholder profile.
    The WHY (feature contributions) lives in the waterfall plot —
    referenced at the bottom of each card.
    """
    for (idx, row), wf_path in zip(cases.iterrows(), waterfall_paths):
        merchant = str(row.get("merchant_original", "unknown")).replace("fraud_", "")
        category = str(row.get("category_original", "unknown")).replace("_", " ")
        day      = DAYS[int(row["day_of_week"])]
        timing   = "night" if row["is_night"] == 1 else "day"

        print(f"\n{'─'*62}")
        print(f"  FRAUD ALERT  │  Transaction #{idx}  │  {row['predicted_prob']:.0%} confidence")
        print(f"{'─'*62}")

        print(f"\n  📋 TRANSACTION")
        print(f"     ${row['amt']:.2f}  at  {merchant}  ({category})")
        print(f"     {day} {int(row['hour'])}:00 ({timing})  •  {row['distance_km']:.1f} km from home")

        print(f"\n  👤 CARDHOLDER")
        print(f"     {int(row['age'])} y/o  •  {row.get('job_original', 'unknown')}")
        print(f"     {row.get('state_original', 'unknown')}  •  {row.get('gender_original', 'unknown')}")

        print(f"\n  📊 EXPLANATION")
        print(f"     {wf_path}")
        print(f"     Each bar shows how much that feature pushed")
        print(f"     the fraud score up (red) or down (blue).")

        print(f"\n  ⚠️  Recommendation: Contact cardholder to verify.")
        print(f"{'─'*62}")


# ── Full explanation pipeline ──────────────────────────────────────────────────

def run_explanation(
    model,
    X_test:    pd.DataFrame,
    test_df:   pd.DataFrame,
    y_test:    pd.Series,
    y_prob:    np.ndarray,
    threshold: float
):
    """
    Full explanation pipeline:

    1. Build SHAP explainer + compute values (with readable display labels)
    2. Global plots  — beeswarm + bar  (what drives fraud overall)
    3. Case selection — 4 true positive fraud cases
    4. Waterfall plots — one per case  (why THIS transaction was flagged)
    5. Fraud alert cards — terminal output with waterfall references
    6. Save case summary CSV

    Returns cases DataFrame for downstream use (e.g. api.py).
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── 1. SHAP ──
    explainer, shap_values, explanation = build_explainer(model, X_test, test_df)

    # ── 2. Global plots ──
    plot_shap_summary(shap_values, X_test, save_path=f"{OUTPUTS_DIR}/shap_summary.png")
    plot_shap_bar(shap_values, X_test,     save_path=f"{OUTPUTS_DIR}/shap_bar.png")

    # ── 3. Case selection ──
    X_test_copy = build_test_copy(X_test, test_df, y_test, y_prob, threshold)
    cases       = select_interesting_cases(X_test_copy, n=4)

    # ── 4. Waterfall plots ──
    waterfall_paths = plot_waterfall(explanation, X_test, cases)

    # ── 5. Alert cards ──
    print_fraud_alert(cases, waterfall_paths)

    # ── 6. Save ──
    cases_path = f"{OUTPUTS_DIR}/explained_cases.csv"
    cases.drop(columns=["shap_top_features"], errors="ignore").to_csv(cases_path)
    logger.info(f"Case summary saved to {cases_path}")

    logger.info("=== Explanation complete ===")
    logger.info(f"Global:    {OUTPUTS_DIR}/shap_summary.png, shap_bar.png")
    logger.info(f"Per-case:  {OUTPUTS_DIR}/waterfall_case_1..{len(cases)}.png")

    return cases


# ── Run standalone ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append("src")

    from data_loader import load_data
    from preprocess  import run_preprocessing
    from train       import run_training

    train_df, test_df = load_data(
        train_path="data/fraudTrain.csv",
        test_path="data/fraudTest.csv"
    )

    X_train, y_train, X_test, y_test, encoders, test_df, features = run_preprocessing(
        train_df, test_df
    )

    model, threshold, y_prob = run_training(
        X_train, y_train, X_test, y_test, features
    )

    cases = run_explanation(
        model, X_test, test_df, y_test, y_prob, threshold
    )