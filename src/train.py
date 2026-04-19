# src/train.py
import logging
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import (
    precision_recall_curve, auc,
    precision_score, recall_score,
    f1_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_PARAMS = {
    "n_estimators":    500,
    "learning_rate":   0.05,
    "num_leaves":      31,
    "random_state":    42,
    "n_jobs":          -1,
}

# business decision: catch at least 80% of fraud (min recall)
# then maximize precision at that point to reduce false alarms
MIN_RECALL   = 0.80
OUTPUTS_DIR  = "outputs"
MODELS_DIR   = "models"


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    """
    Train LightGBM with class imbalance correction.
    scale_pos_weight compensates for the heavy fraud/non-fraud imbalance
    in the Sparkov dataset (~0.5% fraud rate).
    """
    logger.info("Training LightGBM model...")

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        **MODEL_PARAMS,
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train)

    logger.info("Training complete.")
    return model


# ── Threshold selection ────────────────────────────────────────────────────────

def select_threshold(
    y_test: pd.Series,
    y_prob: np.ndarray,
    min_recall: float = MIN_RECALL
) -> tuple[float, float]:
    """
    Business-driven threshold selection:
    Find the threshold that maximizes precision while keeping recall >= min_recall.

    In fraud detection, missing fraud (low recall) is more costly than
    a false alarm, so we set a recall floor and optimize precision above it.

    Returns:
        chosen_threshold  — the selected operating threshold
        pr_auc            — area under the precision-recall curve
    """
    logger.info(f"Selecting threshold (min recall = {min_recall})...")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recalls, precisions)

    valid_mask  = recalls[:-1] >= min_recall
    best_idx    = np.argmax(precisions[:-1][valid_mask])
    threshold   = thresholds[valid_mask][best_idx]

    logger.info(f"PR-AUC:           {pr_auc:.3f}")
    logger.info(f"Chosen threshold: {threshold:.3f}")

    return threshold, pr_auc, precisions, recalls, thresholds


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    y_test: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    label: str = "tuned"
) -> dict:
    """
    Compute precision, recall, F1 at a given threshold.
    Returns a dict of metrics (easy to log to MLflow).
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "threshold": threshold,
    }

    logger.info(f"\n── Metrics at {label} threshold ({threshold:.2f}) ──")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.3f}")

    return metrics, y_pred


def compare_thresholds(
    y_test: pd.Series,
    y_prob: np.ndarray,
    chosen_threshold: float
) -> pd.DataFrame:
    """Compare default (0.5) vs business-tuned threshold side by side."""
    metrics_default, _ = evaluate(y_test, y_prob, threshold=0.5,              label="default")
    metrics_tuned,   _ = evaluate(y_test, y_prob, threshold=chosen_threshold, label="tuned")

    comparison = pd.DataFrame(
        [metrics_default, metrics_tuned],
        index=[f"Default (0.50)", f"Tuned ({chosen_threshold:.2f})"]
    )
    print("\n── Threshold Comparison ──")
    print(comparison.round(3))
    return comparison


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_pr_curve(
    precisions, recalls, thresholds,
    chosen_threshold: float,
    pr_auc: float,
    save_path: str = None
):
    """Precision-Recall curve with chosen threshold marked."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── PR curve ──
    ax = axes[0]
    ax.plot(recalls, precisions, linewidth=2, color="steelblue",
            label=f"PR Curve (AUC = {pr_auc:.3f})")

    idx = np.argmin(np.abs(thresholds - chosen_threshold))
    ax.scatter(recalls[idx], precisions[idx],
               color="red", s=100, zorder=5,
               label=f"Chosen threshold = {chosen_threshold:.2f}")

    for t in [0.3, 0.5, 0.7, 0.9]:
        i = np.argmin(np.abs(thresholds - t))
        ax.annotate(f"t={t}",
            xy=(recalls[i], precisions[i]),
            xytext=(recalls[i] + 0.02, precisions[i] - 0.06),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── F1 vs threshold ──
    ax = axes[1]
    f1_scores = (
        2 * (precisions[:-1] * recalls[:-1])
        / (precisions[:-1] + recalls[:-1] + 1e-8)
    )
    ax.plot(thresholds, f1_scores, linewidth=2, color="darkorange")
    ax.axvline(x=chosen_threshold, color="red", linestyle="--",
               label=f"Chosen = {chosen_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PR curve saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_test: pd.Series,
    y_prob: np.ndarray,
    chosen_threshold: float,
    save_path: str = None
):
    """Side-by-side confusion matrices for default vs tuned threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, threshold, title in zip(
        axes,
        [0.5, chosen_threshold],
        [f"Default Threshold (0.50)", f"Tuned Threshold ({chosen_threshold:.2f})"]
    ):
        y_pred = (y_prob >= threshold).astype(int)
        cm     = confusion_matrix(y_test, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        labels = np.array([
            [f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
            for row_v, row_p in zip(cm, cm_pct)
        ])

        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
                    cbar=False, ax=ax, linewidths=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()


# ── Saving ─────────────────────────────────────────────────────────────────────

def save_model(model, threshold: float, features: list):
    """Save model, threshold, and feature list to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model,    f"{MODELS_DIR}/model.pkl")
    joblib.dump(threshold, f"{MODELS_DIR}/threshold.pkl")
    joblib.dump(features,  f"{MODELS_DIR}/features.pkl")

    logger.info(f"Model and metadata saved to {MODELS_DIR}/")


# ── Full training run ──────────────────────────────────────────────────────────

def run_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    features: list
):
    """
    Full training pipeline with MLflow tracking.
    Call this from pipeline.py or standalone.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with mlflow.start_run():

        # ── log parameters ──
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("min_recall_floor", MIN_RECALL)
        mlflow.log_param("n_features", len(features))

        # ── train ──
        model  = train_model(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ── threshold selection ──
        threshold, pr_auc, precisions, recalls, thresholds = select_threshold(
            y_test, y_prob, min_recall=MIN_RECALL
        )

        # ── evaluate ──
        metrics, _ = evaluate(y_test, y_prob, threshold)
        compare_thresholds(y_test, y_prob, threshold)

        # ── log metrics to MLflow ──
        mlflow.log_metric("pr_auc",    pr_auc)
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])
        mlflow.log_metric("f1",        metrics["f1"])
        mlflow.log_metric("threshold", threshold)

        # ── plots ──
        pr_path = f"{OUTPUTS_DIR}/pr_curve.png"
        cm_path = f"{OUTPUTS_DIR}/confusion_matrix.png"

        plot_pr_curve(precisions, recalls, thresholds, threshold, pr_auc, save_path=pr_path)
        plot_confusion_matrix(y_test, y_prob, threshold, save_path=cm_path)

        mlflow.log_artifact(pr_path)
        mlflow.log_artifact(cm_path)

        # ── save model ──
        save_model(model, threshold, features)
        mlflow.lightgbm.log_model(model, artifact_path="model")

        logger.info("=== Training run complete ===")

    return model, threshold, y_prob


# ── Run standalone ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append("src")

    from data_loader  import load_data
    from preprocess   import run_preprocessing

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