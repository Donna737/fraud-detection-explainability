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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
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

MIN_RECALL  = 0.90
VAL_SIZE    = 0.2
OUTPUTS_DIR = "outputs"
MODELS_DIR  = "models"


# ── Training ───────────────────────────────────────────────────────────────────

def split_train_val(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    val_size: float = VAL_SIZE
):
    """
    Split training data into train and validation sets.

    Why stratify:
        Fraud rate is ~0.5% — without stratify, your validation set
        might randomly end up with very few fraud cases, making
        threshold selection unreliable.

    Why we need this split at all:
        Threshold selection must happen on data the model hasn't
        been trained on, but the final test set should only be
        touched once for honest evaluation. The validation set
        is the correct place to tune the threshold.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=42,
        stratify=y_train
    )

    logger.info(f"Train:      {X_tr.shape}  | fraud rate: {y_tr.mean():.4f}")
    logger.info(f"Validation: {X_val.shape} | fraud rate: {y_val.mean():.4f}")

    return X_tr, X_val, y_tr, y_val


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


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train logistic regression as a challenger/baseline model.
    Requires scaling since LR is sensitive to feature magnitudes.
    Uses class_weight='balanced' for imbalance correction.

    Why logistic regression as baseline:
        It's the simplest interpretable model — if LightGBM doesn't
        meaningfully outperform it, the complexity isn't justified.
        It also serves as a sanity check on the feature engineering.
    """
    logger.info("Training logistic regression baseline...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_scaled, y_train)

    logger.info("Baseline training complete.")
    return lr, scaler


# ── Threshold selection ────────────────────────────────────────────────────────

def select_threshold(
    y_val: pd.Series,
    y_val_prob: np.ndarray,
    min_recall: float = MIN_RECALL
):
    """
    Business-driven threshold selection on the VALIDATION set.

    Find the threshold that maximizes precision while keeping
    recall >= min_recall.

    Why validation and not test:
        We're searching over many thresholds and picking the best one —
        that's a form of optimization. Doing it on the test set would
        make our final metrics optimistic. The validation set is used
        for this tuning step; the test set is only touched once at the end.
    """
    logger.info(f"Selecting threshold on validation set (min recall = {min_recall})...")

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    pr_auc = auc(recalls, precisions)

    valid_mask = recalls[:-1] >= min_recall
    if not valid_mask.any():
        logger.warning(
            f"No threshold achieves recall >= {min_recall}. "
            "Relaxing to best available recall."
        )
        valid_mask = np.ones(len(thresholds), dtype=bool)

    best_idx  = np.argmax(precisions[:-1][valid_mask])
    threshold = thresholds[valid_mask][best_idx]

    logger.info(f"Validation PR-AUC:  {pr_auc:.3f}")
    logger.info(f"Chosen threshold:   {threshold:.3f}")

    return threshold, pr_auc, precisions, recalls, thresholds


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    label: str = "tuned"
) -> tuple[dict, np.ndarray]:
    """Compute precision, recall, F1 at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "threshold": threshold,
    }

    logger.info(f"\n── Metrics at {label} threshold ({threshold:.2f}) ──")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.3f}")

    return metrics, y_pred


def compare_thresholds(
    y_true: pd.Series,
    y_prob: np.ndarray,
    chosen_threshold: float
) -> pd.DataFrame:
    """Compare default (0.5) vs business-tuned threshold side by side."""
    metrics_default, _ = evaluate(y_true, y_prob, threshold=0.5,              label="default")
    metrics_tuned,   _ = evaluate(y_true, y_prob, threshold=chosen_threshold, label="tuned")

    comparison = pd.DataFrame(
        [metrics_default, metrics_tuned],
        index=["Default (0.50)", f"Tuned ({chosen_threshold:.2f})"]
    )
    print("\n── Threshold Comparison (on test set) ──")
    print(comparison.round(3))
    return comparison


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_pr_curve(
    precisions, recalls, thresholds,
    chosen_threshold: float,
    pr_auc: float,
    title_suffix: str = "Validation Set",
    save_path: str = None
):
    """Precision-Recall curve with chosen threshold marked."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    ax.set_title(f"Precision-Recall Curve ({title_suffix})")
    ax.legend()
    ax.grid(alpha=0.3)

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
    ax.set_title(f"F1 Score vs Threshold ({title_suffix})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PR curve saved to {save_path}")

    plt.show()


def plot_roc_pr_comparison(
    y_test: pd.Series,
    y_test_prob_lgbm: np.ndarray,
    y_test_prob_lr: np.ndarray,
    save_path: str = None
):
    """
    Side-by-side ROC-AUC vs PR-AUC for LightGBM and Logistic Regression.

    Why this comparison matters:
        ROC-AUC is misleading for imbalanced datasets — a model that
        always predicts 'not fraud' can still score ~0.99 ROC-AUC
        because the true negative rate dominates. PR-AUC focuses only
        on the positive class (fraud) and is far more informative when
        fraud rate is 0.58%. This plot makes that argument visually.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── ROC curves ──
    ax = axes[0]
    for prob, label, color in [
        (y_test_prob_lgbm, "LightGBM", "steelblue"),
        (y_test_prob_lr,   "Logistic Regression (baseline)", "darkorange"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Misleading for Imbalanced Data")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── PR curves ──
    ax = axes[1]
    for prob, label, color in [
        (y_test_prob_lgbm, "LightGBM", "steelblue"),
        (y_test_prob_lr,   "Logistic Regression (baseline)", "darkorange"),
    ]:
        prec, rec, _ = precision_recall_curve(y_test, prob)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, linewidth=2, color=color,
                label=f"{label} (AUC = {pr_auc:.3f})")

    baseline_rate = y_test.mean()
    ax.axhline(y=baseline_rate, color="gray", linestyle="--", linewidth=1,
           label=f"Random classifier — fraud rate in test set ({baseline_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve — Better Metric for Fraud Detection")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        "ROC-AUC vs PR-AUC: Why metric choice matters at 0.58% fraud rate",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC vs PR comparison saved to {save_path}")

    plt.show()


def plot_model_comparison(
    y_test: pd.Series,
    y_test_prob_lgbm: np.ndarray,
    y_test_prob_lr: np.ndarray,
    threshold: float,
    save_path: str = None
) -> pd.DataFrame:
    """
    Bar chart comparing LightGBM vs Logistic Regression at the same threshold.
    Shows why the added complexity of LightGBM is justified.
    """
    metrics = {}
    for prob, name in [
        (y_test_prob_lgbm, "LightGBM"),
        (y_test_prob_lr,   "Logistic Regression"),
    ]:
        y_pred = (prob >= 0.5).astype(int)
        metrics[name] = {
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1":        f1_score(y_test, y_pred),
        }

    comparison_df = pd.DataFrame(metrics).T
    logger.info("\n── Model Comparison (test set, same threshold) ──")
    logger.info(comparison_df.round(3).to_string())

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comparison_df.columns))
    width = 0.35

    bars1 = ax.bar(x - width/2, comparison_df.loc["LightGBM"],
                   width, label="LightGBM", color="steelblue")
    bars2 = ax.bar(x + width/2, comparison_df.loc["Logistic Regression"],
                   width, label="Logistic Regression (baseline)", color="darkorange",
                   alpha=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.columns)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"LightGBM vs Logistic Regression (both at default threshold 0.50)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Model comparison saved to {save_path}")

    plt.show()
    return comparison_df


def plot_confusion_matrix(
    y_true: pd.Series,
    y_prob: np.ndarray,
    chosen_threshold: float,
    save_path: str = None
):
    """Side-by-side confusion matrices: default (0.5) vs tuned threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, threshold, title in zip(
        axes,
        [0.5, chosen_threshold],
        ["Default Threshold (0.50)", f"Tuned Threshold ({chosen_threshold:.2f})"]
    ):
        y_pred = (y_prob >= threshold).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
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

    model.booster_.save_model(f"{MODELS_DIR}/model.txt")
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
    Full training pipeline with proper train / val / test separation:

        X_train  →  split into X_tr (fit model) + X_val (select threshold)
        X_test   →  final honest evaluation only, never touched before this step

    MLflow tracks everything: params, val metrics, test metrics, artifacts.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with mlflow.start_run():

        # ── log parameters ──
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("min_recall_floor", MIN_RECALL)
        mlflow.log_param("val_size",         VAL_SIZE)
        mlflow.log_param("n_features",       len(features))

        # ── split train → train + val ──
        X_tr, X_val, y_tr, y_val = split_train_val(X_train, y_train)

        # ── fit LightGBM on train only ──
        model = train_model(X_tr, y_tr)

        # ── fit logistic regression baseline on train only ──
        lr_model, scaler = train_baseline(X_tr, y_tr)
        joblib.dump(scaler, f"{MODELS_DIR}/lr_scaler.pkl")
        joblib.dump(lr_model, f"{MODELS_DIR}/lr_model.pkl")

        # ── select threshold on validation set ──
        y_val_prob = model.predict_proba(X_val)[:, 1]
        threshold, pr_auc_val, precisions, recalls, thresholds = select_threshold(
            y_val, y_val_prob, min_recall=MIN_RECALL
        )

        # log validation metrics
        val_metrics, _ = evaluate(y_val, y_val_prob, threshold, label="val")
        mlflow.log_metric("val_pr_auc",    pr_auc_val)
        mlflow.log_metric("val_precision", val_metrics["precision"])
        mlflow.log_metric("val_recall",    val_metrics["recall"])
        mlflow.log_metric("val_f1",        val_metrics["f1"])
        mlflow.log_metric("threshold",     threshold)

        # ── PR curve plot ──
        pr_path = f"{OUTPUTS_DIR}/pr_curve.png"
        plot_pr_curve(
            precisions, recalls, thresholds,
            threshold, pr_auc_val,
            title_suffix="Validation Set",
            save_path=pr_path
        )
        mlflow.log_artifact(pr_path)

        # ── final honest evaluation on test set ──
        logger.info("\n=== Final evaluation on held-out test set ===")
        y_test_prob    = model.predict_proba(X_test)[:, 1]
        y_test_prob_lr = lr_model.predict_proba(
            scaler.transform(X_test)
        )[:, 1]

        test_metrics, _ = evaluate(y_test, y_test_prob, threshold, label="test")
        compare_thresholds(y_test, y_test_prob, threshold)

        mlflow.log_metric("test_precision", test_metrics["precision"])
        mlflow.log_metric("test_recall",    test_metrics["recall"])
        mlflow.log_metric("test_f1",        test_metrics["f1"])

        # ── ROC vs PR comparison plot ──
        roc_pr_path = f"{OUTPUTS_DIR}/roc_pr_comparison.png"
        plot_roc_pr_comparison(
            y_test, y_test_prob, y_test_prob_lr,
            save_path=roc_pr_path
        )
        mlflow.log_artifact(roc_pr_path)

        # ── model comparison plot ──
        model_comp_path = f"{OUTPUTS_DIR}/model_comparison.png"
        comparison_df = plot_model_comparison(
            y_test, y_test_prob, y_test_prob_lr,
            threshold, save_path=model_comp_path
        )
        mlflow.log_artifact(model_comp_path)

        # ── log baseline metrics ──
        y_pred_lr = (y_test_prob_lr >= threshold).astype(int)
        mlflow.log_metric("baseline_precision", precision_score(y_test, y_pred_lr))
        mlflow.log_metric("baseline_recall",    recall_score(y_test, y_pred_lr))
        mlflow.log_metric("baseline_f1",        f1_score(y_test, y_pred_lr))

        # ── confusion matrix ──
        cm_path = f"{OUTPUTS_DIR}/confusion_matrix.png"
        plot_confusion_matrix(y_test, y_test_prob, threshold, save_path=cm_path)
        mlflow.log_artifact(cm_path)

        # ── save model ──
        save_model(model, threshold, features)
        mlflow.lightgbm.log_model(model, artifact_path="model")

        logger.info("=== Training run complete ===")

    return model, threshold, y_test_prob


# ── Run standalone ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append("src")

    from data_loader import load_data
    from preprocess  import run_preprocessing

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