# tests/test_model.py
import pytest
import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@pytest.fixture(scope="module")
def model_artifacts(models_available):
    """Load model artifacts — skip all model tests if not available."""
    if not models_available:
        pytest.skip("Model artifacts not found — skipping model tests")
    return {
        "model":          joblib.load(f"{MODELS_DIR}/model.pkl"),
        "threshold":      joblib.load(f"{MODELS_DIR}/threshold.pkl"),
        "encoders":       joblib.load(f"{MODELS_DIR}/encoders.pkl"),
        "category_stats": joblib.load(f"{MODELS_DIR}/category_stats.pkl"),
        "train_columns":  joblib.load(f"{MODELS_DIR}/train_columns.pkl"),
    }


class TestModelLoading:

    def test_model_loads(self, model_artifacts):
        """Model must load without error."""
        assert model_artifacts["model"] is not None

    def test_threshold_in_range(self, model_artifacts):
        """Threshold must be between 0 and 1."""
        t = model_artifacts["threshold"]
        assert 0.0 < t < 1.0, f"Threshold {t} out of range (0, 1)"

    def test_encoders_have_expected_keys(self, model_artifacts):
        """Encoders must exist for all categorical columns."""
        expected = {"merchant", "category", "gender", "state", "job"}
        actual   = set(model_artifacts["encoders"].keys())
        assert expected == actual, f"Encoder keys mismatch: {actual}"

    def test_train_columns_non_empty(self, model_artifacts):
        """train_columns must be a non-empty list."""
        cols = model_artifacts["train_columns"]
        assert isinstance(cols, list) and len(cols) > 0


class TestModelPrediction:

    def _make_input(self, model_artifacts):
        """Build a minimal valid feature row aligned to train_columns."""
        cols = model_artifacts["train_columns"]
        return pd.DataFrame([np.zeros(len(cols))], columns=cols)

    def test_predict_proba_shape(self, model_artifacts):
        """predict_proba must return shape (1, 2)."""
        X     = self._make_input(model_artifacts)
        proba = model_artifacts["model"].predict_proba(X)
        assert proba.shape == (1, 2), f"Unexpected shape: {proba.shape}"

    def test_predict_proba_sums_to_one(self, model_artifacts):
        """Class probabilities must sum to 1."""
        X     = self._make_input(model_artifacts)
        proba = model_artifacts["model"].predict_proba(X)
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_fraud_probability_in_range(self, model_artifacts):
        """Fraud probability must be between 0 and 1."""
        X    = self._make_input(model_artifacts)
        prob = float(model_artifacts["model"].predict_proba(X)[:, 1][0])
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range"

    def test_prediction_is_deterministic(self, model_artifacts):
        """Same input must always produce same output."""
        X     = self._make_input(model_artifacts)
        prob1 = float(model_artifacts["model"].predict_proba(X)[:, 1][0])
        prob2 = float(model_artifacts["model"].predict_proba(X)[:, 1][0])
        assert prob1 == prob2, "Model is not deterministic"

    def test_threshold_produces_binary_prediction(self, model_artifacts):
        """Applying threshold must produce 0 or 1."""
        X         = self._make_input(model_artifacts)
        prob      = float(model_artifacts["model"].predict_proba(X)[:, 1][0])
        threshold = model_artifacts["threshold"]
        prediction = int(prob >= threshold)
        assert prediction in {0, 1}
