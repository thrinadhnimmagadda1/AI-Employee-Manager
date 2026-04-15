"""
Tests for the burnout predictor ML model.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
import pickle


HIGH_RISK_FEATURES = {
    "satisfaction_level": 0.11,
    "last_evaluation": 0.88,
    "number_project": 7,
    "average_montly_hours": 310,
    "time_spend_company": 5,
    "work_accident": 0,
    "promotion_last_5years": 0,
}

LOW_RISK_FEATURES = {
    "satisfaction_level": 0.82,
    "last_evaluation": 0.73,
    "number_project": 3,
    "average_montly_hours": 165,
    "time_spend_company": 2,
    "work_accident": 0,
    "promotion_last_5years": 1,
}


def _make_mock_artifact(prob: float) -> dict:
    """Build a mock artifact dict for the burnout predictor."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - prob, prob]])

    mock_base = MagicMock()

    # SHAP mock: return array of feature impacts
    feature_cols = list(HIGH_RISK_FEATURES.keys())
    mock_explainer = MagicMock()
    mock_shap_vals = np.array([[0.1 * i for i in range(len(feature_cols))]])
    mock_explainer.shap_values.return_value = [None, mock_shap_vals]

    return {
        "model": mock_model,
        "base_model": mock_base,
        "explainer": mock_explainer,
        "feature_columns": feature_cols,
        "metrics": {"accuracy": 0.90, "f1": 0.85, "auc": 0.92},
    }


class TestBurnoutPredictor:
    def test_high_risk_score_above_threshold(self):
        """High-risk feature set should produce a score above 0.5."""
        with patch("src.ml.burnout_predictor._load_artifact", return_value=_make_mock_artifact(0.85)):
            from src.ml.burnout_predictor import predict
            result = predict(HIGH_RISK_FEATURES)
        assert result["burnout_risk"] >= 0.5, \
            f"Expected burnout_risk >= 0.5 for high-risk features, got {result['burnout_risk']}"

    def test_low_risk_score_below_threshold(self):
        """Low-risk feature set should produce a score below 0.5."""
        with patch("src.ml.burnout_predictor._load_artifact", return_value=_make_mock_artifact(0.12)):
            from src.ml.burnout_predictor import predict
            result = predict(LOW_RISK_FEATURES)
        assert result["burnout_risk"] < 0.5, \
            f"Expected burnout_risk < 0.5 for low-risk features, got {result['burnout_risk']}"

    def test_output_score_between_0_and_1(self):
        """Burnout risk score must always be in [0.0, 1.0]."""
        for prob in [0.0, 0.25, 0.5, 0.75, 1.0]:
            with patch("src.ml.burnout_predictor._load_artifact", return_value=_make_mock_artifact(prob)):
                from src.ml.burnout_predictor import predict
                result = predict(HIGH_RISK_FEATURES)
            assert 0.0 <= result["burnout_risk"] <= 1.0, \
                f"burnout_risk = {result['burnout_risk']} outside [0, 1]"

    def test_shap_values_present_in_output(self):
        """Response must include shap_values dict."""
        with patch("src.ml.burnout_predictor._load_artifact", return_value=_make_mock_artifact(0.72)):
            from src.ml.burnout_predictor import predict
            result = predict(HIGH_RISK_FEATURES)
        assert "shap_values" in result, "shap_values key missing from predict() output"
        assert isinstance(result["shap_values"], dict), "shap_values must be a dict"

    def test_top_reasons_list_returned(self):
        """top_reasons must be a list with at most 3 items."""
        with patch("src.ml.burnout_predictor._load_artifact", return_value=_make_mock_artifact(0.80)):
            from src.ml.burnout_predictor import predict
            result = predict(HIGH_RISK_FEATURES)
        assert "top_reasons" in result
        assert isinstance(result["top_reasons"], list)
        assert len(result["top_reasons"]) <= 3

    def test_top_reasons_have_required_keys(self):
        """Each top_reason must have 'feature', 'impact', and 'direction' keys."""
        with patch("src.ml.burnout_predictor._load_artifact", return_value=_make_mock_artifact(0.80)):
            from src.ml.burnout_predictor import predict
            result = predict(HIGH_RISK_FEATURES)
        for reason in result.get("top_reasons", []):
            assert "feature" in reason, f"Missing 'feature' key in reason: {reason}"
            assert "impact" in reason, f"Missing 'impact' key in reason: {reason}"
            assert "direction" in reason, f"Missing 'direction' key in reason: {reason}"

    def test_direction_is_valid_string(self):
        """Direction must be either 'increases_risk' or 'decreases_risk'."""
        with patch("src.ml.burnout_predictor._load_artifact", return_value=_make_mock_artifact(0.80)):
            from src.ml.burnout_predictor import predict
            result = predict(HIGH_RISK_FEATURES)
        valid_directions = {"increases_risk", "decreases_risk"}
        for reason in result.get("top_reasons", []):
            assert reason["direction"] in valid_directions, \
                f"Invalid direction: {reason['direction']}"
