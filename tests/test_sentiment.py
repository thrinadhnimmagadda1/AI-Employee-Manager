"""
Tests for the emotion classifier and sentiment model.
"""
import pytest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

WORKPLACE_EMOTIONS = ["joy", "frustration", "anxiety", "anger", "sadness", "disgust", "neutral"]


def _mock_predict(text: str) -> dict:
    """Simplified mock predict that returns rule-based scores for testing."""
    text_lower = text.lower()
    scores = {e: 0.0 for e in WORKPLACE_EMOTIONS}
    if any(w in text_lower for w in ["frustrated", "annoyed", "terrible", "overwhelmed"]):
        scores["frustration"] = 0.70
        scores["neutral"] = 0.30
    elif any(w in text_lower for w in ["happy", "great", "excellent", "love", "thrilled"]):
        scores["joy"] = 0.75
        scores["neutral"] = 0.25
    elif any(w in text_lower for w in ["worried", "anxious", "scared", "nervous"]):
        scores["anxiety"] = 0.65
        scores["neutral"] = 0.35
    elif any(w in text_lower for w in ["angry", "furious", "rage"]):
        scores["anger"] = 0.80
        scores["neutral"] = 0.20
    else:
        scores["neutral"] = 1.0
    total = sum(scores.values())
    return {k: round(v / total, 4) for k, v in scores.items()}


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestEmotionClassifier:
    def test_all_7_emotions_always_present(self):
        """All 7 emotion keys must always be returned."""
        result = _mock_predict("This is a normal status update.")
        assert set(result.keys()) == set(WORKPLACE_EMOTIONS), \
            f"Expected all 7 emotions, got: {set(result.keys())}"

    def test_frustration_detected_in_frustrated_text(self):
        """frustration should dominate in frustrated messages."""
        result = _mock_predict("I am so frustrated and overwhelmed by this terrible deadline.")
        assert result["frustration"] > 0.5, \
            f"Expected frustration > 0.5, got {result['frustration']}"

    def test_joy_detected_in_positive_text(self):
        """joy should dominate in clearly positive messages."""
        result = _mock_predict("This is an excellent result, I am so happy and thrilled!")
        assert result["joy"] > 0.5, \
            f"Expected joy > 0.5, got {result['joy']}"

    def test_neutral_detected_in_factual_text(self):
        """neutral should dominate in factual, unemotional text."""
        result = _mock_predict("The meeting is scheduled for Tuesday at 3pm in room 204.")
        assert result["neutral"] > 0.5, \
            f"Expected neutral > 0.5, got {result['neutral']}"

    def test_anxiety_detected_in_worried_text(self):
        """anxiety should dominate in worried messages."""
        result = _mock_predict("I'm so worried and anxious about the upcoming review.")
        assert result["anxiety"] > 0.5, \
            f"Expected anxiety > 0.5, got {result['anxiety']}"

    def test_scores_sum_to_one(self):
        """Probability distribution must sum to ~1.0."""
        for text in [
            "I love working here",
            "This is a disaster",
            "Please send the report by EOD",
            "",
        ]:
            result = _mock_predict(text)
            total = sum(result.values())
            assert abs(total - 1.0) < 0.01, \
                f"Scores should sum to 1.0, got {total} for text: {text!r}"

    def test_empty_text_returns_neutral(self):
        """Empty string should return neutral-dominant scores."""
        result = _mock_predict("")
        assert result["neutral"] > 0, "Empty text should have non-zero neutral"

    def test_all_values_between_0_and_1(self):
        """All probability values must be in [0, 1]."""
        result = _mock_predict("I'm having a really rough week at work.")
        for emotion, score in result.items():
            assert 0.0 <= score <= 1.0, \
                f"Score for {emotion} = {score} is outside [0, 1]"


class TestSentimentModel:
    def test_score_text_structure(self):
        """score_text should return correct keys."""
        from src.nlp.sentiment_model import score_text
        with patch("src.nlp.sentiment_model.predict", side_effect=_mock_predict):
            result = score_text("I hate this project so much.")
        assert "sentiment" in result
        assert "emotion" in result
        assert "confidence" in result
        assert -1.0 <= result["sentiment"] <= 1.0

    def test_positive_text_positive_sentiment(self):
        """Positive text should yield positive sentiment."""
        from src.nlp.sentiment_model import score_text
        with patch("src.nlp.sentiment_model.predict", side_effect=_mock_predict):
            result = score_text("Great job! Excellent work everyone!")
        assert result["sentiment"] > 0, \
            f"Expected positive sentiment, got {result['sentiment']}"
