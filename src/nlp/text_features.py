"""
Text Feature Extractor
Detects linguistic signals associated with burnout and disengagement:
hedging, urgency, passive-aggression, and disengagement patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Pattern definitions
# ─────────────────────────────────────────────────────────────

_HEDGING = re.compile(
    r"\b(I think|I guess|maybe|perhaps|possibly|probably|sort of|kind of|"
    r"not sure|I wonder|might be|could be|it seems|I feel like)\b",
    re.IGNORECASE,
)

_URGENCY = re.compile(
    r"\b(ASAP|urgent|urgently|immediately|right away|as soon as possible|"
    r"critical|deadline|overdue|NOW|time sensitive|no later than)\b",
    re.IGNORECASE,
)

_PASSIVE_AGGRESSIVE = re.compile(
    r"\b(fine\.|whatever\.|as you wish|noted\.|per my (last )?email|"
    r"as I mentioned|moving forward|going forward|at your earliest convenience|"
    r"just wanted to follow up|friendly reminder|hope this helps|"
    r"thanks in advance|no worries|no problem)\b",
    re.IGNORECASE,
)

_DISENGAGEMENT = re.compile(
    r"^(ok|okay|sure|yep|yes|no|k|got it|noted|understood|fine|alright|"
    r"will do|sounds good|agreed|makes sense|thanks|thx|ty|lol)\.?$",
    re.IGNORECASE,
)

_NEGATIVE_EMOTION = re.compile(
    r"\b(frustrated|exhausted|overwhelmed|burned out|burned-out|"
    r"stressed|anxious|worried|scared|angry|annoyed|disappointed|"
    r"depressed|hopeless|unmotivated|disengaged|checked out|"
    r"can't cope|can't handle|falling behind)\b",
    re.IGNORECASE,
)

_EXCLAMATION = re.compile(r"!")
_ALL_CAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")


# ─────────────────────────────────────────────────────────────
# Feature dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class TextFeatures:
    hedging_count: int = 0
    urgency_count: int = 0
    passive_aggressive_count: int = 0
    is_disengaged_reply: bool = False
    negative_emotion_count: int = 0
    exclamation_count: int = 0
    all_caps_word_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    hedging_rate: float = 0.0    # hedging_count / word_count
    urgency_rate: float = 0.0
    pa_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "hedging_count": self.hedging_count,
            "urgency_count": self.urgency_count,
            "passive_aggressive_count": self.passive_aggressive_count,
            "is_disengaged_reply": int(self.is_disengaged_reply),
            "negative_emotion_count": self.negative_emotion_count,
            "exclamation_count": self.exclamation_count,
            "all_caps_word_count": self.all_caps_word_count,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "avg_sentence_length": self.avg_sentence_length,
            "hedging_rate": self.hedging_rate,
            "urgency_rate": self.urgency_rate,
            "pa_rate": self.pa_rate,
        }


# ─────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────

def extract_features(text: str) -> TextFeatures:
    """
    Extract linguistic features from a single message.

    Args:
        text: Raw message string.

    Returns:
        TextFeatures dataclass with all computed features.
    """
    if not isinstance(text, str) or not text.strip():
        return TextFeatures()

    text_stripped = text.strip()
    words = text_stripped.split()
    word_count = len(words)

    # Sentence splitting (rough)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text_stripped) if s.strip()]
    sentence_count = max(len(sentences), 1)
    avg_sentence_length = word_count / sentence_count

    hedging_count = len(_HEDGING.findall(text_stripped))
    urgency_count = len(_URGENCY.findall(text_stripped))
    pa_count = len(_PASSIVE_AGGRESSIVE.findall(text_stripped))
    neg_count = len(_NEGATIVE_EMOTION.findall(text_stripped))
    exclamation_count = len(_EXCLAMATION.findall(text_stripped))
    caps_count = len(_ALL_CAPS_WORD.findall(text_stripped))

    # Disengagement: short replies (≤5 words) matching known one-word patterns
    is_disengaged = bool(_DISENGAGEMENT.match(text_stripped)) or word_count <= 3

    hedge_rate = hedging_count / word_count if word_count > 0 else 0.0
    urgency_rate = urgency_count / word_count if word_count > 0 else 0.0
    pa_rate = pa_count / word_count if word_count > 0 else 0.0

    return TextFeatures(
        hedging_count=hedging_count,
        urgency_count=urgency_count,
        passive_aggressive_count=pa_count,
        is_disengaged_reply=is_disengaged,
        negative_emotion_count=neg_count,
        exclamation_count=exclamation_count,
        all_caps_word_count=caps_count,
        word_count=word_count,
        sentence_count=sentence_count,
        avg_sentence_length=round(avg_sentence_length, 2),
        hedging_rate=round(hedge_rate, 4),
        urgency_rate=round(urgency_rate, 4),
        pa_rate=round(pa_rate, 4),
    )


def extract_features_batch(texts: list[str]) -> list[TextFeatures]:
    """Batch extraction for a list of messages."""
    return [extract_features(t) for t in texts]


def aggregate_weekly_features(features: list[TextFeatures]) -> dict:
    """
    Aggregate a list of per-message features into a weekly summary.

    Returns:
        Dict of aggregated feature values suitable for feature_engineering.py.
    """
    if not features:
        return TextFeatures().to_dict()

    n = len(features)
    agg = {
        "avg_hedging_rate": sum(f.hedging_rate for f in features) / n,
        "avg_urgency_rate": sum(f.urgency_rate for f in features) / n,
        "avg_pa_rate": sum(f.pa_rate for f in features) / n,
        "disengaged_reply_rate": sum(f.is_disengaged_reply for f in features) / n,
        "total_negative_emotion": sum(f.negative_emotion_count for f in features),
        "total_exclamations": sum(f.exclamation_count for f in features),
        "avg_word_count": sum(f.word_count for f in features) / n,
        "avg_sentence_length": sum(f.avg_sentence_length for f in features) / n,
    }
    return {k: round(v, 4) for k, v in agg.items()}
