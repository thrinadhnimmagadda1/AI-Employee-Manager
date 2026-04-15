"""
BERT Emotion Classifier
Fine-tunes bert-base-uncased on GoEmotions (filtered to 7 workplace emotions).
Provides predict() for inference.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_SAVE_PATH = MODELS_DIR / "emotion_classifier"
BASE_MODEL = "bert-base-uncased"

# 7 workplace-relevant emotions mapped from GoEmotions label indices
# GoEmotions 28 labels (0-indexed):
#   0=admiration, 1=amusement, 2=anger, 3=annoyance, 4=approval,
#   5=caring, 6=confusion, 7=curiosity, 8=desire, 9=disappointment,
#   10=disapproval, 11=disgust, 12=embarrassment, 13=excitement,
#   14=fear, 15=gratitude, 16=grief, 17=joy, 18=love,
#   19=nervousness, 20=optimism, 21=pride, 22=realization,
#   23=relief, 24=remorse, 25=sadness, 26=surprise, 27=neutral
WORKPLACE_EMOTION_MAP = {
    "joy": [1, 13, 15, 17, 20],        # amusement, excitement, gratitude, joy, optimism
    "frustration": [3, 9, 10],         # annoyance, disappointment, disapproval
    "anxiety": [6, 14, 19],            # confusion, fear, nervousness
    "anger": [2, 11],                  # anger, disgust
    "sadness": [16, 25, 24],           # grief, sadness, remorse
    "disgust": [11, 10],               # disgust, disapproval
    "neutral": [27],                   # neutral
}
WORKPLACE_EMOTIONS = list(WORKPLACE_EMOTION_MAP.keys())
NUM_LABELS = len(WORKPLACE_EMOTIONS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(WORKPLACE_EMOTIONS)}

MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# Label mapping from 28-class → 7-class
# ─────────────────────────────────────────────────────────────

def _map_go_emotions_label(original_labels: list[int]) -> int:
    """
    Convert a list of GoEmotions label indices to a single
    7-class workplace emotion index.
    Priority order: anger > frustration > anxiety > sadness > disgust > joy > neutral
    """
    priority = ["anger", "frustration", "anxiety", "sadness", "disgust", "joy", "neutral"]
    for emotion in priority:
        for go_label in WORKPLACE_EMOTION_MAP[emotion]:
            if go_label in original_labels:
                return EMOTION_TO_IDX[emotion]
    return EMOTION_TO_IDX["neutral"]


# ─────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────

def prepare_go_emotions_dataset() -> tuple:
    """Load GoEmotions and remap labels to 7 workplace emotions."""
    logger.info("Loading GoEmotions dataset from HuggingFace …")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

    def remap(example: dict) -> dict:
        example["mapped_label"] = _map_go_emotions_label(example["labels"])
        return example

    dataset = dataset.map(remap)
    logger.info(f"Train size: {len(dataset['train']):,} | Validation size: {len(dataset['validation']):,}")
    return dataset["train"], dataset["validation"]


# ─────────────────────────────────────────────────────────────
# Fine-tuning entry point (delegates to fine_tuning.py)
# ─────────────────────────────────────────────────────────────

def fine_tune_emotion_classifier(
    num_epochs: int = 3,
    output_dir: Optional[Path] = None,
) -> dict:
    """Full fine-tuning pipeline. Imports fine_tuning module. Returns eval metrics."""
    from src.nlp.fine_tuning import run_fine_tuning  # avoid circular at module level

    output_dir = output_dir or MODEL_SAVE_PATH
    train_ds, val_ds = prepare_go_emotions_dataset()
    return run_fine_tuning(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_labels=NUM_LABELS,
        label_col="mapped_label",
        output_dir=output_dir,
        num_epochs=num_epochs,
    )


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

_classifier_pipeline = None  # module-level singleton


def _load_inference_pipeline() -> object:
    global _classifier_pipeline
    if _classifier_pipeline is not None:
        return _classifier_pipeline

    if MODEL_SAVE_PATH.exists():
        model_path = str(MODEL_SAVE_PATH)
        logger.info(f"Loading fine-tuned emotion classifier from {model_path}")
    else:
        logger.warning(
            "Fine-tuned model not found. Using base BERT — run fine_tune_emotion_classifier() first."
        )
        model_path = BASE_MODEL

    _classifier_pipeline = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=0 if DEVICE == "cuda" else -1,
        top_k=None,  # return all labels
        max_length=MAX_LEN,
        truncation=True,
    )
    return _classifier_pipeline


def predict(text: str) -> dict[str, float]:
    """
    Predict workplace emotion probabilities for a given text.

    Args:
        text: Input text string.

    Returns:
        Dictionary mapping each of 7 emotions to a probability (0.0–1.0).
        All values sum to ~1.0.

    Example:
        >>> predict("I'm overwhelmed with deadlines and can't sleep")
        {'joy': 0.02, 'frustration': 0.55, 'anxiety': 0.30, ...}
    """
    if not text or not text.strip():
        return {e: (1.0 if e == "neutral" else 0.0) for e in WORKPLACE_EMOTIONS}

    clf = _load_inference_pipeline()
    try:
        raw_results = clf(text[:512])[0]  # truncate to avoid token limit issues
    except Exception as exc:
        logger.warning(f"Emotion classifier inference error: {exc}")
        return {e: (1.0 if e == "neutral" else 0.0) for e in WORKPLACE_EMOTIONS}

    # raw_results is a list of {label: "LABEL_X", score: float}
    scores: dict[str, float] = {e: 0.0 for e in WORKPLACE_EMOTIONS}
    for item in raw_results:
        label_str = item["label"]  # e.g. "LABEL_0" → index 0
        try:
            idx = int(label_str.split("_")[-1])
            emotion = WORKPLACE_EMOTIONS[idx]
            scores[emotion] = float(item["score"])
        except (ValueError, IndexError):
            continue

    # Normalize to sum to 1
    total = sum(scores.values())
    if total > 0:
        scores = {e: v / total for e, v in scores.items()}

    return scores


def predict_batch(texts: list[str]) -> list[dict[str, float]]:
    """Batch version of predict() for efficiency."""
    return [predict(t) for t in texts]


def dominant_emotion(scores: dict[str, float]) -> str:
    """Return the emotion with the highest probability."""
    return max(scores, key=scores.get)
