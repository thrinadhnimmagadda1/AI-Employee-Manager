"""
Conflict Detector
Trains XGBoost on communication pair features to predict conflict probability.
Scans all employee pairs weekly and returns pairs above threshold.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_PATH = MODELS_DIR / "conflict_model.pkl"
RANDOM_STATE = 42
CONFLICT_THRESHOLD = 0.6

PAIR_FEATURE_COLUMNS = [
    "response_time_trend",        # slope of response time over last 4 weeks
    "sentiment_trend",            # slope of sentiment over last 4 weeks
    "frequency_ratio",            # current_week_msgs / prev_4_week_avg
    "pa_score",                   # passive-aggressive language detection score
    "avg_sentiment",              # average sentiment of this pair
    "avg_response_h",             # average response time
    "relationship_health",        # current relationship health score
    "message_count",              # total messages in current week
]


# ─────────────────────────────────────────────────────────────
# Synthetic training data (for demo — replace with labeled pairs)
# ─────────────────────────────────────────────────────────────

def _generate_synthetic_training_data(n: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic labeled pair data for training when no labeled data exists.
    In production, replace this with real annotated conflict labels.
    """
    np.random.seed(RANDOM_STATE)

    # Conflict class (label=1): deteriorating sentiment, slow responses, low health
    n_conflict = n // 4
    conflict = pd.DataFrame({
        "response_time_trend": np.random.uniform(0.5, 5.0, n_conflict),
        "sentiment_trend": np.random.uniform(-0.5, -0.1, n_conflict),
        "frequency_ratio": np.random.uniform(0.1, 0.6, n_conflict),
        "pa_score": np.random.uniform(0.3, 1.0, n_conflict),
        "avg_sentiment": np.random.uniform(-1.0, -0.2, n_conflict),
        "avg_response_h": np.random.uniform(10, 48, n_conflict),
        "relationship_health": np.random.uniform(0.0, 0.35, n_conflict),
        "message_count": np.random.randint(1, 10, n_conflict),
        "label": 1,
    })

    # Healthy class (label=0): improving sentiment, quick responses, high health
    n_healthy = n - n_conflict
    healthy = pd.DataFrame({
        "response_time_trend": np.random.uniform(-1.0, 0.5, n_healthy),
        "sentiment_trend": np.random.uniform(-0.1, 0.5, n_healthy),
        "frequency_ratio": np.random.uniform(0.7, 2.0, n_healthy),
        "pa_score": np.random.uniform(0.0, 0.2, n_healthy),
        "avg_sentiment": np.random.uniform(-0.2, 1.0, n_healthy),
        "avg_response_h": np.random.uniform(0.1, 10, n_healthy),
        "relationship_health": np.random.uniform(0.5, 1.0, n_healthy),
        "message_count": np.random.randint(5, 50, n_healthy),
        "label": 0,
    })

    return pd.concat([conflict, healthy], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(labeled_data: Optional[pd.DataFrame] = None) -> None:
    """
    Train conflict detector.

    Args:
        labeled_data: Optional DataFrame with PAIR_FEATURE_COLUMNS + 'label' column.
                      If None, uses synthetic training data.
    """
    if labeled_data is None:
        logger.info("No labeled pair data provided — using synthetic training data.")
        df = _generate_synthetic_training_data()
    else:
        df = labeled_data.copy()

    X = df[PAIR_FEATURE_COLUMNS].fillna(0.0).astype(float)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    logger.info("Training XGBoost conflict detection model …")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= CONFLICT_THRESHOLD).astype(int)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"Conflict Model — F1: {f1:.4f} | AUC-ROC: {auc:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    explainer = shap.TreeExplainer(model)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "explainer": explainer,
        "feature_columns": PAIR_FEATURE_COLUMNS,
        "threshold": CONFLICT_THRESHOLD,
        "metrics": {"f1": f1, "auc": auc},
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Conflict model saved to {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

_artifact: Optional[dict] = None


def _load_artifact() -> dict:
    global _artifact
    if _artifact is not None:
        return _artifact
    if not MODEL_PATH.exists():
        logger.warning("Conflict model not found — training on synthetic data.")
        train()
    with open(MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)
    return _artifact


def predict_pair(pair_features: dict) -> dict:
    """
    Predict conflict probability for a single employee pair.

    Returns:
        {
            'conflict_risk': float (0.0–1.0),
            'is_conflict': bool,
            'shap_values': dict,
        }
    """
    artifact = _load_artifact()
    model = artifact["model"]
    explainer = artifact["explainer"]
    feature_cols = artifact["feature_columns"]
    threshold = artifact["threshold"]

    row = pd.DataFrame([{col: pair_features.get(col, 0.0) for col in feature_cols}])

    try:
        prob = float(model.predict_proba(row)[0][1])
    except Exception as exc:
        logger.error(f"Conflict prediction error: {exc}")
        prob = 0.0

    try:
        shap_vals = explainer.shap_values(row)
        shap_array = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        shap_dict = {col: round(float(v), 4) for col, v in zip(feature_cols, shap_array)}
    except Exception as exc:
        logger.debug(f"SHAP error: {exc}")
        shap_dict = {}

    return {
        "conflict_risk": round(prob, 4),
        "is_conflict": prob >= threshold,
        "shap_values": shap_dict,
    }


def detect_conflicts(all_pair_features: list[dict]) -> list[dict]:
    """
    Scan all employee pairs and return those above conflict threshold.

    Args:
        all_pair_features: List of pair feature dicts, each containing
                           'employee_a', 'employee_b', + feature columns.

    Returns:
        Filtered list of dicts for pairs where conflict risk >= threshold.
    """
    flagged = []
    for pair in all_pair_features:
        result = predict_pair(pair)
        if result["is_conflict"]:
            flagged.append({
                "employee_a": pair.get("employee_a"),
                "employee_b": pair.get("employee_b"),
                "conflict_risk": result["conflict_risk"],
                "shap_values": result["shap_values"],
            })

    logger.info(
        f"Conflict detection: {len(all_pair_features)} pairs scanned, "
        f"{len(flagged)} conflicts detected (threshold={CONFLICT_THRESHOLD})"
    )
    return flagged


if __name__ == "__main__":
    train()
