"""
Attrition / Flight Risk Predictor
Trains XGBoost on HR Analytics dataset.
Outputs calibrated 30/60/90-day attrition probability with SHAP.
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
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
DATA_RAW = Path(os.getenv("DATA_DIR", "data")) / "raw"
MODEL_PATH = MODELS_DIR / "attrition_model.pkl"
RANDOM_STATE = 42

HR_FEATURE_COLUMNS = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "work_accident",
    "promotion_last_5years",
    "salary",
]
HR_TARGET = "left"

# Decay factors to convert base probability to horizon-specific probability
# Logic: 30d ≈ 30% of annual risk, 60d ≈ 50%, 90d ≈ 65%
DECAY_30D = 0.30
DECAY_60D = 0.50
DECAY_90D = 0.65


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(csv_path: Optional[Path] = None) -> None:
    """Train and persist attrition model."""
    csv_path = csv_path or DATA_RAW / "hr_analytics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HR Analytics CSV not found at {csv_path}. "
            "Download from: https://www.kaggle.com/datasets/giripujar/hr-analytics"
        )

    logger.info("Loading HR Analytics dataset for attrition model …")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    feature_cols = [c for c in HR_FEATURE_COLUMNS if c in df.columns]

    # Encode categoricals
    le = LabelEncoder()
    for col in df[feature_cols].select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[feature_cols].fillna(0.0).astype(float)
    y = df[HR_TARGET].astype(int)

    logger.info(f"Attrition dataset: {len(X):,} rows | Features: {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_r, y_train_r = smote.fit_resample(X_train, y_train)
    except Exception as exc:
        logger.warning(f"SMOTE skipped: {exc}")
        X_train_r, y_train_r = X_train, y_train

    base_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    logger.info("Training XGBoost attrition model …")
    base_model.fit(X_train_r, y_train_r, eval_set=[(X_test, y_test)], verbose=False)

    # Calibrate using cross-validation on training data
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    calibrated.fit(X_train_r, y_train_r)

    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    logger.info(f"Attrition Model — Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    explainer = shap.TreeExplainer(base_model)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": calibrated,
        "base_model": base_model,
        "explainer": explainer,
        "feature_columns": feature_cols,
        "metrics": {"accuracy": acc, "f1": f1, "auc": auc},
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Attrition model saved to {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

_artifact: Optional[dict] = None


def _load_artifact() -> dict:
    global _artifact
    if _artifact is not None:
        return _artifact
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Attrition model not found at {MODEL_PATH}. Run train() first."
        )
    with open(MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)
    return _artifact


def predict(features: dict) -> dict:
    """
    Predict attrition risk at 30, 60, and 90-day horizons.

    Args:
        features: Dict with feature values (same as training features).

    Returns:
        {
            'attrition_risk_30d': float,
            'attrition_risk_60d': float,
            'attrition_risk_90d': float,
            'top_reasons': list[dict],
            'shap_values': dict,
        }
    """
    artifact = _load_artifact()
    model = artifact["model"]
    explainer = artifact["explainer"]
    feature_cols = artifact["feature_columns"]

    row = pd.DataFrame([{col: features.get(col, 0.0) for col in feature_cols}])

    try:
        base_prob = float(model.predict_proba(row)[0][1])
    except Exception as exc:
        logger.error(f"Attrition prediction error: {exc}")
        base_prob = 0.0

    # SHAP
    try:
        shap_vals = explainer.shap_values(row)
        shap_array = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        shap_dict = {col: round(float(v), 4) for col, v in zip(feature_cols, shap_array)}
        top_reasons = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_reasons = [
            {"feature": f, "impact": v, "direction": "increases_risk" if v > 0 else "decreases_risk"}
            for f, v in top_reasons
        ]
    except Exception as exc:
        logger.debug(f"SHAP error: {exc}")
        shap_dict = {}
        top_reasons = []

    return {
        "attrition_risk_30d": round(min(base_prob * DECAY_30D, 1.0), 4),
        "attrition_risk_60d": round(min(base_prob * DECAY_60D, 1.0), 4),
        "attrition_risk_90d": round(min(base_prob * DECAY_90D, 1.0), 4),
        "top_reasons": top_reasons,
        "shap_values": shap_dict,
    }


if __name__ == "__main__":
    train()
