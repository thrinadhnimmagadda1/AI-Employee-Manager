"""
Burnout Risk Predictor
Trains XGBoost on HR Analytics dataset. Provides SHAP-explained predictions.
Produces a burnout risk score 0.0–1.0 per employee.
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
DATA_RAW = Path(os.getenv("DATA_DIR", "data")) / "raw"
MODEL_PATH = MODELS_DIR / "burnout_model.pkl"
RANDOM_STATE = 42

# Features used for training (14 behavioral features)
FEATURE_COLUMNS = [
    "avg_response_hours",
    "after_hours_count",
    "message_count",
    "avg_message_length",
    "participation_rate",
    "manager_comm_ratio",
    "sentiment_score",
    "sentiment_velocity",
    "graph_centrality",
    "isolation_score",
    "response_rate",
    "initiation_ratio",
    "thread_depth",
    # HR Analytics proxy features (when available)
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "work_accident",
    "promotion_last_5years",
]

# Columns definitely present in HR Analytics CSV
HR_FEATURE_COLUMNS = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "work_accident",
    "promotion_last_5years",
]
HR_TARGET = "left"


# ─────────────────────────────────────────────────────────────
# Training on HR Analytics dataset
# ─────────────────────────────────────────────────────────────

def load_hr_analytics(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load HR Analytics dataset from Kaggle CSV."""
    csv_path = csv_path or DATA_RAW / "hr_analytics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HR Analytics CSV not found at {csv_path}. "
            "Download from: https://www.kaggle.com/datasets/giripujar/hr-analytics"
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def train(csv_path: Optional[Path] = None) -> None:
    """
    Train XGBoost burnout predictor on HR Analytics dataset.
    Saves trained model + SHAP explainer to models/burnout_model.pkl.
    """
    logger.info("Loading HR Analytics dataset …")
    df = load_hr_analytics(csv_path)

    feature_cols = [c for c in HR_FEATURE_COLUMNS if c in df.columns]
    if HR_TARGET not in df.columns:
        raise ValueError(f"Target column '{HR_TARGET}' not found in HR Analytics CSV.")

    # Encode categorical columns
    cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[feature_cols].fillna(0.0).astype(float)
    y = df[HR_TARGET].astype(int)

    logger.info(f"Dataset: {len(X):,} rows | Features: {feature_cols} | Class balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Handle class imbalance with SMOTE
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {len(X_train_res):,} training samples")
    except Exception as exc:
        logger.warning(f"SMOTE failed ({exc}), using original training data.")
        X_train_res, y_train_res = X_train, y_train

    base_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    logger.info("Training XGBoost burnout model …")
    base_model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Calibrate for reliable probability outputs using cross-validation on training data
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    calibrated.fit(X_train_res, y_train_res)

    # Evaluation on held-out test set
    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    logger.info(
        f"Burnout Model — Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
        f"Recall: {rec:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}"
    )
    logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Build SHAP explainer on the underlying XGBoost model
    explainer = shap.TreeExplainer(base_model)

    # Persist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": calibrated,
        "base_model": base_model,
        "explainer": explainer,
        "feature_columns": feature_cols,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        },
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Burnout model saved to {MODEL_PATH}")


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
            f"Burnout model not found at {MODEL_PATH}. Run train() first."
        )
    with open(MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)
    return _artifact


def predict(features: dict) -> dict:
    """
    Predict burnout risk for a single employee.

    Args:
        features: Dict containing behavioral feature values.

    Returns:
        {
            'burnout_risk':  float (0.0–1.0),
            'top_reasons':   list of top-3 SHAP factor dicts,
            'shap_values':   full SHAP dict for all features,
        }
    """
    artifact = _load_artifact()
    model = artifact["model"]
    explainer = artifact["explainer"]
    feature_cols = artifact["feature_columns"]

    row = pd.DataFrame([{col: features.get(col, 0.0) for col in feature_cols}])

    try:
        prob = float(model.predict_proba(row)[0][1])
    except Exception as exc:
        logger.error(f"Burnout prediction error: {exc}")
        prob = 0.0

    # SHAP explanation
    try:
        shap_vals = explainer.shap_values(row)
        if isinstance(shap_vals, list):
            shap_array = shap_vals[1][0]  # class 1
        else:
            shap_array = shap_vals[0]

        shap_dict = {col: round(float(v), 4) for col, v in zip(feature_cols, shap_array)}

        # Top 3 contributing factors (absolute magnitude)
        sorted_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_reasons = [
            {
                "feature": feat,
                "impact": val,
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            }
            for feat, val in sorted_factors[:3]
        ]
    except Exception as exc:
        logger.debug(f"SHAP computation error: {exc}")
        shap_dict = {}
        top_reasons = []

    return {
        "burnout_risk": round(prob, 4),
        "top_reasons": top_reasons,
        "shap_values": shap_dict,
    }


def predict_batch(feature_list: list[dict]) -> list[dict]:
    """Batch prediction for multiple employees."""
    return [predict(f) for f in feature_list]


if __name__ == "__main__":
    train()
