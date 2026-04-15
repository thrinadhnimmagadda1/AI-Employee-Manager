#!/usr/bin/env python3
"""
CogniTeam — Model Training Orchestrator
=========================================
Trains all 4 prediction models in sequence and saves them to models/.

Usage:
    python scripts/train_all_models.py              # train all 4
    python scripts/train_all_models.py --skip-bert  # skip emotion BERT (fast ML only)
    python scripts/train_all_models.py --only burnout
    python scripts/train_all_models.py --only attrition
    python scripts/train_all_models.py --only conflict
    python scripts/train_all_models.py --only emotion
    python scripts/train_all_models.py --epochs 1   # fewer epochs for quick test

Models trained:
    1. Emotion Classifier  (BERT fine-tuned on GoEmotions → 7 workplace emotions)
    2. Burnout Predictor   (XGBoost + SMOTE + SHAP on HR Analytics)
    3. Attrition Model     (XGBoost multi-horizon 30/60/90d on HR Analytics)
    4. Conflict Detector   (XGBoost on HR Analytics proxy labels)

Output:
    models/emotion_classifier/    ← BERT model + tokenizer
    models/burnout_model.pkl
    models/attrition_model.pkl
    models/conflict_model.pkl
    models/metrics.json           ← all metrics for /api/info endpoint
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# ── repo root on sys.path ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_RAW   = REPO_ROOT / "data" / "raw"
MODELS_DIR = REPO_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ── pretty print helpers ─────────────────────────────────────────────────────
_SEP_WIDE  = "═" * 56
_SEP_THIN  = "─" * 56


def _header(title: str) -> None:
    print(f"\n{'━' * 56}")
    print(f"  {title}")
    print(f"{'━' * 56}")


def _ok(msg: str) -> None:
    print(f"  ✅  {msg}", flush=True)


def _err(msg: str) -> None:
    # Always use stdout so output order is preserved (stderr is unbuffered and jumps the queue)
    print(f"  ❌  {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"  ⚠️   {msg}", flush=True)


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ── pre-flight checks ────────────────────────────────────────────────────────

def _check_data_files(need_hr: bool, need_goemotions: bool) -> bool:
    """Return False (and print errors) if any required data file is missing."""
    ok = True

    hr_path = DATA_RAW / "hr_analytics.csv"
    ge_train = DATA_RAW / "go_emotions_train.csv"
    ge_val   = DATA_RAW / "go_emotions_val.csv"

    if need_hr and not hr_path.exists():
        _err(f"HR Analytics CSV not found: {hr_path}")
        _err("  Run:  python scripts/download_data.py --only hr")
        ok = False

    if need_goemotions and not ge_train.exists():
        _err(f"GoEmotions train CSV not found: {ge_train}")
        _err("  Run:  python scripts/download_data.py --only goemotions")
        ok = False

    if need_goemotions and not ge_val.exists():
        _err(f"GoEmotions val CSV not found: {ge_val}")
        _err("  Run:  python scripts/download_data.py --only goemotions")
        ok = False

    return ok


def _check_torch() -> bool:
    """Check that torch + transformers are importable."""
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        return True
    except ImportError as exc:
        _err(f"torch/transformers not installed: {exc}")
        _err("  Install with:  pip install torch transformers datasets")
        _err("  (large download ~2GB — GPU recommended for BERT fine-tuning)")
        return False


# ── Model 1: Emotion Classifier ──────────────────────────────────────────────

def train_emotion_classifier(num_epochs: int = 3) -> dict | None:
    """Fine-tune BERT on GoEmotions (7 workplace emotion labels)."""
    _header("Model 1 / 4 — Emotion Classifier (BERT)")
    print(f"  Base model:  bert-base-uncased")
    print(f"  Epochs:      {num_epochs}")
    print(f"  Output:      models/emotion_classifier/")
    print(f"  ⏱  Expected: ~20-40 min CPU  |  ~5-10 min GPU")
    print()

    t0 = time.time()

    from src.nlp.emotion_classifier import fine_tune_emotion_classifier  # noqa: PLC0415

    try:
        metrics = fine_tune_emotion_classifier(num_epochs=num_epochs)
    except Exception as exc:
        _err(f"Emotion classifier training failed: {exc}")
        import traceback
        traceback.print_exc()
        return None

    elapsed = time.time() - t0
    f1 = metrics.get("f1_macro", 0.0) if metrics else 0.0
    acc = metrics.get("accuracy", 0.0) if metrics else 0.0

    _ok(f"Emotion Classifier trained in {_elapsed(elapsed)}")
    print(f"     F1 (macro):   {f1 * 100:.1f}%")
    print(f"     Accuracy:     {acc * 100:.1f}%")
    print(f"     Labels:       joy · frustration · anxiety · anger · sadness · disgust · neutral")

    return {
        "model": "emotion_classifier",
        "f1_macro": round(f1, 4),
        "accuracy": round(acc, 4),
        "epochs": num_epochs,
        "elapsed_s": round(elapsed, 1),
        "save_path": "models/emotion_classifier/",
    }


# ── Model 2: Burnout Predictor ───────────────────────────────────────────────

def train_burnout_predictor() -> dict | None:
    """XGBoost burnout risk model on HR Analytics."""
    _header("Model 2 / 4 — Burnout Predictor (XGBoost)")
    hr_path = DATA_RAW / "hr_analytics.csv"
    import pandas as pd  # noqa: PLC0415
    n_rows = sum(1 for _ in open(hr_path)) - 1
    print(f"  Dataset:     {hr_path.name}  ({n_rows:,} rows)")
    print(f"  Features:    satisfaction_level, last_evaluation, number_project,")
    print(f"               average_montly_hours, time_spend_company,")
    print(f"               work_accident, promotion_last_5years")
    print(f"  Target:      left  (1 = attrited = burnout proxy)")
    print(f"  Balancing:   SMOTE oversampling")
    print(f"  Output:      models/burnout_model.pkl  +  SHAP explainer")
    print()

    t0 = time.time()

    from src.ml.burnout_predictor import train, MODEL_PATH  # noqa: PLC0415

    try:
        train(csv_path=hr_path)
    except Exception as exc:
        _err(f"Burnout predictor training failed: {exc}")
        import traceback
        traceback.print_exc()
        return None

    elapsed = time.time() - t0

    # Load saved artifact to read metrics
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    m = artifact["metrics"]

    _ok(f"Burnout Predictor trained in {_elapsed(elapsed)}")
    print(f"     Accuracy:   {m['accuracy'] * 100:.1f}%")
    print(f"     Precision:  {m['precision'] * 100:.1f}%")
    print(f"     Recall:     {m['recall'] * 100:.1f}%")
    print(f"     F1 Score:   {m['f1'] * 100:.1f}%")
    print(f"     AUC-ROC:    {m['auc'] * 100:.1f}%")
    print(f"     Features:   {artifact['feature_columns']}")

    return {
        "model": "burnout_predictor",
        "accuracy": round(m["accuracy"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "auc": round(m["auc"], 4),
        "elapsed_s": round(elapsed, 1),
        "save_path": str(MODEL_PATH.resolve().relative_to(REPO_ROOT)),
    }


# ── Model 3: Attrition Model ─────────────────────────────────────────────────

def train_attrition_model() -> dict | None:
    """XGBoost 30/60/90-day attrition risk model on HR Analytics."""
    _header("Model 3 / 4 — Attrition Model (XGBoost)")
    hr_path = DATA_RAW / "hr_analytics.csv"
    n_rows = sum(1 for _ in open(hr_path)) - 1
    print(f"  Dataset:     {hr_path.name}  ({n_rows:,} rows)")
    print(f"  Features:    satisfaction_level, last_evaluation, number_project,")
    print(f"               average_montly_hours, time_spend_company,")
    print(f"               work_accident, promotion_last_5years, salary")
    print(f"  Target:      left  (1 = attrited)")
    print(f"  Horizons:    30d (×0.30) · 60d (×0.50) · 90d (×0.65)")
    print(f"  Output:      models/attrition_model.pkl  +  SHAP explainer")
    print()

    t0 = time.time()

    from src.ml.attrition_model import train, MODEL_PATH  # noqa: PLC0415

    try:
        train(csv_path=hr_path)
    except Exception as exc:
        _err(f"Attrition model training failed: {exc}")
        import traceback
        traceback.print_exc()
        return None

    elapsed = time.time() - t0

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    m = artifact["metrics"]

    _ok(f"Attrition Model trained in {_elapsed(elapsed)}")
    print(f"     Accuracy:   {m['accuracy'] * 100:.1f}%")
    print(f"     F1 Score:   {m['f1'] * 100:.1f}%")
    print(f"     AUC-ROC:    {m['auc'] * 100:.1f}%")
    print(f"     Features:   {artifact['feature_columns']}")

    return {
        "model": "attrition_model",
        "accuracy": round(m["accuracy"], 4),
        "f1": round(m["f1"], 4),
        "auc": round(m["auc"], 4),
        "elapsed_s": round(elapsed, 1),
        "save_path": str(MODEL_PATH.resolve().relative_to(REPO_ROOT)),
    }


# ── Model 4: Conflict Detector ───────────────────────────────────────────────

def _build_conflict_proxy_dataset(hr_path: Path) -> "pd.DataFrame":
    """
    Build a conflict-labeled DataFrame from HR Analytics using domain heuristics.

    Label logic (proxy for real conflict labels):
      conflict = 1  when  satisfaction_level < 0.30  AND  average_montly_hours > 250
      conflict = 0  otherwise

    HR Analytics columns are mapped to the pair-feature space expected by
    conflict_detector.PAIR_FEATURE_COLUMNS so the existing train() function
    can consume the data without modification.

    Mapping rationale:
      avg_sentiment      ← satisfaction_level (direct proxy)
      avg_response_h     ← average_montly_hours / 22 (monthly hrs → daily response proxy)
      message_count      ← number_project × 8 (projects → weekly msg volume)
      frequency_ratio    ← last_evaluation (performance ↔ communication engagement)
      response_time_trend← (1 − last_evaluation) × 3 (poor perf → worsening response)
      sentiment_trend    ← satisfaction_level − 0.5 (centered, slope proxy)
      pa_score           ← (1 − satisfaction_level) × 0.8 (dissatisfied → passive-aggression)
      relationship_health← satisfaction_level × (1 − work_accident × 0.4)
    """
    import pandas as pd  # noqa: PLC0415
    import numpy as np   # noqa: PLC0415

    df = pd.read_csv(hr_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = [
        "satisfaction_level", "last_evaluation", "number_project",
        "average_montly_hours", "time_spend_company", "work_accident",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"HR Analytics CSV missing columns: {missing}")

    proxy = pd.DataFrame()
    proxy["avg_sentiment"]       = df["satisfaction_level"].clip(0, 1)
    proxy["avg_response_h"]      = (df["average_montly_hours"] / 22.0).clip(0.5, 48)
    proxy["message_count"]       = (df["number_project"] * 8).clip(1, 200).astype(int)
    proxy["frequency_ratio"]     = df["last_evaluation"].clip(0, 1)
    proxy["response_time_trend"] = ((1 - df["last_evaluation"]) * 3).clip(0, 5)
    proxy["sentiment_trend"]     = (df["satisfaction_level"] - 0.5).clip(-0.5, 0.5)
    proxy["pa_score"]            = ((1 - df["satisfaction_level"]) * 0.8).clip(0, 1)
    proxy["relationship_health"] = (
        df["satisfaction_level"] * (1 - df["work_accident"].clip(0, 1) * 0.4)
    ).clip(0, 1)

    # Proxy label: chronic overwork + very low satisfaction = conflict indicator
    proxy["label"] = (
        (df["satisfaction_level"] < 0.30) & (df["average_montly_hours"] > 250)
    ).astype(int)

    pos = proxy["label"].sum()
    neg = (proxy["label"] == 0).sum()
    print(f"  Proxy labels: {pos:,} conflict (1) · {neg:,} non-conflict (0)  "
          f"({pos / len(proxy) * 100:.1f}% positive rate)")

    return proxy


def train_conflict_detector() -> dict | None:
    """XGBoost conflict risk model using HR Analytics as proxy training data."""
    _header("Model 4 / 4 — Conflict Detector (XGBoost, HR proxy labels)")
    hr_path = DATA_RAW / "hr_analytics.csv"
    n_rows = sum(1 for _ in open(hr_path)) - 1
    print(f"  Dataset:     {hr_path.name}  ({n_rows:,} rows)")
    print(f"  Label:       satisfaction_level < 0.30 AND average_montly_hours > 250  →  conflict=1")
    print(f"  Features:    8 pair-level proxies mapped from HR Analytics columns")
    print(f"  Output:      models/conflict_model.pkl  +  SHAP explainer")
    print(f"  Note:        Will retrain on real Enron comm_graph data once loaded")
    print()

    t0 = time.time()

    try:
        proxy_df = _build_conflict_proxy_dataset(hr_path)
    except Exception as exc:
        _err(f"Failed to build conflict proxy dataset: {exc}")
        return None

    from src.ml.conflict_detector import train, MODEL_PATH  # noqa: PLC0415

    try:
        train(labeled_data=proxy_df)
    except Exception as exc:
        _err(f"Conflict detector training failed: {exc}")
        import traceback
        traceback.print_exc()
        return None

    elapsed = time.time() - t0

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    m = artifact["metrics"]

    _ok(f"Conflict Detector trained in {_elapsed(elapsed)}")
    print(f"     F1 Score:   {m['f1'] * 100:.1f}%")
    print(f"     AUC-ROC:    {m['auc'] * 100:.1f}%")
    print(f"     Training:   HR Analytics proxy  (not synthetic random data)")
    print(f"     ℹ️  Will retrain on Enron comm_graph data once enron_loader runs")

    return {
        "model": "conflict_detector",
        "f1": round(m["f1"], 4),
        "auc": round(m["auc"], 4),
        "training_data": "HR Analytics proxy labels",
        "label_rule": "satisfaction_level < 0.30 AND average_montly_hours > 250",
        "elapsed_s": round(elapsed, 1),
        "save_path": str(MODEL_PATH.resolve().relative_to(REPO_ROOT)),
    }


# ── Final summary ─────────────────────────────────────────────────────────────

def _print_final_report(results: dict) -> None:
    print(f"\n{_SEP_WIDE}")
    print("  ALL MODELS TRAINED SUCCESSFULLY")
    print(_SEP_WIDE)

    emotion   = results.get("emotion_classifier")
    burnout   = results.get("burnout_predictor")
    attrition = results.get("attrition_model")
    conflict  = results.get("conflict_detector")

    rows = [
        ("Emotion Classifier", "F1 (macro)",
         f"{emotion['f1_macro']*100:.1f}%" if emotion else "⚠️  skipped"),
        ("Burnout Predictor",  "AUC-ROC",
         f"{burnout['auc']*100:.1f}%" if burnout else "⚠️  skipped"),
        ("Attrition Model",    "AUC-ROC",
         f"{attrition['auc']*100:.1f}%" if attrition else "⚠️  skipped"),
        ("Conflict Detector",  "F1 Score",
         f"{conflict['f1']*100:.1f}%" if conflict else "⚠️  skipped"),
    ]

    for name, metric, value in rows:
        print(f"  {name:<26}  {metric:<12}  =  {value}")

    print()
    print(f"  Models saved to  models/")
    print(f"  Metrics saved to models/metrics.json")
    print()
    print(f"  Start the API and all predictions are now live:")
    print(f"    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
    print(_SEP_WIDE)


# ── metrics.json ──────────────────────────────────────────────────────────────

def _save_metrics(results: dict) -> None:
    """Save all training metrics to models/metrics.json for the API."""
    import datetime  # noqa: PLC0415

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "models": results,
    }

    # Merge with any existing metrics so we don't lose previous runs
    if METRICS_PATH.exists():
        try:
            existing = json.loads(METRICS_PATH.read_text())
            existing_models = existing.get("models", {})
            existing_models.update(results)
            payload["models"] = existing_models
        except Exception:
            pass

    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    _ok(f"Metrics written to {METRICS_PATH.relative_to(REPO_ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train all CogniTeam ML models")
    parser.add_argument(
        "--only",
        choices=["emotion", "burnout", "attrition", "conflict"],
        help="Train only one specific model",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip BERT fine-tuning (train only the 3 XGBoost models)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of BERT fine-tuning epochs (default: 3)",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       CogniTeam — Model Training Orchestrator       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Repo root:   {REPO_ROOT}")
    print(f"  Models dir:  {MODELS_DIR}")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    t_total = time.time()
    only = args.only

    # ── Single pre-flight check before any training starts ──────────────────
    print(f"{'─' * 56}")
    print("  Pre-flight checks")
    print(f"{'─' * 56}")
    need_hr = only in (None, "burnout", "attrition", "conflict")
    need_ge = only in (None, "emotion")
    run_bert = (only in (None, "emotion")) and not args.skip_bert

    pre_ok = True
    if need_hr:
        hr_path = DATA_RAW / "hr_analytics.csv"
        if hr_path.exists():
            n = sum(1 for _ in open(hr_path)) - 1
            _ok(f"data/raw/hr_analytics.csv  ({n:,} rows)")
        else:
            _err(f"data/raw/hr_analytics.csv  NOT FOUND")
            _err("    Fix:  python scripts/download_data.py --only hr")
            pre_ok = False

    if run_bert:
        ge_train = DATA_RAW / "go_emotions_train.csv"
        ge_val   = DATA_RAW / "go_emotions_val.csv"
        if ge_train.exists() and ge_val.exists():
            _ok("data/raw/go_emotions_train.csv  ✓")
            _ok("data/raw/go_emotions_val.csv  ✓")
        else:
            _err("GoEmotions CSVs NOT FOUND")
            _err("    Fix:  python scripts/download_data.py --only goemotions")
            pre_ok = False

        if not _check_torch():
            pre_ok = False

    if not pre_ok:
        print()
        print("  Resolve the missing files/packages above and re-run.")
        sys.exit(1)

    print()

    # ── Model 1: Emotion Classifier ──
    if only in (None, "emotion") and not args.skip_bert:
        result = train_emotion_classifier(num_epochs=args.epochs)
        if result:
            results["emotion_classifier"] = result
    elif args.skip_bert and only is None:
        _warn("Skipping Emotion Classifier (--skip-bert)")

    # ── Model 2: Burnout Predictor ──
    if only in (None, "burnout"):
        result = train_burnout_predictor()
        if result:
            results["burnout_predictor"] = result

    # ── Model 3: Attrition Model ──
    if only in (None, "attrition"):
        result = train_attrition_model()
        if result:
            results["attrition_model"] = result

    # ── Model 4: Conflict Detector ──
    if only in (None, "conflict"):
        result = train_conflict_detector()
        if result:
            results["conflict_detector"] = result

    if not results:
        print("\n  No models were trained. Check errors above.")
        sys.exit(1)

    _save_metrics(results)

    total_elapsed = time.time() - t_total
    print(f"\n  Total training time: {_elapsed(total_elapsed)}")

    _print_final_report(results)


if __name__ == "__main__":
    main()
