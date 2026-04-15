"""
Admin & Compliance API routes
==============================
All endpoints require the `hr` role.

Routes:
  GET  /audit-log            — paginated, filterable GDPR audit trail
  POST /admin/run-pipeline   — manually trigger the weekly analytics pipeline
  POST /admin/train-models   — retrain all ML models from latest data
  GET  /admin/jobs/{job_id}  — poll async job status
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.api.middleware.auth import TokenData, require_role

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")

router = APIRouter(tags=["admin"])

# ── Simple in-memory job store (Redis would replace this in production) ───────
_JOBS: dict[str, dict] = {}


def _get_session() -> Session:
    return Session(create_engine(DATABASE_URL))


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1: GET /audit-log
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/audit-log",
    summary="GDPR audit log",
    description=(
        "Returns a paginated, filterable log of every data-access event recorded by "
        "the `AuditLoggingMiddleware`. Demonstrates GDPR/SOC-2 compliance. "
        "Only accessible by users with the **hr** role.\n\n"
        "Filters:\n"
        "- `user_id` — show actions by a specific internal user ID\n"
        "- `from_date` / `to_date` — ISO date range (e.g. `2024-01-01`)\n"
        "- `action_contains` — substring match on the action field (e.g. `GET /employees`)\n\n"
        "Pagination: `page` (1-based) × `page_size` (max 200)."
    ),
)
async def get_audit_log(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=50, ge=1, le=200, description="Rows per page (max 200)"),
    user_id: Optional[int] = Query(default=None, description="Filter by internal user ID"),
    from_date: Optional[date] = Query(default=None, description="Earliest date inclusive (YYYY-MM-DD)"),
    to_date: Optional[date] = Query(default=None, description="Latest date inclusive (YYYY-MM-DD)"),
    action_contains: Optional[str] = Query(default=None, description="Substring match on action field"),
    _user: TokenData = Depends(require_role("hr")),
) -> dict:
    filters: list[str] = []
    params: dict = {}

    if user_id is not None:
        filters.append("user_id = :user_id")
        params["user_id"] = user_id

    if from_date:
        filters.append("timestamp >= :from_date")
        params["from_date"] = datetime(from_date.year, from_date.month, from_date.day, tzinfo=timezone.utc)

    if to_date:
        filters.append("timestamp < :to_date_excl")
        # inclusive: advance to midnight of the NEXT day
        from datetime import timedelta  # noqa: PLC0415
        next_day = datetime(to_date.year, to_date.month, to_date.day, tzinfo=timezone.utc) + timedelta(days=1)
        params["to_date_excl"] = next_day

    if action_contains:
        filters.append("action ILIKE :action_pat")
        params["action_pat"] = f"%{action_contains}%"

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""
    offset = (page - 1) * page_size
    params["limit"]  = page_size
    params["offset"] = offset

    with _get_session() as session:
        total_row = session.execute(
            text(f"SELECT COUNT(*) FROM audit_log {where_clause}"),
            params,
        ).scalar() or 0

        rows = session.execute(
            text(f"""
                SELECT id, user_id, action, target, timestamp, ip_address
                FROM audit_log
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).fetchall()

    entries = [
        {
            "id":         r[0],
            "user_id":    r[1],
            "action":     r[2],
            "target":     r[3],
            "timestamp":  str(r[4]),
            "ip_address": r[5],
        }
        for r in rows
    ]

    return {
        "total":       total_row,
        "page":        page,
        "page_size":   page_size,
        "total_pages": max(1, -(-total_row // page_size)),  # ceiling division
        "entries":     entries,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Background job helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mark_job(job_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
    _JOBS[job_id].update({
        "status":       status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result":       result,
        "error":        error,
    })


def _run_pipeline_job(job_id: str, week_str: Optional[str]) -> None:
    """Background task that runs the full analytics pipeline for one week."""
    from datetime import timedelta  # noqa: PLC0415

    try:
        if week_str and week_str != "current":
            try:
                week = date.fromisoformat(week_str)
            except ValueError:
                _mark_job(job_id, "failed", error=f"Invalid date format: {week_str}")
                return
        else:
            today = date.today()
            week  = today - timedelta(days=today.weekday())  # current Monday

        result: dict = {"week": str(week)}
        t0 = time.time()

        # Step 1: Feature engineering
        try:
            from src.ml.feature_engineering import compute_features_for_week  # noqa: PLC0415
            n_feat = compute_features_for_week(week)
            result["features_computed"] = n_feat
        except Exception as exc:
            result["feature_error"] = str(exc)

        # Step 2: Graph
        try:
            from src.graph.graph_builder import build_and_persist_weekly_graph  # noqa: PLC0415
            G = build_and_persist_weekly_graph(week)
            result["graph_nodes"] = G.number_of_nodes()
            result["graph_edges"] = G.number_of_edges()
        except Exception as exc:
            result["graph_error"] = str(exc)

        # Step 3: Network analysis
        try:
            from src.graph.network_analyzer import analyze_network_for_week  # noqa: PLC0415
            metrics = analyze_network_for_week(week)
            result["network_analyzed"] = len(metrics)
        except Exception as exc:
            result["network_error"] = str(exc)

        # Step 4: ML predictions → health_scores
        try:
            n_scores = _compute_health_scores_for_week(week)
            result["health_scores_written"] = n_scores
        except Exception as exc:
            result["health_score_error"] = str(exc)

        # Step 5: Alerts
        try:
            from src.api.main import _generate_weekly_alerts  # noqa: PLC0415
            n_alerts = _generate_weekly_alerts(week)
            result["alerts_generated"] = n_alerts
        except Exception as exc:
            result["alert_error"] = str(exc)

        result["elapsed_seconds"] = round(time.time() - t0, 1)
        _mark_job(job_id, "completed", result=result)

    except Exception as exc:
        _mark_job(job_id, "failed", error=str(exc))


def _compute_health_scores_for_week(week: date) -> int:
    """
    Core ML scoring logic for one week.
    Maps behavioral features → HR-proxy features → burnout/attrition/conflict predictions.
    Upserts results into health_scores.
    """
    import pickle  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    models_dir = Path(os.getenv("MODELS_DIR", "models"))
    burnout_path   = models_dir / "burnout_model.pkl"
    attrition_path = models_dir / "attrition_model.pkl"
    conflict_path  = models_dir / "conflict_model.pkl"

    if not (burnout_path.exists() and attrition_path.exists() and conflict_path.exists()):
        raise FileNotFoundError("One or more ML model files missing. Run POST /admin/train-models first.")

    with open(burnout_path, "rb")   as f: burnout_art   = pickle.load(f)
    with open(attrition_path, "rb") as f: attrition_art = pickle.load(f)
    with open(conflict_path, "rb")  as f: conflict_art  = pickle.load(f)

    import pandas as pd  # noqa: PLC0415

    engine = create_engine(DATABASE_URL)
    written = 0

    with Session(engine) as session:
        bf_rows = session.execute(
            text("""
                SELECT
                    bf.employee_id,
                    bf.avg_response_hours,
                    bf.after_hours_count,
                    bf.message_count,
                    bf.avg_message_length,
                    bf.participation_rate,
                    bf.manager_comm_ratio,
                    bf.sentiment_velocity,
                    e.tenure_months
                FROM behavioral_features bf
                JOIN employees e ON e.id = bf.employee_id
                WHERE bf.week = :week
            """),
            {"week": week},
        ).fetchall()

        # Conflict proxy from comm_graph
        conflict_by_emp: dict[int, float] = {}
        cg_rows = session.execute(
            text("""
                SELECT employee_a, employee_b, relationship_health, avg_sentiment
                FROM comm_graph WHERE week = :week
            """),
            {"week": week},
        ).fetchall()
        for ea, eb, rh, av in cg_rows:
            rh = float(rh or 1.0)
            av = float(av or 0.0)
            c  = max(0.0, (1 - rh) * 0.6 + max(0, -av) * 0.4)
            for e in (ea, eb):
                conflict_by_emp[e] = max(conflict_by_emp.get(e, 0.0), c)

        for row in bf_rows:
            (emp_id, avg_resp, after_h, msg_cnt, avg_len,
             prate, mgr_ratio, svel, tenure) = row

            # Map communication features → HR Analytics feature space
            ah     = float(after_h  or 0)
            p      = float(prate    or 0.8)
            sv     = float(svel     or 0.0)
            msgs   = float(msg_cnt  or 20)
            resp_h = float(avg_resp or 2.0)

            satisfaction = max(0.05, min(0.95,
                p * 0.6 + max(0.0, sv) * 0.3 - min(0.3, ah / 30)
            ))
            last_eval = max(0.2, min(1.0,
                p * 0.7 + max(0.0, sv * 2) * 0.15 - min(0.2, max(0.0, resp_h - 4) / 48)
            ))
            monthly_h = int(min(310, 160 + ah * 3.5 + msgs * 0.4))
            n_proj = 2 if msgs <= 15 else 3 if msgs <= 30 else 4 if msgs <= 50 else 5 if msgs <= 80 else 6

            hr_feats = {
                "satisfaction_level":    round(satisfaction, 3),
                "last_evaluation":       round(last_eval, 3),
                "number_project":        n_proj,
                "average_montly_hours":  monthly_h,
                "time_spend_company":    max(1, min(10, int(tenure or 12) // 12)),
                "work_accident":         0,
                "promotion_last_5years": 0,
                "salary":                1,
            }

            def _predict(artifact: dict, feats: dict) -> tuple[float, dict]:
                feat_cols = artifact["feature_columns"]
                row_df = pd.DataFrame([{c: feats.get(c, 0.0) for c in feat_cols}])
                try:
                    prob = float(artifact["model"].predict_proba(row_df)[0][1])
                except Exception:
                    prob = 0.0
                try:
                    sv_arr = artifact["explainer"].shap_values(row_df)
                    if isinstance(sv_arr, list):
                        sv_arr = sv_arr[1][0]
                    else:
                        sv_arr = sv_arr[0]
                    shap_d = {c: round(float(v), 4) for c, v in zip(feat_cols, sv_arr)}
                except Exception:
                    shap_d = {}
                return prob, shap_d

            burnout_risk, shap_vals = _predict(burnout_art, hr_feats)

            # Attrition model outputs multiple horizons
            try:
                a_feat_cols = attrition_art["feature_columns"]
                a_row = pd.DataFrame([{c: hr_feats.get(c, 0.0) for c in a_feat_cols}])
                a_probs = attrition_art["model"].predict_proba(a_row)[0]
                # Encode 30/60/90-day as scaling of the base probability
                base_attr = float(a_probs[1])
                attrition_30d = round(min(1.0, base_attr * 0.6), 4)
                attrition_60d = round(min(1.0, base_attr * 0.8), 4)
                attrition_90d = round(min(1.0, base_attr), 4)
            except Exception:
                attrition_30d = attrition_60d = attrition_90d = 0.0

            conflict_risk   = round(min(1.0, conflict_by_emp.get(emp_id, 0.0)), 4)
            engagement      = round(max(0.0, min(1.0, p * 0.7 + max(0.0, sv) * 0.3)), 4)
            overall         = max(0, min(100, int(100 - (
                burnout_risk * 40 + attrition_60d * 30 + conflict_risk * 15 + (1 - engagement) * 15
            ))))

            flags: list[str] = []
            if burnout_risk >= 0.85:  flags.append("CRITICAL_BURNOUT")
            elif burnout_risk >= 0.70: flags.append("HIGH_BURNOUT")
            if attrition_60d >= 0.65: flags.append("FLIGHT_RISK")
            elif attrition_60d >= 0.45: flags.append("ATTRITION_WATCH")
            if conflict_risk >= 0.60:  flags.append("CONFLICT_RISK")
            if ah >= 15:               flags.append("OVERWORK")
            if p < 0.4:               flags.append("DISENGAGED")

            try:
                session.execute(
                    text("""
                        INSERT INTO health_scores
                            (employee_id, week, burnout_risk, attrition_risk_30d,
                             attrition_risk_60d, attrition_risk_90d, conflict_risk,
                             engagement_score, overall_score, shap_values, flags)
                        VALUES
                            (:eid, :week, :burnout, :a30, :a60, :a90, :conflict,
                             :engage, :overall, cast(:shap as jsonb), cast(:flags as jsonb))
                        ON CONFLICT (employee_id, week) DO UPDATE SET
                            burnout_risk       = EXCLUDED.burnout_risk,
                            attrition_risk_30d = EXCLUDED.attrition_risk_30d,
                            attrition_risk_60d = EXCLUDED.attrition_risk_60d,
                            attrition_risk_90d = EXCLUDED.attrition_risk_90d,
                            conflict_risk      = EXCLUDED.conflict_risk,
                            engagement_score   = EXCLUDED.engagement_score,
                            overall_score      = EXCLUDED.overall_score,
                            shap_values        = EXCLUDED.shap_values,
                            flags              = EXCLUDED.flags,
                            computed_at        = now()
                    """),
                    {
                        "eid":     emp_id,
                        "week":    week,
                        "burnout": round(burnout_risk, 4),
                        "a30":     attrition_30d,
                        "a60":     attrition_60d,
                        "a90":     attrition_90d,
                        "conflict":conflict_risk,
                        "engage":  engagement,
                        "overall": overall,
                        "shap":    json.dumps(shap_vals),
                        "flags":   json.dumps(flags),
                    },
                )
                written += 1
            except Exception:
                session.rollback()

        session.commit()

    return written


def _run_training_job(job_id: str, skip_bert: bool) -> None:
    """Background task that retrains all ML models."""
    from pathlib import Path  # noqa: PLC0415

    result: dict = {"skip_bert": skip_bert}
    t0 = time.time()

    # Burnout predictor
    try:
        from src.ml.burnout_predictor import train as train_burnout  # noqa: PLC0415
        train_burnout()
        result["burnout_predictor"] = "retrained"
    except Exception as exc:
        result["burnout_predictor"] = f"failed: {exc}"

    # Attrition model
    try:
        from src.ml.attrition_model import train as train_attrition  # noqa: PLC0415
        train_attrition()
        result["attrition_model"] = "retrained"
    except Exception as exc:
        result["attrition_model"] = f"failed: {exc}"

    # Conflict detector
    try:
        from src.ml.conflict_detector import train as train_conflict  # noqa: PLC0415
        train_conflict()
        result["conflict_detector"] = "retrained"
    except Exception as exc:
        result["conflict_detector"] = f"failed: {exc}"

    # Emotion classifier (optional — BERT training takes 30+ minutes)
    if not skip_bert:
        try:
            from src.nlp.emotion_classifier import fine_tune_emotion_classifier  # noqa: PLC0415
            metrics = fine_tune_emotion_classifier()
            result["emotion_classifier"] = {"status": "retrained", "metrics": metrics}
        except Exception as exc:
            result["emotion_classifier"] = f"failed: {exc}"
    else:
        result["emotion_classifier"] = "skipped (skip_bert=true)"

    # Reload the metrics.json file and append
    metrics_path = Path(os.getenv("MODELS_DIR", "models")) / "metrics.json"
    existing: dict = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text())
        except Exception:
            pass
    existing["_last_retrained"] = datetime.now(timezone.utc).isoformat()
    metrics_path.write_text(json.dumps(existing, indent=2))

    result["elapsed_seconds"] = round(time.time() - t0, 1)
    _mark_job(job_id, "completed", result=result)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2: POST /admin/run-pipeline
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    week: Optional[str] = None  # ISO date string "YYYY-MM-DD" or "current"


@router.post(
    "/admin/run-pipeline",
    summary="Manually trigger the analytics pipeline",
    description=(
        "Manually triggers the full weekly analytics pipeline for HR admins, "
        "without waiting for the Sunday 02:00 UTC Celery schedule.\n\n"
        "**Pipeline steps** (run in order):\n"
        "1. Behavioural feature engineering (`behavioral_features` table)\n"
        "2. Communication graph rebuild (`comm_graph` table)\n"
        "3. Network centrality analysis\n"
        "4. ML predictions → `health_scores` table (burnout, attrition, conflict)\n"
        "5. Alert generation for employees above risk thresholds\n\n"
        "The job runs **asynchronously**. Poll `GET /admin/jobs/{job_id}` for completion.\n\n"
        "`week` defaults to the current Monday if omitted. Pass `'current'` or an ISO date "
        "(`2024-03-18`) to target a specific week.\n\n"
        "**Requires: hr role**"
    ),
)
async def run_pipeline_manually(
    body: PipelineRequest = PipelineRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _user: TokenData = Depends(require_role("hr")),
) -> dict:
    job_id = uuid.uuid4().hex[:8]
    _JOBS[job_id] = {
        "job_id":     job_id,
        "type":       "pipeline",
        "status":     "running",
        "week":       body.week or "current",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_by": _user.email,
        "completed_at": None,
        "result":     None,
        "error":      None,
    }
    background_tasks.add_task(_run_pipeline_job, job_id, body.week)
    return {
        "job_id":  job_id,
        "status":  "running",
        "message": "Pipeline started. Poll GET /admin/jobs/{job_id} for results.",
        "week":    body.week or "current",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 3: POST /admin/train-models
# ─────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    skip_bert: bool = True  # BERT fine-tuning takes 30+ minutes


@router.post(
    "/admin/train-models",
    summary="Retrain all ML models",
    description=(
        "Triggers retraining of all four ML models from the latest data on disk. "
        "Requires `data/raw/hr_analytics.csv` to be present.\n\n"
        "**Models retrained:**\n"
        "- Burnout Predictor (XGBoost + SMOTE, ~30 seconds)\n"
        "- Attrition Model (XGBoost calibrated, ~30 seconds)\n"
        "- Conflict Detector (XGBoost proxy labels, ~20 seconds)\n"
        "- Emotion Classifier (BERT fine-tune, **~30 minutes** — set `skip_bert=true` to skip)\n\n"
        "Models are saved to the `models/` directory and picked up immediately by the prediction pipeline.\n\n"
        "The job runs **asynchronously**. Poll `GET /admin/jobs/{job_id}` for completion and metrics.\n\n"
        "**Requires: hr role**"
    ),
)
async def trigger_training(
    body: TrainRequest = TrainRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _user: TokenData = Depends(require_role("hr")),
) -> dict:
    job_id = uuid.uuid4().hex[:8]
    _JOBS[job_id] = {
        "job_id":       job_id,
        "type":         "training",
        "status":       "running",
        "skip_bert":    body.skip_bert,
        "started_at":   datetime.now(timezone.utc).isoformat(),
        "started_by":   _user.email,
        "completed_at": None,
        "result":       None,
        "error":        None,
    }
    background_tasks.add_task(_run_training_job, job_id, body.skip_bert)
    return {
        "job_id":    job_id,
        "status":    "running",
        "skip_bert": body.skip_bert,
        "message":   (
            "Training started. This takes 1–2 minutes (or 30+ min if skip_bert=false). "
            "Poll GET /admin/jobs/{job_id} for results."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 4: GET /admin/jobs/{job_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/admin/jobs/{job_id}",
    summary="Poll async job status",
    description=(
        "Returns the current status of an asynchronous pipeline or training job.\n\n"
        "**Status values:** `running` → `completed` → `failed`\n\n"
        "When `status == 'completed'`, the `result` field contains step-by-step output "
        "including rows written, graph edge count, and model metrics.\n\n"
        "**Requires: hr role**"
    ),
)
async def get_job_status(
    job_id: str,
    _user: TokenData = Depends(require_role("hr")),
) -> dict:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. Jobs are stored in memory and reset on server restart.",
        )
    return job
