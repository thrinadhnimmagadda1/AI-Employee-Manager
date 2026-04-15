"""
CogniTeam FastAPI Application
Main entry point: registers routers, middleware, Celery, and health check.
"""
from __future__ import annotations

# Load .env before any os.getenv() calls.
# override=True so the .env file is the single source of truth in development.
# In production, inject real secrets via the OS environment — they take precedence
# when override=False, but explicit .env settings are correct for local dev.
try:
    from pathlib import Path as _PPath                   # noqa: PLC0415
    from dotenv import load_dotenv as _load_dotenv       # noqa: PLC0415
    _load_dotenv(
        dotenv_path=_PPath(__file__).resolve().parent.parent.parent / ".env",
        override=True,
    )
except ImportError:
    pass

import json as _json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path as _Path
from typing import Optional

from celery import Celery
from celery.schedules import crontab
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from src.api.middleware.privacy import AuditLoggingMiddleware
from src.api.routes import admin, alerts, chat, dashboard, employees, graph

# httpOnly cookie is Secure only in production (HTTPS)
_COOKIE_SECURE   = os.getenv("COOKIE_SECURE",  "false").lower() == "true"
_COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# ─────────────────────────────────────────────────────────────
# Celery setup
# ─────────────────────────────────────────────────────────────

celery_app = Celery(
    "cogniteam",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "weekly-scoring-job": {
            "task": "src.api.main.run_weekly_scoring",
            "schedule": crontab(day_of_week="sunday", hour=2, minute=0),
        },
    },
)


# ─────────────────────────────────────────────────────────────
# Celery Tasks
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="src.api.main.run_weekly_scoring")
def run_weekly_scoring() -> dict:
    """
    Sunday 02:00 UTC weekly job:
    1. Score all NLP sentiment for the past week
    2. Compute behavioral features
    3. Rebuild graph and network metrics
    4. Run all ML models and update health_scores
    5. Generate new alerts for anyone above risk threshold
    6. Send email digests (mocked)
    """
    from datetime import datetime, timezone

    from src.ml.feature_engineering import compute_features_for_week
    from src.nlp.sentiment_model import score_all_employees_for_week
    from src.graph.graph_builder import build_and_persist_weekly_graph
    from src.graph.network_analyzer import analyze_network_for_week

    # Target last completed week (Monday)
    today = date.today()
    week = today - timedelta(days=today.weekday() + 7)  # previous Monday

    logger.info(f"=== Weekly scoring job started for week {week} ===")
    results: dict = {"week": str(week), "started_at": datetime.now(timezone.utc).isoformat()}

    try:
        nlp_count = score_all_employees_for_week(week)
        results["nlp_scored"] = nlp_count
        logger.info(f"NLP scoring complete: {nlp_count} employees")
    except Exception as exc:
        logger.error(f"NLP scoring failed: {exc}")
        results["nlp_error"] = str(exc)

    try:
        feat_count = compute_features_for_week(week)
        results["features_computed"] = feat_count
        logger.info(f"Feature engineering complete: {feat_count} employees")
    except Exception as exc:
        logger.error(f"Feature engineering failed: {exc}")
        results["feature_error"] = str(exc)

    try:
        G = build_and_persist_weekly_graph(week)
        results["graph_nodes"] = G.number_of_nodes()
        results["graph_edges"] = G.number_of_edges()
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as exc:
        logger.error(f"Graph build failed: {exc}")
        results["graph_error"] = str(exc)

    try:
        network_metrics = analyze_network_for_week(week)
        results["network_analyzed"] = len(network_metrics)
        logger.info(f"Network analysis complete: {len(network_metrics)} employees")
    except Exception as exc:
        logger.error(f"Network analysis failed: {exc}")
        results["network_error"] = str(exc)

    try:
        alert_count = _generate_weekly_alerts(week)
        results["alerts_generated"] = alert_count
        logger.info(f"Alerts generated: {alert_count}")
    except Exception as exc:
        logger.error(f"Alert generation failed: {exc}")
        results["alert_error"] = str(exc)

    logger.info(f"=== Weekly scoring job complete: {results} ===")
    return results


def _generate_weekly_alerts(week: date) -> int:
    """Scan health_scores and generate alerts for employees above risk thresholds."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    import json

    BURNOUT_HIGH = 0.70
    ATTRITION_HIGH = 0.65
    CONFLICT_HIGH = 0.60

    engine = create_engine(DATABASE_URL)
    alert_count = 0

    with Session(engine) as session:
        rows = session.execute(
            text("""
                SELECT employee_id, burnout_risk, attrition_risk_90d,
                       conflict_risk, flags
                FROM health_scores
                WHERE week = :week
            """),
            {"week": week},
        ).fetchall()

        for row in rows:
            emp_id, burnout, attrition, conflict, flags = row

            if burnout and burnout >= BURNOUT_HIGH:
                session.execute(
                    text("""
                        INSERT INTO alerts (employee_id, alert_type, severity, description, recommendations)
                        VALUES (:eid, 'burnout', :sev, :desc, :recs)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "eid": emp_id,
                        "sev": "critical" if burnout >= 0.85 else "high",
                        "desc": f"Burnout risk score {burnout:.1%} exceeds threshold.",
                        "recs": json.dumps(["Schedule 1-on-1", "Review workload", "Consider EAP referral"]),
                    },
                )
                alert_count += 1

            if attrition and attrition >= ATTRITION_HIGH:
                session.execute(
                    text("""
                        INSERT INTO alerts (employee_id, alert_type, severity, description, recommendations)
                        VALUES (:eid, 'attrition', :sev, :desc, :recs)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "eid": emp_id,
                        "sev": "high",
                        "desc": f"90-day attrition risk {attrition:.1%} is elevated.",
                        "recs": json.dumps(["Career conversation", "Compensation review", "Recognition"]),
                    },
                )
                alert_count += 1

            if conflict and conflict >= CONFLICT_HIGH:
                session.execute(
                    text("""
                        INSERT INTO alerts (employee_id, alert_type, severity, description, recommendations)
                        VALUES (:eid, 'conflict', :sev, :desc, :recs)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "eid": emp_id,
                        "sev": "medium",
                        "desc": f"Conflict risk score {conflict:.1%} detected in communication patterns.",
                        "recs": json.dumps(["Mediation session", "Team alignment meeting"]),
                    },
                )
                alert_count += 1

        session.commit()

    return alert_count


# ─────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure refresh_tokens table exists (idempotent migration)
    try:
        from src.api.middleware.auth import _ensure_db_schema  # noqa: PLC0415
        _ensure_db_schema()
        logger.info("DB schema verified (refresh_tokens ready).")
    except Exception as exc:
        logger.warning(f"Schema bootstrap skipped — DB may not be ready yet: {exc}")
    logger.info("CogniTeam API starting up …")
    yield
    logger.info("CogniTeam API shutting down.")


app = FastAPI(
    title="CogniTeam API",
    description="AI-powered organizational intelligence platform",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuditLoggingMiddleware)

# ── Routers ───────────────────────────────────────────────────
app.include_router(dashboard.router)
app.include_router(employees.router)
app.include_router(alerts.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(graph.router)


# ─────────────────────────────────────────────────────────────
# Cookie helper
# ─────────────────────────────────────────────────────────────

def _set_refresh_cookie(response: Response, raw_token: str, expires_at: datetime) -> None:
    """
    Write the refresh token as a Secure, httpOnly cookie scoped to /auth/*.

    The /auth path scope means the browser will ONLY include this cookie on
    requests to /auth/refresh and /auth/logout — never on data endpoints.
    This limits the cookie's exposure window compared to path="/".
    """
    max_age = max(1, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        key="refresh_token",
        value=raw_token,
        max_age=max_age,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE,
        path="/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key="refresh_token",
        path="/auth",
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE,
    )


# ── Auth routes ───────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post(
    "/auth/login",
    tags=["auth"],
    summary="Obtain access + refresh token pair",
    description=(
        "Authenticates with email/password and returns:\n\n"
        "- **`access_token`** (JWT, 30 min) — include as `Authorization: Bearer <token>` on every request\n"
        "- **`refresh_token`** — set as an httpOnly cookie scoped to `/auth/*`; "
        "never accessible from JavaScript\n\n"
        "The access token expires in 30 minutes. Use `POST /auth/refresh` to silently "
        "obtain a new one without re-entering credentials."
    ),
)
async def login(body: LoginRequest, response: Response) -> dict:
    from src.api.middleware.auth import (  # noqa: PLC0415
        authenticate_user,
        create_access_token,
        create_refresh_token,
        store_refresh_token,
        ACCESS_TOKEN_EXPIRE_MINUTES,
    )

    user = authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive")

    access_token = create_access_token({
        "sub":         str(user.id),
        "email":       user.email,
        "role":        user.role,
        "employee_id": user.employee_id,
    })

    raw_refresh = create_refresh_token()
    expires_at  = store_refresh_token(user.id, raw_refresh)
    _set_refresh_cookie(response, raw_refresh, expires_at)

    return {
        "access_token": access_token,
        "token_type":   "bearer",
        "role":         user.role,
        "expires_in":   ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@app.post(
    "/auth/refresh",
    tags=["auth"],
    summary="Silently refresh the access token",
    description=(
        "Issues a new access token using the httpOnly refresh-token cookie "
        "set during login. Also **rotates** the refresh token — the old cookie "
        "is invalidated and a fresh one is set.\n\n"
        "**Token rotation** guarantees that a stolen refresh token can only be used "
        "once: when the legitimate client next refreshes, the system detects the "
        "revoked-token reuse and terminates all sessions for that user.\n\n"
        "The frontend calls this automatically via an axios interceptor when any "
        "request returns HTTP 401."
    ),
)
async def refresh_token_endpoint(request: Request, response: Response) -> dict:
    from src.api.middleware.auth import (  # noqa: PLC0415
        verify_and_rotate_refresh_token,
        create_access_token,
        create_refresh_token,
        store_refresh_token,
        ACCESS_TOKEN_EXPIRE_MINUTES,
    )

    raw_token = request.cookies.get("refresh_token")
    if not raw_token:
        raise HTTPException(
            status_code=401,
            detail="No refresh token cookie. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = verify_and_rotate_refresh_token(raw_token)
    if not user:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=401,
            detail="Refresh token expired or revoked. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=403, detail="Account is inactive.")

    access_token = create_access_token({
        "sub":         str(user.id),
        "email":       user.email,
        "role":        user.role,
        "employee_id": user.employee_id,
    })

    # Issue a brand-new refresh token (rotation)
    new_raw_refresh = create_refresh_token()
    new_expires_at  = store_refresh_token(user.id, new_raw_refresh)
    _set_refresh_cookie(response, new_raw_refresh, new_expires_at)

    return {
        "access_token": access_token,
        "token_type":   "bearer",
        "role":         user.role,
        "expires_in":   ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@app.post(
    "/auth/logout",
    tags=["auth"],
    summary="Revoke the refresh token and end the session",
    description=(
        "Invalidates the current refresh token in the database and clears the "
        "httpOnly cookie. The access token expires naturally (within 30 min).\n\n"
        "After logout the client must call `POST /auth/login` to start a new session. "
        "The refresh cookie cannot be used again even if the client still holds a copy."
    ),
)
async def logout_endpoint(request: Request, response: Response) -> dict:
    from src.api.middleware.auth import revoke_refresh_token  # noqa: PLC0415

    raw_token = request.cookies.get("refresh_token")
    if raw_token:
        try:
            revoke_refresh_token(raw_token)
        except Exception:
            pass  # DB errors must not prevent the client from receiving the cookie-clear

    _clear_refresh_cookie(response)
    return {"message": "Logged out successfully."}


# ─────────────────────────────────────────────────────────────
# Shared sub-system probes  (reused by /health and /api/info)
# ─────────────────────────────────────────────────────────────

def _probe_database() -> bool:
    from sqlalchemy import create_engine, text  # noqa: PLC0415
    try:
        eng = create_engine(DATABASE_URL)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _probe_redis() -> str:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis as _redis  # noqa: PLC0415
        r = _redis.from_url(redis_url, socket_connect_timeout=1)
        r.ping()
        return "connected"
    except Exception:
        return "disconnected"


def _model_file_status() -> dict:
    models_dir = _Path(os.getenv("MODELS_DIR", "models"))
    return {
        "burnout_predictor":  (models_dir / "burnout_model.pkl").exists(),
        "attrition_model":    (models_dir / "attrition_model.pkl").exists(),
        "conflict_detector":  (models_dir / "conflict_model.pkl").exists(),
        "emotion_classifier": (models_dir / "emotion_classifier").is_dir(),
    }


def _load_metrics_json() -> dict:
    metrics_path = _Path(os.getenv("MODELS_DIR", "models")) / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return _json.loads(metrics_path.read_text())
    except Exception:
        return {}


# ── Health check (lightweight liveness probe) ─────────────────
@app.get(
    "/health",
    tags=["system"],
    summary="Liveness probe",
    description=(
        "Lightweight health check for load-balancers and uptime monitors. "
        "Returns overall status and sub-system connectivity."
    ),
)
async def health_check() -> dict:
    db_ok       = _probe_database()
    model_files = _model_file_status()

    return {
        "status":         "healthy" if db_ok else "degraded",
        "version":        "1.0.0",
        "database":       "connected" if db_ok else "disconnected",
        "models_trained": f"{sum(model_files.values())}/4",
        "model_files":    model_files,
    }


# ── /api/info — rich system status for demos and documentation ─
@app.get(
    "/api/info",
    tags=["system"],
    summary="Full system information",
    description=(
        "Returns comprehensive project metadata for demo and documentation purposes.\n\n"
        "Includes:\n"
        "- All 4 ML model names, training status, algorithm, and live metrics from `models/metrics.json`\n"
        "- Live dataset row counts (employees, emails, health scores, alerts)\n"
        "- Sub-system health: PostgreSQL, Redis, and Ollama\n\n"
        "**No authentication required** — safe to embed in demo landing pages."
    ),
)
async def system_info() -> dict:
    from sqlalchemy import create_engine, text  # noqa: PLC0415
    from sqlalchemy.orm import Session          # noqa: PLC0415

    db_ok          = _probe_database()
    redis_status   = _probe_redis()
    model_files    = _model_file_status()
    raw_metrics    = _load_metrics_json()

    # ── Live dataset stats ────────────────────────────────────
    dataset_stats: dict = {
        "employees": 0,
        "message_metadata_rows": 0,
        "weeks_of_data": 0,
        "health_scores_computed": 0,
        "behavioral_feature_rows": 0,
        "active_alerts": 0,
        "unresolved_critical_alerts": 0,
    }
    if db_ok:
        try:
            eng = create_engine(DATABASE_URL)
            with Session(eng) as s:
                dataset_stats["employees"] = (
                    s.execute(text("SELECT COUNT(*) FROM employees")).scalar() or 0
                )
                dataset_stats["message_metadata_rows"] = (
                    s.execute(text("SELECT COUNT(*) FROM message_metadata")).scalar() or 0
                )
                dataset_stats["health_scores_computed"] = (
                    s.execute(text("SELECT COUNT(*) FROM health_scores")).scalar() or 0
                )
                dataset_stats["behavioral_feature_rows"] = (
                    s.execute(text("SELECT COUNT(*) FROM behavioral_features")).scalar() or 0
                )
                dataset_stats["active_alerts"] = (
                    s.execute(text("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE")).scalar() or 0
                )
                dataset_stats["unresolved_critical_alerts"] = (
                    s.execute(
                        text("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE AND severity = 'critical'")
                    ).scalar() or 0
                )
                dataset_stats["weeks_of_data"] = (
                    s.execute(text("SELECT COUNT(DISTINCT week) FROM health_scores")).scalar() or 0
                )
        except Exception:
            pass

    # ── Per-model metadata + metrics ──────────────────────────
    _MODEL_META = {
        "burnout_predictor": {
            "name":        "Burnout Predictor",
            "algorithm":   "XGBoost + SMOTE + SHAP",
            "trained_on":  "HR Analytics — Kaggle (14,999 employees)",
            "description": "Predicts burnout probability from behavioural signals.",
            "output":      "burnout_risk [0–1]",
        },
        "attrition_model": {
            "name":        "Attrition Model",
            "algorithm":   "XGBoost (calibrated) + SHAP",
            "trained_on":  "HR Analytics — Kaggle (14,999 employees)",
            "description": "Estimates voluntary-leave probability at 30 / 60 / 90 days.",
            "output":      "attrition_risk_30d / 60d / 90d [0–1]",
        },
        "conflict_detector": {
            "name":        "Conflict Detector",
            "algorithm":   "XGBoost (pair-level graph features)",
            "trained_on":  "HR Analytics proxy labels",
            "description": "Identifies dysfunctional communication pairs.",
            "output":      "conflict_risk [0–1]",
        },
        "emotion_classifier": {
            "name":        "Emotion Classifier",
            "algorithm":   "BERT (bert-base-uncased) fine-tuned",
            "trained_on":  "GoEmotions — 58,000 Reddit comments, 7 classes",
            "description": "Classifies dominant emotion in communication text.",
            "output":      "emotion label + confidence",
        },
    }

    models = [
        {
            **meta,
            "trained": model_files.get(key, False),
            "metrics": raw_metrics.get(key, {}),
        }
        for key, meta in _MODEL_META.items()
    ]

    return {
        "project":        "CogniTeam",
        "version":        "1.0.0",
        "description": (
            "AI-powered organisational intelligence platform. "
            "Detects burnout, attrition risk, and workplace conflict "
            "from anonymised communication metadata using ML + LLM."
        ),
        "models":          models,
        "models_trained":  f"{sum(model_files.values())}/4",
        "dataset_stats":   dataset_stats,
        "system": {
            "database": "connected" if db_ok else "disconnected",
            "redis":    redis_status,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
