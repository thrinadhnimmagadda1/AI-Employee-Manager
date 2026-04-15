"""Employee API routes."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.api.middleware.auth import TokenData, get_current_user
from src.api.middleware.privacy import apply_dp_noise_to_response

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")
router = APIRouter(prefix="/employees", tags=["employees"])


def _get_session() -> Session:
    engine = create_engine(DATABASE_URL)
    return Session(engine)


@router.get("")
async def list_employees(user: TokenData = Depends(get_current_user)) -> dict:
    """
    Returns all employees with their current health scores.
    Role: manager or hr only.
    """
    if user.role not in ("manager", "hr"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    with _get_session() as session:
        rows = session.execute(
            text("""
                SELECT
                    e.id, e.name, e.department, e.role, e.tenure_months,
                    hs.overall_score, hs.burnout_risk, hs.conflict_risk,
                    hs.flags, hs.week
                FROM employees e
                LEFT JOIN health_scores hs
                    ON hs.employee_id = e.id
                   AND hs.week = (SELECT MAX(week) FROM health_scores WHERE employee_id = e.id)
                ORDER BY hs.overall_score ASC NULLS LAST
            """),
        ).fetchall()

    employees = [
        {
            "id": r[0],
            "name": r[1],
            "department": r[2],
            "role": r[3],
            "tenure_months": r[4],
            "overall_score": r[5],
            "burnout_risk": r[6],
            "conflict_risk": r[7],
            "flags": r[8] or [],
            "score_week": str(r[9]) if r[9] else None,
        }
        for r in rows
    ]

    return apply_dp_noise_to_response({"employees": employees, "total": len(employees)})


@router.get("/{employee_id}")
async def get_employee(
    employee_id: int,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """
    Returns full profile: all scores, trends, and alerts.
    Employees can only view their own profile.
    Managers see aggregated. HR sees full detail including SHAP.
    """
    if user.role == "employee" and user.employee_id != employee_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    with _get_session() as session:
        emp_row = session.execute(
            text("""
                SELECT id, name, department, role, manager_id, tenure_months
                FROM employees WHERE id = :eid
            """),
            {"eid": employee_id},
        ).fetchone()

        if not emp_row:
            raise HTTPException(status_code=404, detail="Employee not found")

        hs_row = session.execute(
            text("""
                SELECT burnout_risk, attrition_risk_30d, attrition_risk_60d,
                       attrition_risk_90d, conflict_risk, engagement_score,
                       overall_score, shap_values, flags, week
                FROM health_scores
                WHERE employee_id = :eid
                ORDER BY week DESC LIMIT 1
            """),
            {"eid": employee_id},
        ).fetchone()

        bf_row = session.execute(
            text("""
                SELECT avg_response_hours, after_hours_count, message_count,
                       avg_message_length, participation_rate, manager_comm_ratio,
                       sentiment_velocity
                FROM behavioral_features
                WHERE employee_id = :eid
                ORDER BY week DESC LIMIT 1
            """),
            {"eid": employee_id},
        ).fetchone()

        alert_rows = session.execute(
            text("""
                SELECT id, alert_type, severity, description, recommendations, created_at
                FROM alerts
                WHERE employee_id = :eid AND resolved = FALSE
                ORDER BY created_at DESC
            """),
            {"eid": employee_id},
        ).fetchall()

        trend_rows = session.execute(
            text("""
                SELECT hs.week, hs.overall_score, hs.burnout_risk, ss.sentiment, ss.emotion
                FROM health_scores hs
                LEFT JOIN sentiment_scores ss ON ss.employee_id = hs.employee_id AND ss.week = hs.week
                WHERE hs.employee_id = :eid
                ORDER BY hs.week DESC LIMIT 12
            """),
            {"eid": employee_id},
        ).fetchall()

    response = {
        "employee": {
            "id": emp_row[0],
            "name": emp_row[1],
            "department": emp_row[2],
            "role": emp_row[3],
            "manager_id": emp_row[4],
            "tenure_months": emp_row[5],
        },
        "health_scores": {
            "burnout_risk": hs_row[0] if hs_row else None,
            "attrition_risk_30d": hs_row[1] if hs_row else None,
            "attrition_risk_60d": hs_row[2] if hs_row else None,
            "attrition_risk_90d": hs_row[3] if hs_row else None,
            "conflict_risk": hs_row[4] if hs_row else None,
            "engagement_score": hs_row[5] if hs_row else None,
            "overall_score": hs_row[6] if hs_row else None,
            # SHAP only for HR
            "shap_values": (hs_row[7] if hs_row else None) if user.role == "hr" else None,
            "flags": (hs_row[8] if hs_row else None) or [],
        },
        "behavioral_features": {
            "avg_response_hours": bf_row[0] if bf_row else None,
            "after_hours_count": bf_row[1] if bf_row else None,
            "message_count": bf_row[2] if bf_row else None,
            "avg_message_length": bf_row[3] if bf_row else None,
            "participation_rate": bf_row[4] if bf_row else None,
            "manager_comm_ratio": bf_row[5] if bf_row else None,
            "sentiment_velocity": bf_row[6] if bf_row else None,
        } if bf_row else None,
        "active_alerts": [
            {
                "id": r[0],
                "type": r[1],
                "severity": r[2],
                "description": r[3],
                "recommendations": r[4] or [],
                "created_at": str(r[5]),
            }
            for r in alert_rows
        ],
        "sentiment_trend": [
            {"week": str(r[0]), "overall_score": r[1], "burnout_risk": r[2], "sentiment": r[3], "emotion": r[4]}
            for r in trend_rows
        ],
    }

    return apply_dp_noise_to_response(response)


@router.get(
    "/{employee_id}/history",
    summary="Cross-week health history",
    description=(
        "Returns all available weeks of health scores for one employee, "
        "side-by-side — enabling HR and managers to see the full trajectory.\n\n"
        "**Response includes:**\n"
        "- `weeks` — array of weekly snapshots with burnout risk, attrition, conflict, "
        "engagement, overall score, active flags, and key behavioural metrics\n"
        "- `trajectory` — `improving` / `stable` / `declining` based on "
        "4-week rolling average comparison\n"
        "- `best_week` — highest overall score week\n"
        "- `worst_week` — lowest overall score week\n"
        "- `score_range` — min / max / avg across all weeks\n\n"
        "Employees can view only their own history. "
        "Managers and HR can view any employee."
    ),
)
async def get_employee_history(
    employee_id: int,
    weeks: int = 12,
    user: TokenData = Depends(get_current_user),
) -> dict:
    if user.role == "employee" and user.employee_id != employee_id:
        raise HTTPException(status_code=403, detail="Employees can only view their own data.")

    with _get_session() as session:
        # Verify employee exists
        emp_row = session.execute(
            text("SELECT id, name, department FROM employees WHERE id = :eid"),
            {"eid": employee_id},
        ).fetchone()
        if not emp_row:
            raise HTTPException(status_code=404, detail="Employee not found")

        rows = session.execute(
            text("""
                SELECT
                    hs.week,
                    hs.overall_score,
                    hs.burnout_risk,
                    hs.attrition_risk_30d,
                    hs.attrition_risk_60d,
                    hs.attrition_risk_90d,
                    hs.conflict_risk,
                    hs.engagement_score,
                    hs.flags,
                    bf.message_count,
                    bf.after_hours_count,
                    bf.participation_rate,
                    bf.sentiment_velocity,
                    bf.avg_response_hours
                FROM health_scores hs
                LEFT JOIN behavioral_features bf
                    ON bf.employee_id = hs.employee_id
                   AND bf.week = hs.week
                WHERE hs.employee_id = :eid
                ORDER BY hs.week DESC
                LIMIT :weeks
            """),
            {"eid": employee_id, "weeks": max(1, min(52, weeks))},
        ).fetchall()

    if not rows:
        return {
            "employee_id":  employee_id,
            "name":         emp_row[1],
            "department":   emp_row[2],
            "weeks_of_data": 0,
            "trajectory":   "no_data",
            "best_week":    None,
            "worst_week":   None,
            "score_range":  None,
            "weeks":        [],
        }

    week_data = [
        {
            "week":               str(r[0]),
            "overall_score":      r[1],
            "burnout_risk":       r[2],
            "attrition_risk_30d": r[3],
            "attrition_risk_60d": r[4],
            "attrition_risk_90d": r[5],
            "conflict_risk":      r[6],
            "engagement_score":   r[7],
            "flags":              r[8] or [],
            "message_count":      r[9],
            "after_hours_count":  r[10],
            "participation_rate": r[11],
            "sentiment_velocity": r[12],
            "avg_response_hours": r[13],
        }
        for r in rows
    ]

    # ── Trajectory: compare most-recent 4 weeks vs prior 4 weeks ──
    scores = [w["overall_score"] for w in week_data if w["overall_score"] is not None]

    def _avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    recent_scores = scores[:4]
    prior_scores  = scores[4:8]
    if len(scores) < 2:
        trajectory = "insufficient_data"
    elif not prior_scores:
        trajectory = "stable"
    else:
        delta = _avg(recent_scores) - _avg(prior_scores)
        if delta > 5:
            trajectory = "improving"
        elif delta < -5:
            trajectory = "declining"
        else:
            trajectory = "stable"

    # ── Best / worst weeks ──────────────────────────────────────
    scored_weeks = [w for w in week_data if w["overall_score"] is not None]
    best_week  = max(scored_weeks, key=lambda w: w["overall_score"]) if scored_weeks else None
    worst_week = min(scored_weeks, key=lambda w: w["overall_score"]) if scored_weeks else None

    score_range: dict | None = None
    if scores:
        score_range = {
            "min":   min(scores),
            "max":   max(scores),
            "avg":   round(_avg(scores), 1),
            "trend": round(scores[0] - scores[-1], 1) if len(scores) > 1 else 0,
        }

    # ── Burnout / attrition trend summaries ────────────────────
    burnout_values  = [w["burnout_risk"]       for w in week_data if w["burnout_risk"]       is not None]
    attrition_vals  = [w["attrition_risk_60d"] for w in week_data if w["attrition_risk_60d"] is not None]
    risk_summary: dict = {}
    if burnout_values:
        risk_summary["burnout_peak"]     = round(max(burnout_values), 3)
        risk_summary["burnout_current"]  = round(burnout_values[0], 3)
        risk_summary["burnout_trend"]    = round(burnout_values[0] - burnout_values[-1], 3)
    if attrition_vals:
        risk_summary["attrition_peak"]   = round(max(attrition_vals), 3)
        risk_summary["attrition_current"]= round(attrition_vals[0], 3)

    result = {
        "employee_id":  employee_id,
        "name":         emp_row[1],
        "department":   emp_row[2],
        "weeks_of_data": len(week_data),
        "trajectory":   trajectory,
        "best_week":    {"week": best_week["week"],  "overall_score": best_week["overall_score"]}  if best_week  else None,
        "worst_week":   {"week": worst_week["week"], "overall_score": worst_week["overall_score"]} if worst_week else None,
        "score_range":  score_range,
        "risk_summary": risk_summary,
        "weeks":        week_data,
    }
    return apply_dp_noise_to_response(result)


@router.get("/{employee_id}/risk-factors")
async def get_risk_factors(
    employee_id: int,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """
    Returns SHAP-explained risk factors for this employee.
    Accessible by: manager (aggregated), hr (full SHAP detail).
    """
    if user.role == "employee" and user.employee_id != employee_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    with _get_session() as session:
        row = session.execute(
            text("""
                SELECT shap_values, flags, burnout_risk, overall_score
                FROM health_scores
                WHERE employee_id = :eid
                ORDER BY week DESC LIMIT 1
            """),
            {"eid": employee_id},
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No health scores found for this employee.")

    shap_values = row[0] or {}
    flags = row[1] or []

    risk_factors = []
    if user.role == "hr" and shap_values:
        sorted_factors = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        risk_factors = [
            {
                "feature": feat,
                "impact": val,
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            }
            for feat, val in sorted_factors[:10]
        ]
    elif user.role == "manager":
        risk_factors = [{"signal": f} for f in flags]

    return apply_dp_noise_to_response({
        "employee_id": employee_id,
        "burnout_risk": row[2],
        "overall_score": row[3],
        "flags": flags,
        "risk_factors": risk_factors,
    })
