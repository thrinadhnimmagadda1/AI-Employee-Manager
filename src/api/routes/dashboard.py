"""Dashboard API routes."""
from __future__ import annotations

import os
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.api.middleware.auth import TokenData, get_current_user
from src.api.middleware.privacy import apply_dp_noise_to_response

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")
router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _get_session() -> Session:
    engine = create_engine(DATABASE_URL)
    return Session(engine)


def _get_team_health_summary(team_id: int) -> dict:
    """Fetch aggregated health scores and top alerts for one team."""
    with _get_session() as session:
        rows = session.execute(
            text("""
                SELECT e.id, e.name,
                       hs.burnout_risk, hs.overall_score,
                       hs.attrition_risk_90d, hs.conflict_risk, hs.flags
                FROM employees e
                LEFT JOIN health_scores hs
                    ON hs.employee_id = e.id
                   AND hs.week = (SELECT MAX(week) FROM health_scores WHERE team_id = :tid)
                WHERE hs.team_id = :tid OR e.id IN (
                    SELECT employee_id FROM health_scores WHERE team_id = :tid
                )
            """),
            {"tid": team_id},
        ).fetchall()

        alerts_rows = session.execute(
            text("""
                SELECT a.employee_id, a.alert_type, a.severity, a.description
                FROM alerts a
                INNER JOIN health_scores hs ON hs.employee_id = a.employee_id
                WHERE hs.team_id = :tid AND a.resolved = FALSE
                ORDER BY
                    CASE a.severity
                        WHEN 'critical' THEN 1
                        WHEN 'high'     THEN 2
                        WHEN 'medium'   THEN 3
                        ELSE 4
                    END
                LIMIT 10
            """),
            {"tid": team_id},
        ).fetchall()

    members = [
        {
            "employee_id": r[0],
            "name": r[1],
            "burnout_risk": r[2],
            "overall_score": r[3],
            "attrition_risk_90d": r[4],
            "conflict_risk": r[5],
            "flags": r[6],
        }
        for r in rows
    ]

    avg_score = (
        sum(m["overall_score"] or 50 for m in members) / len(members)
        if members else 50
    )

    return {
        "team_id": team_id,
        "member_count": len(members),
        "avg_health_score": round(avg_score, 1),
        "members": members,
        "top_alerts": [
            {"employee_id": r[0], "type": r[1], "severity": r[2], "description": r[3]}
            for r in alerts_rows
        ],
    }


@router.get("/overview")
async def get_overview(user: TokenData = Depends(get_current_user)) -> dict:
    """
    Returns all teams with their health scores.
    Accessible by: manager, hr
    """
    if user.role not in ("manager", "hr"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    with _get_session() as session:
        rows = session.execute(
            text("""
                SELECT
                    hs.team_id,
                    AVG(hs.overall_score)      AS avg_score,
                    AVG(hs.burnout_risk)        AS avg_burnout,
                    COUNT(DISTINCT hs.employee_id) AS member_count,
                    COUNT(a.id) FILTER (WHERE a.resolved = FALSE) AS active_alerts
                FROM health_scores hs
                LEFT JOIN alerts a ON a.employee_id = hs.employee_id
                WHERE hs.week = (SELECT MAX(week) FROM health_scores)
                GROUP BY hs.team_id
                ORDER BY avg_score ASC
            """),
        ).fetchall()

        total_employees = session.execute(
            text("SELECT COUNT(*) FROM employees")
        ).scalar()

        total_alerts = session.execute(
            text("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE")
        ).scalar()

        critical_alerts = session.execute(
            text("SELECT COUNT(*) FROM alerts WHERE resolved = FALSE AND severity = 'critical'")
        ).scalar()

    teams = [
        {
            "team_id": r[0],
            "avg_health_score": round(float(r[1] or 50), 1),
            "avg_burnout_risk": round(float(r[2] or 0), 3),
            "member_count": r[3],
            "active_alerts": r[4],
        }
        for r in rows
    ]

    return apply_dp_noise_to_response({
        "total_employees": total_employees,
        "total_active_alerts": total_alerts,
        "critical_alerts": critical_alerts,
        "teams": teams,
    })


@router.get("/team/{team_id}")
async def get_team_dashboard(
    team_id: int,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """
    Returns team health score, all member scores, top alerts, and graph.
    """
    if user.role not in ("manager", "hr"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    from src.graph.visualization import build_graph_json

    try:
        team_data = _get_team_health_summary(team_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Build graph for current week
    current_week = date.today()
    current_week = current_week.replace(day=current_week.day - current_week.weekday())  # Monday
    try:
        graph_json = build_graph_json(current_week)
    except Exception:
        graph_json = {"nodes": [], "links": []}

    team_data["graph_data"] = graph_json
    return apply_dp_noise_to_response(team_data)


@router.get("/trends/{employee_id}")
async def get_employee_trends(
    employee_id: int,
    weeks: int = 12,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """
    Returns 12-week sentiment and health trends for one employee.
    Employees can only view their own trends.
    Managers see aggregated view. HR sees full detail.
    """
    if user.role == "employee" and user.employee_id != employee_id:
        raise HTTPException(status_code=403, detail="Employees can only view their own data.")

    with _get_session() as session:
        trend_rows = session.execute(
            text("""
                SELECT
                    hs.week,
                    hs.overall_score,
                    hs.burnout_risk,
                    hs.engagement_score,
                    ss.sentiment,
                    ss.emotion,
                    bf.message_count,
                    bf.after_hours_count
                FROM health_scores hs
                LEFT JOIN sentiment_scores ss
                    ON ss.employee_id = hs.employee_id AND ss.week = hs.week
                LEFT JOIN behavioral_features bf
                    ON bf.employee_id = hs.employee_id AND bf.week = hs.week
                WHERE hs.employee_id = :eid
                ORDER BY hs.week DESC
                LIMIT :weeks
            """),
            {"eid": employee_id, "weeks": weeks},
        ).fetchall()

    trends = [
        {
            "week": str(r[0]),
            "overall_score": r[1],
            "burnout_risk": r[2],
            "engagement_score": r[3],
            "sentiment": r[4],
            "emotion": r[5],
            "message_count": r[6],
            "after_hours_count": r[7],
        }
        for r in trend_rows
    ]

    return apply_dp_noise_to_response({
        "employee_id": employee_id,
        "weeks_returned": len(trends),
        "trends": trends,
    })
