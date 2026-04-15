"""
Analyst Agent
Queries the PostgreSQL database for relevant health, alert, and behavioral data.
Returns structured data summaries for the psychologist agent.
"""
from __future__ import annotations

import os
from typing import Optional

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")


def _get_session() -> Session:
    engine = create_engine(DATABASE_URL)
    return Session(engine)


def get_employee_health_summary(employee_id: int) -> dict:
    """Fetch latest health scores and behavioral features for one employee."""
    with _get_session() as session:
        hs_row = session.execute(
            text("""
                SELECT
                    burnout_risk, attrition_risk_30d, attrition_risk_60d,
                    attrition_risk_90d, conflict_risk, engagement_score,
                    overall_score, shap_values, flags, week
                FROM health_scores
                WHERE employee_id = :eid
                ORDER BY week DESC
                LIMIT 1
            """),
            {"eid": employee_id},
        ).fetchone()

        bf_row = session.execute(
            text("""
                SELECT
                    avg_response_hours, after_hours_count, message_count,
                    avg_message_length, participation_rate, manager_comm_ratio,
                    sentiment_velocity, week
                FROM behavioral_features
                WHERE employee_id = :eid
                ORDER BY week DESC
                LIMIT 1
            """),
            {"eid": employee_id},
        ).fetchone()

        emp_row = session.execute(
            text("SELECT name, department, role, tenure_months FROM employees WHERE id = :eid"),
            {"eid": employee_id},
        ).fetchone()

        alerts_rows = session.execute(
            text("""
                SELECT alert_type, severity, description
                FROM alerts
                WHERE employee_id = :eid AND resolved = FALSE
                ORDER BY created_at DESC
                LIMIT 5
            """),
            {"eid": employee_id},
        ).fetchall()

    result = {"employee_id": employee_id}

    if emp_row:
        result["profile"] = {
            "name": emp_row[0],
            "department": emp_row[1],
            "role": emp_row[2],
            "tenure_months": emp_row[3],
        }

    if hs_row:
        result["health_scores"] = {
            "burnout_risk": hs_row[0],
            "attrition_risk_30d": hs_row[1],
            "attrition_risk_60d": hs_row[2],
            "attrition_risk_90d": hs_row[3],
            "conflict_risk": hs_row[4],
            "engagement_score": hs_row[5],
            "overall_score": hs_row[6],
            "shap_values": hs_row[7],
            "flags": hs_row[8],
            "week": str(hs_row[9]) if hs_row[9] else None,
        }

    if bf_row:
        result["behavioral_features"] = {
            "avg_response_hours": bf_row[0],
            "after_hours_count": bf_row[1],
            "message_count": bf_row[2],
            "avg_message_length": bf_row[3],
            "participation_rate": bf_row[4],
            "manager_comm_ratio": bf_row[5],
            "sentiment_velocity": bf_row[6],
            "week": str(bf_row[7]) if bf_row[7] else None,
        }

    result["active_alerts"] = [
        {"type": r[0], "severity": r[1], "description": r[2]}
        for r in alerts_rows
    ]

    return result


def get_team_health_summary(team_id: int) -> dict:
    """Fetch aggregated health scores for an entire team."""
    with _get_session() as session:
        rows = session.execute(
            text("""
                SELECT
                    e.id, e.name,
                    hs.burnout_risk, hs.overall_score,
                    hs.attrition_risk_90d, hs.conflict_risk,
                    hs.flags
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
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        ELSE 4
                    END
                LIMIT 10
            """),
            {"tid": team_id},
        ).fetchall()

    members = []
    for row in rows:
        members.append({
            "employee_id": row[0],
            "name": row[1],
            "burnout_risk": row[2],
            "overall_score": row[3],
            "attrition_risk_90d": row[4],
            "conflict_risk": row[5],
            "flags": row[6],
        })

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


def get_behavioral_trends(employee_id: int, weeks: int = 12) -> list[dict]:
    """Return last N weeks of behavioral features for trend analysis."""
    with _get_session() as session:
        rows = session.execute(
            text("""
                SELECT
                    bf.week,
                    bf.message_count,
                    bf.after_hours_count,
                    bf.participation_rate,
                    bf.sentiment_velocity,
                    ss.sentiment,
                    ss.emotion
                FROM behavioral_features bf
                LEFT JOIN sentiment_scores ss
                    ON ss.employee_id = bf.employee_id AND ss.week = bf.week
                WHERE bf.employee_id = :eid
                ORDER BY bf.week DESC
                LIMIT :weeks
            """),
            {"eid": employee_id, "weeks": weeks},
        ).fetchall()

    return [
        {
            "week": str(r[0]),
            "message_count": r[1],
            "after_hours_count": r[2],
            "participation_rate": r[3],
            "sentiment_velocity": r[4],
            "sentiment": r[5],
            "emotion": r[6],
        }
        for r in rows
    ]


def run(question: str, context: dict) -> dict:
    """
    Entry point called by coordinator.
    Extracts employee/team IDs from context and fetches relevant data.
    """
    employee_id = context.get("employee_id")
    team_id = context.get("team_id")

    data: dict = {"question": question}

    if employee_id:
        try:
            data["employee_summary"] = get_employee_health_summary(employee_id)
            data["behavioral_trends"] = get_behavioral_trends(employee_id)
        except Exception as exc:
            logger.error(f"Analyst agent error for employee {employee_id}: {exc}")
            data["error"] = str(exc)

    if team_id:
        try:
            data["team_summary"] = get_team_health_summary(team_id)
        except Exception as exc:
            logger.error(f"Analyst agent error for team {team_id}: {exc}")

    logger.info(f"Analyst agent completed data retrieval for question: {question[:60]!r}")
    return data
