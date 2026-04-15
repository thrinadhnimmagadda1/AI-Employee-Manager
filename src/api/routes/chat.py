"""
Chat API — data-driven insight endpoint.

Answers natural language questions about team health by querying the
PostgreSQL database directly and returning structured, number-backed
responses. No LLM dependency required; responses are generated from
real ML model outputs stored in the database.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.api.middleware.auth import TokenData, get_current_user

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    employee_id: Optional[int] = None
    team_id: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    data: dict
    insights: list[str]
    recommendations: list[str]


def _get_session() -> Session:
    return Session(create_engine(DATABASE_URL))


def _query_employee_insight(employee_id: int, session: Session) -> dict:
    """Fetch the latest health, behavioral, and alert data for one employee."""
    hs = session.execute(
        text("""
            SELECT e.name, e.department, e.role, e.tenure_months,
                   hs.burnout_risk, hs.attrition_risk_30d, hs.attrition_risk_60d,
                   hs.attrition_risk_90d, hs.conflict_risk, hs.engagement_score,
                   hs.overall_score, hs.shap_values, hs.week
            FROM health_scores hs
            JOIN employees e ON e.id = hs.employee_id
            WHERE hs.employee_id = :eid
            ORDER BY hs.week DESC
            LIMIT 1
        """),
        {"eid": employee_id},
    ).fetchone()

    bf = session.execute(
        text("""
            SELECT avg_response_hours, after_hours_count, message_count,
                   participation_rate, sentiment_velocity
            FROM behavioral_features
            WHERE employee_id = :eid
            ORDER BY week DESC LIMIT 1
        """),
        {"eid": employee_id},
    ).fetchone()

    alerts = session.execute(
        text("""
            SELECT alert_type, severity, description
            FROM alerts
            WHERE employee_id = :eid AND resolved = FALSE
            ORDER BY created_at DESC LIMIT 3
        """),
        {"eid": employee_id},
    ).fetchall()

    return {"health": hs, "behavioral": bf, "alerts": alerts}


def _build_employee_response(employee_id: int) -> ChatResponse:
    with _get_session() as session:
        d = _query_employee_insight(employee_id, session)

    hs = d["health"]
    bf = d["behavioral"]
    active_alerts = d["alerts"]

    if not hs:
        return ChatResponse(
            answer=f"No health score data found for employee {employee_id}. "
                   "Run the analytics pipeline first: python scripts/load_enron_data.py",
            data={},
            insights=[],
            recommendations=["Run scripts/load_enron_data.py to populate health scores"],
        )

    name        = hs[0]
    department  = hs[1]
    burnout     = hs[4] or 0.0
    attr_90d    = hs[7] or 0.0
    conflict    = hs[8] or 0.0
    engagement  = hs[9] or 0.0
    overall     = hs[10] or 50
    week        = str(hs[12])

    insights: list[str] = []
    recommendations: list[str] = []

    # Burnout signal
    if burnout >= 0.70:
        insights.append(f"⚠️  Burnout risk is CRITICAL at {burnout:.0%} — well above the 70% alert threshold.")
        recommendations.append("Schedule an immediate 1-on-1 to discuss workload and stress levels.")
        recommendations.append("Consider a temporary workload reduction or delegation of non-critical tasks.")
    elif burnout >= 0.45:
        insights.append(f"Burnout risk is elevated at {burnout:.0%}. Watch for continued increase.")
        recommendations.append("Check in informally this week before it escalates.")
    else:
        insights.append(f"Burnout risk is healthy at {burnout:.0%}.")

    # Attrition signal
    if attr_90d >= 0.65:
        insights.append(f"90-day attrition risk is {attr_90d:.0%} — this employee may leave within 3 months.")
        recommendations.append("Have a retention conversation: career path, compensation, recognition.")
    elif attr_90d >= 0.40:
        insights.append(f"90-day attrition risk is moderate at {attr_90d:.0%}.")
        recommendations.append("A career development conversation now can reduce this risk significantly.")

    # Behavioral signals
    if bf:
        after_hours = bf[1] or 0
        msg_count   = bf[2] or 0
        sent_vel    = bf[4] or 0.0
        if after_hours > 10:
            insights.append(f"Sent {after_hours} after-hours messages this week — a strong burnout signal.")
        if sent_vel < -0.3:
            insights.append("Sentiment has been declining week-over-week — communication tone is worsening.")
        if msg_count < 5:
            insights.append("Unusually low message volume — possible disengagement or isolation.")

    # Engagement
    if engagement < 0.35:
        insights.append(f"Engagement score is low at {engagement:.0%}.")
        recommendations.append("Involve this employee in a meaningful project or decision this week.")

    # Active alerts
    for alert in active_alerts:
        insights.append(f"Active {alert[1]} {alert[0]} alert: {alert[2]}")

    summary = (
        f"{name} ({department}) — Overall health score: {overall}/100 as of week {week}. "
        f"Burnout: {burnout:.0%} | Attrition (90d): {attr_90d:.0%} | "
        f"Conflict: {conflict:.0%} | Engagement: {engagement:.0%}."
    )

    return ChatResponse(
        answer=summary,
        data={
            "employee_id": employee_id,
            "name": name,
            "department": department,
            "overall_score": overall,
            "burnout_risk": round(burnout, 3),
            "attrition_risk_90d": round(attr_90d, 3),
            "conflict_risk": round(conflict, 3),
            "engagement_score": round(engagement, 3),
            "week": week,
            "active_alerts": len(active_alerts),
        },
        insights=insights,
        recommendations=recommendations,
    )


def _build_team_response(team_id: int) -> ChatResponse:
    with _get_session() as session:
        rows = session.execute(
            text("""
                SELECT e.name, hs.burnout_risk, hs.overall_score,
                       hs.attrition_risk_90d, hs.conflict_risk
                FROM health_scores hs
                JOIN employees e ON e.id = hs.employee_id
                WHERE hs.team_id = :tid
                  AND hs.week = (SELECT MAX(week) FROM health_scores WHERE team_id = :tid)
                ORDER BY hs.burnout_risk DESC NULLS LAST
            """),
            {"tid": team_id},
        ).fetchall()

        alert_count = session.execute(
            text("""
                SELECT COUNT(*) FROM alerts a
                JOIN health_scores hs ON hs.employee_id = a.employee_id
                WHERE hs.team_id = :tid AND a.resolved = FALSE
            """),
            {"tid": team_id},
        ).scalar() or 0

    if not rows:
        return ChatResponse(
            answer=f"No data found for team {team_id}.",
            data={}, insights=[], recommendations=[],
        )

    avg_burnout = sum(r[1] or 0 for r in rows) / len(rows)
    avg_score   = sum(r[2] or 50 for r in rows) / len(rows)
    high_risk   = [r[0] for r in rows if (r[1] or 0) >= 0.70]

    insights: list[str] = []
    recommendations: list[str] = []

    insights.append(f"Team has {len(rows)} members with an average health score of {avg_score:.0f}/100.")
    insights.append(f"Average burnout risk across the team: {avg_burnout:.0%}.")

    if high_risk:
        insights.append(f"{len(high_risk)} member(s) in critical burnout zone: {', '.join(high_risk)}.")
        recommendations.append(f"Prioritise 1-on-1s with: {', '.join(high_risk)}.")

    if alert_count > 0:
        insights.append(f"{alert_count} unresolved alerts active for this team.")
        recommendations.append("Review unresolved alerts in the Alerts tab.")

    if avg_burnout >= 0.55:
        recommendations.append("Consider a team-wide workload audit — average burnout is above safe range.")

    return ChatResponse(
        answer=(
            f"Team {team_id} summary — {len(rows)} members, "
            f"average health score {avg_score:.0f}/100, "
            f"average burnout risk {avg_burnout:.0%}, "
            f"{alert_count} active alerts."
        ),
        data={
            "team_id": team_id,
            "member_count": len(rows),
            "avg_health_score": round(avg_score, 1),
            "avg_burnout_risk": round(avg_burnout, 3),
            "active_alerts": alert_count,
            "high_risk_members": high_risk,
        },
        insights=insights,
        recommendations=recommendations,
    )


def _build_general_response(question: str) -> ChatResponse:
    """Fallback: return platform-wide stats when no specific context is given."""
    with _get_session() as session:
        stats = session.execute(
            text("""
                SELECT
                    COUNT(DISTINCT hs.employee_id)                          AS total_employees,
                    AVG(hs.burnout_risk)                                     AS avg_burnout,
                    AVG(hs.overall_score)                                    AS avg_score,
                    COUNT(a.id) FILTER (WHERE a.resolved = FALSE)            AS open_alerts,
                    COUNT(a.id) FILTER (WHERE a.severity = 'critical'
                                          AND a.resolved = FALSE)            AS critical_alerts
                FROM health_scores hs
                LEFT JOIN alerts a ON a.employee_id = hs.employee_id
                WHERE hs.week = (SELECT MAX(week) FROM health_scores)
            """),
        ).fetchone()

    if not stats or not stats[0]:
        return ChatResponse(
            answer="No data in the database yet. Run scripts/load_enron_data.py to populate it.",
            data={}, insights=[], recommendations=[],
        )

    total, avg_burnout, avg_score, open_alerts, critical = stats
    avg_burnout = avg_burnout or 0.0
    avg_score   = avg_score or 50

    return ChatResponse(
        answer=(
            f"Organisation overview: {total} employees tracked. "
            f"Average health score: {avg_score:.0f}/100. "
            f"Average burnout risk: {avg_burnout:.0%}. "
            f"{open_alerts} open alerts ({critical} critical)."
        ),
        data={
            "total_employees": total,
            "avg_health_score": round(float(avg_score), 1),
            "avg_burnout_risk": round(float(avg_burnout), 3),
            "open_alerts": open_alerts,
            "critical_alerts": critical,
        },
        insights=[
            f"Organisation-wide average burnout risk: {avg_burnout:.0%}.",
            f"{critical} employee(s) in the critical risk category require immediate attention.",
        ],
        recommendations=[
            "Use the Alerts tab to review and action critical cases.",
            "Filter the Dashboard by team to drill into specific groups.",
        ],
    )


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: TokenData = Depends(get_current_user),
) -> ChatResponse:
    """
    Answer a natural language question about team or employee health.

    - Provide `employee_id` for employee-specific insights.
    - Provide `team_id` for team-level summary.
    - Omit both for an organisation-wide overview.

    All responses are generated directly from ML model outputs stored in
    the database — burnout scores, attrition probabilities, SHAP flags,
    and behavioral features.
    """
    if user.role not in ("manager", "hr"):
        raise HTTPException(status_code=403, detail="Chat is only available to managers and HR.")

    if not body.question or not body.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    try:
        if body.employee_id:
            return _build_employee_response(body.employee_id)
        if body.team_id:
            return _build_team_response(body.team_id)
        return _build_general_response(body.question)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}") from exc
