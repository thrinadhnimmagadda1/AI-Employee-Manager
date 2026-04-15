"""Alerts API routes."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.api.middleware.auth import TokenData, get_current_user
from src.api.models.alert import AlertResolveRequest

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")
router = APIRouter(prefix="/alerts", tags=["alerts"])


def _get_session() -> Session:
    engine = create_engine(DATABASE_URL)
    return Session(engine)


@router.get("")
async def list_alerts(
    resolved: bool = False,
    severity: str | None = None,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """
    Returns all active alerts sorted by severity.
    Role: manager or hr.
    """
    if user.role not in ("manager", "hr"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    query = """
        SELECT
            a.id, a.employee_id, e.name AS employee_name,
            a.alert_type, a.severity, a.description,
            a.recommendations, a.resolved, a.created_at
        FROM alerts a
        INNER JOIN employees e ON e.id = a.employee_id
        WHERE a.resolved = :resolved
    """
    params: dict = {"resolved": resolved}

    if severity:
        query += " AND a.severity = :severity"
        params["severity"] = severity

    query += """
        ORDER BY
            CASE a.severity
                WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                WHEN 'medium' THEN 3 ELSE 4
            END,
            a.created_at DESC
    """

    with _get_session() as session:
        rows = session.execute(text(query), params).fetchall()

    alerts = [
        {
            "id": r[0],
            "employee_id": r[1],
            "employee_name": r[2],
            "alert_type": r[3],
            "severity": r[4],
            "description": r[5],
            "recommendations": r[6] or [],
            "resolved": r[7],
            "created_at": str(r[8]),
        }
        for r in rows
    ]

    return {"alerts": alerts, "total": len(alerts)}


@router.get("/{alert_id}")
async def get_alert(
    alert_id: int,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Full alert details with recommendations."""
    if user.role not in ("manager", "hr"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    with _get_session() as session:
        row = session.execute(
            text("""
                SELECT a.id, a.employee_id, e.name, a.alert_type,
                       a.severity, a.description, a.recommendations,
                       a.resolved, a.created_at
                FROM alerts a
                INNER JOIN employees e ON e.id = a.employee_id
                WHERE a.id = :aid
            """),
            {"aid": alert_id},
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {
        "id": row[0],
        "employee_id": row[1],
        "employee_name": row[2],
        "alert_type": row[3],
        "severity": row[4],
        "description": row[5],
        "recommendations": row[6] or [],
        "resolved": row[7],
        "created_at": str(row[8]),
    }


@router.patch("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    body: AlertResolveRequest,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Mark an alert as resolved."""
    if user.role not in ("manager", "hr"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    with _get_session() as session:
        result = session.execute(
            text("UPDATE alerts SET resolved = :resolved WHERE id = :aid RETURNING id"),
            {"resolved": body.resolved, "aid": alert_id},
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Alert not found")
        session.commit()

    return {"alert_id": alert_id, "resolved": body.resolved, "status": "updated"}
