"""
Audit Logger
Logs every database read and API call for GDPR compliance.
Has export function for generating compliance reports.
"""
from __future__ import annotations

import csv
import io
import os
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")


def log_database_read(
    user_id: Optional[int],
    table: str,
    record_id: Optional[int] = None,
    ip_address: str = "system",
) -> None:
    """
    Log a database read operation to the audit_log table.

    Args:
        user_id:    ID of the user performing the read.
        table:      Database table name that was read.
        record_id:  Optional specific record ID that was accessed.
        ip_address: Client IP address.
    """
    action = f"READ:{table}"
    target = f"{table}:{record_id}" if record_id else table
    _insert_log(user_id, action, target, ip_address)


def log_api_call(
    user_id: Optional[int],
    endpoint: str,
    params: Optional[dict] = None,
    ip_address: str = "unknown",
) -> None:
    """
    Log an API endpoint call to the audit_log table.
    """
    action = f"API:{endpoint}"
    target = str(params) if params else endpoint
    _insert_log(user_id, action, target, ip_address)


def log_model_inference(
    user_id: Optional[int],
    model_name: str,
    employee_id: Optional[int] = None,
) -> None:
    """Log when an ML model is run for inference on a specific employee."""
    action = f"INFERENCE:{model_name}"
    target = f"employee:{employee_id}" if employee_id else model_name
    _insert_log(user_id, action, target, "system")


def _insert_log(
    user_id: Optional[int],
    action: str,
    target: str,
    ip_address: str,
) -> None:
    try:
        engine = create_engine(DATABASE_URL)
        with Session(engine) as session:
            session.execute(
                text("""
                    INSERT INTO audit_log (user_id, action, target, ip_address)
                    VALUES (:uid, :action, :target, :ip)
                """),
                {
                    "uid": user_id,
                    "action": action[:100],
                    "target": target[:100],
                    "ip": ip_address[:45],
                },
            )
            session.commit()
    except Exception as exc:
        logger.debug(f"Audit log insert failed (non-critical): {exc}")


def export_audit_log_csv(
    user_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> str:
    """
    Export audit log as CSV for GDPR compliance reports.

    Args:
        user_id:    Filter by specific user (None = all users).
        start_date: Filter start timestamp.
        end_date:   Filter end timestamp.

    Returns:
        CSV string with audit log entries.
    """
    query = "SELECT id, user_id, action, target, timestamp, ip_address FROM audit_log WHERE 1=1"
    params: dict = {}

    if user_id is not None:
        query += " AND user_id = :uid"
        params["uid"] = user_id
    if start_date:
        query += " AND timestamp >= :start"
        params["start"] = start_date
    if end_date:
        query += " AND timestamp <= :end"
        params["end"] = end_date

    query += " ORDER BY timestamp DESC LIMIT 50000"

    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        rows = session.execute(text(query), params).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "user_id", "action", "target", "timestamp", "ip_address"])
    for row in rows:
        writer.writerow(row)

    logger.info(f"Audit log exported: {len(rows)} rows")
    return output.getvalue()


def get_access_summary(employee_id: int) -> dict:
    """
    Return a summary of who accessed a specific employee's data.
    Used for GDPR data subject access requests.
    """
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        rows = session.execute(
            text("""
                SELECT user_id, action, timestamp, ip_address
                FROM audit_log
                WHERE target LIKE :pattern
                ORDER BY timestamp DESC
                LIMIT 100
            """),
            {"pattern": f"%employee:{employee_id}%"},
        ).fetchall()

    return {
        "employee_id": employee_id,
        "total_accesses": len(rows),
        "accesses": [
            {"user_id": r[0], "action": r[1], "timestamp": str(r[2]), "ip": r[3]}
            for r in rows
        ],
    }
