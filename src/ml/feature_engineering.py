"""
Behavioral Feature Engineering
Computes 14 behavioral features per employee per week from
message_metadata and sentiment_scores tables.
Stores results in behavioral_features table.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")

WORK_START = 8
WORK_END = 19
WORK_DAYS_PER_WEEK = 5


# ─────────────────────────────────────────────────────────────
# Raw data fetchers
# ─────────────────────────────────────────────────────────────

def _fetch_message_metadata(
    session: Session,
    employee_id: int,
    week_start: date,
    week_end: date,
) -> pd.DataFrame:
    rows = session.execute(
        text("""
            SELECT
                sender_id, receiver_id, timestamp,
                word_count, is_after_hours, response_time_hours
            FROM message_metadata
            WHERE sender_id = :eid
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"eid": employee_id, "start": week_start, "end": week_end},
    ).fetchall()
    return pd.DataFrame(
        rows,
        columns=["sender_id", "receiver_id", "timestamp",
                 "word_count", "is_after_hours", "response_time_hours"],
    )


def _fetch_received_metadata(
    session: Session,
    employee_id: int,
    week_start: date,
    week_end: date,
) -> pd.DataFrame:
    rows = session.execute(
        text("""
            SELECT sender_id, receiver_id, timestamp, word_count
            FROM message_metadata
            WHERE receiver_id = :eid
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"eid": employee_id, "start": week_start, "end": week_end},
    ).fetchall()
    return pd.DataFrame(
        rows,
        columns=["sender_id", "receiver_id", "timestamp", "word_count"],
    )


def _fetch_sentiment(session: Session, employee_id: int, week: date) -> Optional[dict]:
    row = session.execute(
        text("""
            SELECT sentiment, emotion
            FROM sentiment_scores
            WHERE employee_id = :eid AND week = :week
        """),
        {"eid": employee_id, "week": week},
    ).fetchone()
    if row:
        return {"sentiment": row[0], "emotion": row[1]}
    return None


def _fetch_prev_sentiment(session: Session, employee_id: int, week: date) -> Optional[float]:
    prev_week = week - timedelta(weeks=1)
    row = session.execute(
        text("""
            SELECT sentiment
            FROM sentiment_scores
            WHERE employee_id = :eid AND week = :week
        """),
        {"eid": employee_id, "week": prev_week},
    ).fetchone()
    return float(row[0]) if row else None


def _fetch_manager_id(session: Session, employee_id: int) -> Optional[int]:
    row = session.execute(
        text("SELECT manager_id FROM employees WHERE id = :eid"),
        {"eid": employee_id},
    ).fetchone()
    return row[0] if row else None


# ─────────────────────────────────────────────────────────────
# Per-feature computations
# ─────────────────────────────────────────────────────────────

def _avg_response_hours(sent_df: pd.DataFrame) -> float:
    valid = sent_df["response_time_hours"].dropna()
    return round(float(valid.mean()), 2) if len(valid) > 0 else 0.0


def _after_hours_count(sent_df: pd.DataFrame) -> int:
    return int(sent_df["is_after_hours"].sum())


def _message_count(sent_df: pd.DataFrame) -> int:
    return len(sent_df)


def _avg_message_length(sent_df: pd.DataFrame) -> float:
    valid = sent_df["word_count"].dropna()
    return round(float(valid.mean()), 2) if len(valid) > 0 else 0.0


def _participation_rate(sent_df: pd.DataFrame) -> float:
    """Active days / 5 workdays."""
    if sent_df.empty:
        return 0.0
    active_days = sent_df["timestamp"].dt.date.nunique()
    return round(min(active_days / WORK_DAYS_PER_WEEK, 1.0), 4)


def _manager_comm_ratio(
    sent_df: pd.DataFrame,
    manager_id: Optional[int],
) -> float:
    """Messages to manager / total messages sent."""
    if sent_df.empty or manager_id is None:
        return 0.0
    to_manager = sent_df[sent_df["receiver_id"] == manager_id]
    return round(len(to_manager) / max(len(sent_df), 1), 4)


def _sentiment_velocity(
    current_sentiment: Optional[float],
    prev_sentiment: Optional[float],
) -> float:
    """Rate of sentiment change from previous week."""
    if current_sentiment is None or prev_sentiment is None:
        return 0.0
    return round(current_sentiment - prev_sentiment, 4)


def _response_rate(sent_df: pd.DataFrame, received_df: pd.DataFrame) -> float:
    """% of received messages that the employee replied to."""
    received_count = len(received_df)
    if received_count == 0:
        return 1.0
    replied = len(sent_df[sent_df["receiver_id"].isin(received_df["sender_id"])])
    return round(min(replied / received_count, 1.0), 4)


def _initiation_ratio(sent_df: pd.DataFrame, received_df: pd.DataFrame) -> float:
    """Sent / (sent + received)."""
    total = len(sent_df) + len(received_df)
    return round(len(sent_df) / max(total, 1), 4)


# ─────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────

def compute_features_for_employee(
    employee_id: int,
    week: date,
    session: Session,
    graph_centrality: float = 0.0,
    isolation_score: float = 0.0,
    thread_depth: float = 1.0,
) -> dict:
    """
    Compute all 14 behavioral features for one employee-week.

    Args:
        employee_id:       DB employee ID.
        week:              ISO week start date (Monday).
        session:           Active SQLAlchemy session.
        graph_centrality:  Provided by graph layer after graph is built.
        isolation_score:   Provided by graph layer.
        thread_depth:      Average thread depth (injected from graph layer).

    Returns:
        Feature dict with 14 keys.
    """
    week_end = week + timedelta(weeks=1)

    sent_df = _fetch_message_metadata(session, employee_id, week, week_end)
    received_df = _fetch_received_metadata(session, employee_id, week, week_end)

    # Convert timestamps
    if not sent_df.empty:
        sent_df["timestamp"] = pd.to_datetime(sent_df["timestamp"], utc=True)
    if not received_df.empty:
        received_df["timestamp"] = pd.to_datetime(received_df["timestamp"], utc=True)

    sentiment_data = _fetch_sentiment(session, employee_id, week)
    prev_sentiment = _fetch_prev_sentiment(session, employee_id, week)
    manager_id = _fetch_manager_id(session, employee_id)

    current_sentiment = sentiment_data["sentiment"] if sentiment_data else 0.0
    dominant_emotion = sentiment_data["emotion"] if sentiment_data else "neutral"

    features = {
        "employee_id": employee_id,
        "week": week,
        # 1
        "avg_response_hours": _avg_response_hours(sent_df),
        # 2
        "after_hours_count": _after_hours_count(sent_df),
        # 3
        "message_count": _message_count(sent_df),
        # 4
        "avg_message_length": _avg_message_length(sent_df),
        # 5
        "participation_rate": _participation_rate(sent_df),
        # 6
        "manager_comm_ratio": _manager_comm_ratio(sent_df, manager_id),
        # 7
        "sentiment_score": current_sentiment,
        # 8
        "sentiment_velocity": _sentiment_velocity(current_sentiment, prev_sentiment),
        # 9
        "dominant_emotion": dominant_emotion,
        # 10
        "graph_centrality": round(graph_centrality, 4),
        # 11
        "isolation_score": round(isolation_score, 4),
        # 12
        "response_rate": _response_rate(sent_df, received_df),
        # 13
        "initiation_ratio": _initiation_ratio(sent_df, received_df),
        # 14
        "thread_depth": round(thread_depth, 2),
    }
    return features


def upsert_behavioral_features(features: dict, session: Session) -> None:
    """Write computed features to behavioral_features table."""
    session.execute(
        text("""
            INSERT INTO behavioral_features
                (employee_id, week, avg_response_hours, after_hours_count,
                 message_count, avg_message_length, participation_rate,
                 manager_comm_ratio, sentiment_velocity)
            VALUES
                (:employee_id, :week, :avg_response_hours, :after_hours_count,
                 :message_count, :avg_message_length, :participation_rate,
                 :manager_comm_ratio, :sentiment_velocity)
            ON CONFLICT (employee_id, week)
            DO UPDATE SET
                avg_response_hours  = EXCLUDED.avg_response_hours,
                after_hours_count   = EXCLUDED.after_hours_count,
                message_count       = EXCLUDED.message_count,
                avg_message_length  = EXCLUDED.avg_message_length,
                participation_rate  = EXCLUDED.participation_rate,
                manager_comm_ratio  = EXCLUDED.manager_comm_ratio,
                sentiment_velocity  = EXCLUDED.sentiment_velocity
        """),
        {
            "employee_id": features["employee_id"],
            "week": features["week"],
            "avg_response_hours": features["avg_response_hours"],
            "after_hours_count": features["after_hours_count"],
            "message_count": features["message_count"],
            "avg_message_length": features["avg_message_length"],
            "participation_rate": features["participation_rate"],
            "manager_comm_ratio": features["manager_comm_ratio"],
            "sentiment_velocity": features["sentiment_velocity"],
        },
    )


def compute_features_for_week(week: date) -> int:
    """
    Compute and store behavioral features for all active employees for a given week.
    Called by the Celery weekly job.

    Returns:
        Number of employees processed.
    """
    engine = create_engine(DATABASE_URL)
    processed = 0

    with Session(engine) as session:
        employee_ids_rows = session.execute(
            text("""
                SELECT DISTINCT sender_id
                FROM message_metadata
                WHERE DATE_TRUNC('week', timestamp) = :week
            """),
            {"week": week},
        ).fetchall()

        employee_ids = [r[0] for r in employee_ids_rows]
        logger.info(f"Computing features for {len(employee_ids)} employees for week {week}")

        for emp_id in employee_ids:
            try:
                features = compute_features_for_employee(emp_id, week, session)
                upsert_behavioral_features(features, session)
                processed += 1
            except Exception as exc:
                logger.error(f"Feature computation failed for employee {emp_id}: {exc}")
                session.rollback()

        session.commit()

    logger.info(f"Behavioral feature computation complete: {processed} employees for week {week}")
    return processed
