"""
Communication Graph Builder
Builds weekly directed NetworkX graphs from message_metadata.
Nodes = employees, edges = communication with sentiment and frequency weights.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")
DATA_PROCESSED = Path(os.getenv("DATA_DIR", "data")) / "processed"


# ─────────────────────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────────────────────

def _fetch_week_messages(session: Session, week_start: date) -> pd.DataFrame:
    """Fetch all message metadata for a given ISO week."""
    week_end = week_start + timedelta(weeks=1)
    rows = session.execute(
        text("""
            SELECT
                mm.sender_id,
                mm.receiver_id,
                mm.response_time_hours,
                COALESCE(ss_s.sentiment, 0.0) AS sender_sentiment
            FROM message_metadata mm
            LEFT JOIN sentiment_scores ss_s
                ON ss_s.employee_id = mm.sender_id
               AND ss_s.week = DATE_TRUNC('week', mm.timestamp)::date
            WHERE mm.timestamp >= :start AND mm.timestamp < :end
        """),
        {"start": week_start, "end": week_end},
    ).fetchall()

    return pd.DataFrame(
        rows,
        columns=["sender_id", "receiver_id", "response_time_hours", "sender_sentiment"],
    )


def _fetch_all_employees(session: Session) -> list[int]:
    rows = session.execute(text("SELECT id FROM employees")).fetchall()
    return [r[0] for r in rows]


# ─────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────

def build_weekly_graph(week_start: date, session: Optional[Session] = None) -> nx.DiGraph:
    """
    Build a directed weighted communication graph for the given week.

    Edge attributes:
        weight           — message count between pair
        avg_sentiment    — mean sender sentiment on this edge
        avg_response_h   — mean response time in hours

    Returns:
        nx.DiGraph with employee IDs as nodes.
    """
    engine = create_engine(DATABASE_URL) if session is None else None
    own_session = session is None
    if own_session:
        session = Session(engine)

    try:
        df = _fetch_week_messages(session, week_start)
        employee_ids = _fetch_all_employees(session)
    finally:
        if own_session:
            session.close()

    G = nx.DiGraph()

    # Add all known employees as nodes
    for eid in employee_ids:
        G.add_node(eid)

    if df.empty:
        logger.warning(f"No messages found for week {week_start}")
        return G

    # Aggregate per (sender, receiver) pair
    pair_stats = (
        df.groupby(["sender_id", "receiver_id"])
        .agg(
            message_count=("sender_id", "count"),
            avg_sentiment=("sender_sentiment", "mean"),
            avg_response_h=("response_time_hours", "mean"),
        )
        .reset_index()
    )

    for _, row in pair_stats.iterrows():
        src = int(row["sender_id"])
        dst = int(row["receiver_id"])
        if src == dst:
            continue

        G.add_edge(
            src,
            dst,
            weight=int(row["message_count"]),
            avg_sentiment=round(float(row["avg_sentiment"]), 4),
            avg_response_h=round(float(row["avg_response_h"]) if pd.notna(row["avg_response_h"]) else 0.0, 2),
        )

    logger.info(
        f"Graph for week {week_start}: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


# ─────────────────────────────────────────────────────────────
# Database persistence
# ─────────────────────────────────────────────────────────────

def save_graph_to_db(G: nx.DiGraph, week: date, session: Session) -> None:
    """Upsert comm_graph rows from the graph edges."""
    for src, dst, data in G.edges(data=True):
        # Relationship health: blend of sentiment and response time normalisation
        sentiment = data.get("avg_sentiment", 0.0)
        resp_h = data.get("avg_response_h", 0.0)
        # Normalize response time: <1h = 1.0, >24h = 0.0
        resp_score = max(0.0, 1.0 - resp_h / 24.0)
        rel_health = round((sentiment + 1.0) / 2.0 * 0.7 + resp_score * 0.3, 4)
        rel_health = max(0.0, min(1.0, rel_health))

        try:
            session.execute(
                text("""
                    INSERT INTO comm_graph
                        (employee_a, employee_b, week, message_count,
                         avg_sentiment, avg_response_hours, relationship_health)
                    VALUES
                        (:a, :b, :week, :mc, :as, :arh, :rh)
                    ON CONFLICT (employee_a, employee_b, week)
                    DO UPDATE SET
                        message_count       = EXCLUDED.message_count,
                        avg_sentiment       = EXCLUDED.avg_sentiment,
                        avg_response_hours  = EXCLUDED.avg_response_hours,
                        relationship_health = EXCLUDED.relationship_health
                """),
                {
                    "a": src,
                    "b": dst,
                    "week": week,
                    "mc": data.get("weight", 1),
                    "as": data.get("avg_sentiment", 0.0),
                    "arh": data.get("avg_response_h", 0.0),
                    "rh": rel_health,
                },
            )
        except Exception as exc:
            logger.debug(f"Graph edge insert error ({src}→{dst}): {exc}")
            session.rollback()


def export_graph_csv(G: nx.DiGraph, week: date) -> Path:
    """Export graph edges to data/processed/graph_edges.csv."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, dst, data in G.edges(data=True):
        rows.append({
            "week": str(week),
            "source": src,
            "target": dst,
            "weight": data.get("weight", 1),
            "avg_sentiment": data.get("avg_sentiment", 0.0),
            "avg_response_h": data.get("avg_response_h", 0.0),
        })
    df = pd.DataFrame(rows)
    out = DATA_PROCESSED / "graph_edges.csv"
    df.to_csv(out, index=False)
    logger.info(f"Exported {len(rows)} graph edges to {out}")
    return out


def build_and_persist_weekly_graph(week: date) -> nx.DiGraph:
    """Convenience function: build graph, save to DB and CSV."""
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        G = build_weekly_graph(week, session=session)
        save_graph_to_db(G, week, session)
        session.commit()
    export_graph_csv(G, week)
    return G
