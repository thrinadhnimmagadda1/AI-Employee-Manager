"""
Graph Visualization
Exports weekly communication graph as JSON for the D3.js frontend.
Node color encodes burnout risk. Edge color encodes relationship health.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Optional

import networkx as nx
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")


def _risk_to_node_color(burnout_risk: float) -> str:
    """Map burnout risk [0,1] to hex color: green → yellow → red."""
    if burnout_risk < 0.35:
        return "#22c55e"   # green
    elif burnout_risk < 0.65:
        return "#eab308"   # yellow
    else:
        return "#ef4444"   # red


def _health_to_edge_color(relationship_health: float) -> str:
    """Map relationship health [0,1] to edge color."""
    if relationship_health > 0.65:
        return "#3b82f6"   # blue (healthy)
    elif relationship_health > 0.35:
        return "#f97316"   # orange (declining)
    else:
        return "#ef4444"   # red (conflict)


def _activity_to_node_size(message_count: int, max_count: int) -> float:
    """Scale node size between 8 and 28 based on message activity."""
    if max_count == 0:
        return 12.0
    ratio = message_count / max_count
    return round(8.0 + ratio * 20.0, 1)


def build_graph_json(week: date, session: Optional[Session] = None) -> dict:
    """
    Build a D3.js-compatible force graph JSON for the given week.

    Returns:
        {
            "nodes": [
                {
                    "id": int,
                    "name": str (anonymized),
                    "department": str,
                    "burnout_risk": float,
                    "color": str (hex),
                    "size": float,
                    "message_count": int,
                    "overall_score": int,
                }
            ],
            "links": [
                {
                    "source": int,
                    "target": int,
                    "weight": int,
                    "avg_sentiment": float,
                    "relationship_health": float,
                    "color": str (hex),
                    "thickness": float,
                }
            ],
            "week": str,
            "stats": {
                "total_nodes": int,
                "total_edges": int,
                "isolated_count": int,
            }
        }
    """
    engine = create_engine(DATABASE_URL) if session is None else None
    own_session = session is None
    if own_session:
        session = Session(engine)

    try:
        # Fetch employees with their latest health scores
        employee_rows = session.execute(
            text("""
                SELECT
                    e.id,
                    e.name,
                    e.department,
                    COALESCE(hs.burnout_risk, 0.0)     AS burnout_risk,
                    COALESCE(hs.overall_score, 50)     AS overall_score
                FROM employees e
                LEFT JOIN health_scores hs
                    ON hs.employee_id = e.id
                   AND hs.week = (
                       SELECT MAX(week) FROM health_scores WHERE employee_id = e.id
                   )
            """),
        ).fetchall()

        # Fetch message counts per employee for this week
        activity_rows = session.execute(
            text("""
                SELECT sender_id, COUNT(*) AS msg_count
                FROM message_metadata
                WHERE DATE_TRUNC('week', timestamp) = :week
                GROUP BY sender_id
            """),
            {"week": week},
        ).fetchall()
        activity_map = {r[0]: r[1] for r in activity_rows}
        max_activity = max(activity_map.values()) if activity_map else 1

        # Fetch graph edges for this week
        edge_rows = session.execute(
            text("""
                SELECT
                    employee_a,
                    employee_b,
                    message_count,
                    avg_sentiment,
                    relationship_health
                FROM comm_graph
                WHERE week = :week
            """),
            {"week": week},
        ).fetchall()

    finally:
        if own_session:
            session.close()

    # Build nodes
    employee_ids_in_graph = set()
    nodes = []
    for row in employee_rows:
        emp_id, name, dept, burnout_risk, overall_score = row
        msg_count = activity_map.get(emp_id, 0)
        employee_ids_in_graph.add(emp_id)
        nodes.append({
            "id": emp_id,
            "name": name,
            "department": dept or "Unknown",
            "burnout_risk": round(float(burnout_risk), 3),
            "overall_score": int(overall_score),
            "color": _risk_to_node_color(float(burnout_risk)),
            "size": _activity_to_node_size(msg_count, max_activity),
            "message_count": msg_count,
        })

    # Build links
    max_edge_weight = max((r[2] for r in edge_rows), default=1)
    links = []
    for row in edge_rows:
        src, dst, msg_count, avg_sent, rel_health = row
        if src not in employee_ids_in_graph or dst not in employee_ids_in_graph:
            continue
        thickness = round(1.0 + (msg_count / max_edge_weight) * 5.0, 1)
        links.append({
            "source": src,
            "target": dst,
            "weight": int(msg_count),
            "avg_sentiment": round(float(avg_sent or 0.0), 3),
            "relationship_health": round(float(rel_health or 0.5), 3),
            "color": _health_to_edge_color(float(rel_health or 0.5)),
            "thickness": thickness,
        })

    isolated_count = sum(1 for n in nodes if n["message_count"] == 0)

    result = {
        "nodes": nodes,
        "links": links,
        "week": str(week),
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "isolated_count": isolated_count,
        },
    }

    logger.info(
        f"Graph JSON for week {week}: {len(nodes)} nodes, {len(links)} edges"
    )
    return result
