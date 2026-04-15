"""
Graph API routes
=================
Serves D3.js-ready force-graph JSON for any historical week.
Lets the frontend animate how the communication network evolved over time.

Routes:
  GET /graph/weeks         — list every week that has graph data
  GET /graph/{week}        — D3 graph JSON for a specific week
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.api.middleware.auth import TokenData, get_current_user
from src.api.middleware.privacy import apply_dp_noise_to_response

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")

router = APIRouter(prefix="/graph", tags=["analytics"])


def _get_session() -> Session:
    return Session(create_engine(DATABASE_URL))


def _current_week() -> date:
    today = date.today()
    return today - timedelta(days=today.weekday())


# ─────────────────────────────────────────────────────────────────────────────
# GET /graph/weeks
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/weeks",
    summary="List available graph weeks",
    description=(
        "Returns every distinct week for which communication graph data exists in `comm_graph`. "
        "Use the returned date strings as the `{week}` parameter in `GET /graph/{week}`.\n\n"
        "Results are ordered newest-first."
    ),
)
async def list_graph_weeks(
    _user: TokenData = Depends(get_current_user),
) -> dict:
    with _get_session() as session:
        rows = session.execute(
            text("""
                SELECT DISTINCT week, COUNT(*) AS edge_count
                FROM comm_graph
                GROUP BY week
                ORDER BY week DESC
            """)
        ).fetchall()

    weeks = [{"week": str(r[0]), "edge_count": r[1]} for r in rows]
    return {
        "available_weeks": weeks,
        "total_weeks":     len(weeks),
        "current_week":    str(_current_week()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /graph/{week}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{week}",
    summary="D3 communication graph for a specific week",
    description=(
        "Returns a D3.js force-graph JSON for the given week, ready for the "
        "frontend `<NetworkGraph />` component.\n\n"
        "**`week` parameter:**\n"
        "- `current` — resolves to the current Monday\n"
        "- ISO date string `YYYY-MM-DD` — must be a Monday (auto-snapped if not)\n\n"
        "**Node fields:**\n"
        "- `id`, `name` (anonymised), `department`, `burnout_risk`, `overall_score`\n"
        "- `color` — hex colour encoding burnout risk (🟢 green < 35% · 🟡 yellow < 65% · 🔴 red)\n"
        "- `size` — proportional to weekly message count\n\n"
        "**Link fields:**\n"
        "- `source`, `target`, `weight` (message count), `avg_sentiment`, `relationship_health`\n"
        "- `color` — hex colour encoding relationship health (🔵 blue → 🟠 orange → 🔴 red)\n"
        "- `thickness` — proportional to message volume\n\n"
        "**`stats` block** — `total_nodes`, `total_edges`, `isolated_count`, `week_label`\n\n"
        "**`comparison`** — automatic delta vs the previous week "
        "(edge count change, avg health score change, new isolated employees).\n\n"
        "Applies differential-privacy noise to `burnout_risk` in node data."
    ),
)
async def get_graph_for_week(
    week: str,
    _user: TokenData = Depends(get_current_user),
) -> dict:
    # ── Parse week ────────────────────────────────────────────
    if week == "current":
        target_week = _current_week()
    else:
        try:
            target_week = date.fromisoformat(week)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid week format '{week}'. Use 'current' or 'YYYY-MM-DD'.",
            )
        # Auto-snap to Monday
        target_week = target_week - timedelta(days=target_week.weekday())

    # ── Build graph JSON ──────────────────────────────────────
    from src.graph.visualization import build_graph_json  # noqa: PLC0415
    try:
        graph_json = build_graph_json(target_week)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph build failed: {exc}") from exc

    # ── Previous-week comparison ──────────────────────────────
    prev_week = target_week - timedelta(weeks=1)
    comparison: dict = {}
    try:
        with _get_session() as session:
            prev_edges = session.execute(
                text("SELECT COUNT(*) FROM comm_graph WHERE week = :w"),
                {"w": prev_week},
            ).scalar() or 0

            curr_edges = graph_json["stats"]["total_edges"]
            edge_delta = curr_edges - prev_edges

            prev_avg = session.execute(
                text("""
                    SELECT AVG(overall_score)
                    FROM health_scores
                    WHERE week = (SELECT MAX(week) FROM health_scores WHERE week <= :w)
                """),
                {"w": prev_week},
            ).scalar()

            curr_avg = session.execute(
                text("""
                    SELECT AVG(overall_score)
                    FROM health_scores
                    WHERE week = (SELECT MAX(week) FROM health_scores WHERE week <= :w)
                """),
                {"w": target_week},
            ).scalar()

        prev_avg_f = round(float(prev_avg), 1) if prev_avg else None
        curr_avg_f = round(float(curr_avg), 1) if curr_avg else None
        health_delta = round(curr_avg_f - prev_avg_f, 1) if (prev_avg_f and curr_avg_f) else None

        comparison = {
            "prev_week":           str(prev_week),
            "edge_count_change":   edge_delta,
            "edge_count_prev":     prev_edges,
            "avg_health_score":    curr_avg_f,
            "avg_health_score_prev": prev_avg_f,
            "health_score_delta":  health_delta,
            "trend":               (
                "improving" if health_delta and health_delta > 2
                else "declining" if health_delta and health_delta < -2
                else "stable"
            ),
        }
    except Exception:
        comparison = {"error": "Comparison unavailable"}

    # ── Available weeks (for timeline navigation) ─────────────
    available_weeks: list[str] = []
    try:
        with _get_session() as session:
            rows = session.execute(
                text("SELECT DISTINCT week FROM comm_graph ORDER BY week DESC LIMIT 20")
            ).fetchall()
        available_weeks = [str(r[0]) for r in rows]
    except Exception:
        pass

    graph_json["stats"]["week_label"] = target_week.strftime("%b %d, %Y")
    graph_json["comparison"]           = comparison
    graph_json["available_weeks"]      = available_weeks
    graph_json["requested_week"]       = str(target_week)

    return apply_dp_noise_to_response(graph_json)
