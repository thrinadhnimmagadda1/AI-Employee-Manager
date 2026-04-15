"""
Network Analyzer
Computes centrality, clustering, isolation, bottleneck detection,
and relationship health trends from weekly communication graphs.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import networkx as nx
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.graph.graph_builder import build_weekly_graph

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")

ISOLATION_THRESHOLD = 0.05       # degree centrality below this → isolated
BOTTLENECK_THRESHOLD = 0.20      # betweenness centrality above this → bottleneck
FREQ_DROP_THRESHOLD = 0.50       # 50%+ message frequency drop → flagged
SENTIMENT_DECLINE_THRESHOLD = -0.2  # week-over-week sentiment decline → flagged


# ─────────────────────────────────────────────────────────────
# Centrality computations
# ─────────────────────────────────────────────────────────────

def compute_degree_centrality(G: nx.DiGraph) -> dict[int, float]:
    """Degree centrality (in + out) for each node."""
    undirected = G.to_undirected()
    return nx.degree_centrality(undirected)


def compute_betweenness_centrality(G: nx.DiGraph) -> dict[int, float]:
    """Betweenness centrality — identifies bridge employees."""
    return nx.betweenness_centrality(G, weight="weight", normalized=True)


def compute_clustering_coefficient(G: nx.DiGraph) -> dict[int, float]:
    """Local clustering coefficient per node."""
    undirected = G.to_undirected()
    return nx.clustering(undirected, weight="weight")


# ─────────────────────────────────────────────────────────────
# Isolation and bottleneck detection
# ─────────────────────────────────────────────────────────────

def identify_isolated_employees(
    degree_centrality: dict[int, float],
    threshold: float = ISOLATION_THRESHOLD,
) -> list[int]:
    """Return employee IDs whose degree centrality is below threshold."""
    return [
        emp_id
        for emp_id, centrality in degree_centrality.items()
        if centrality < threshold
    ]


def identify_bottlenecks(
    betweenness: dict[int, float],
    threshold: float = BOTTLENECK_THRESHOLD,
) -> list[int]:
    """Return employee IDs with high betweenness — overloaded connectors."""
    return [
        emp_id
        for emp_id, bw in betweenness.items()
        if bw > threshold
    ]


# ─────────────────────────────────────────────────────────────
# Trend-based edge analysis
# ─────────────────────────────────────────────────────────────

def detect_declining_relationships(
    current_G: nx.DiGraph,
    prev_G: nx.DiGraph,
    sentiment_threshold: float = SENTIMENT_DECLINE_THRESHOLD,
    freq_threshold: float = FREQ_DROP_THRESHOLD,
) -> list[dict]:
    """
    Compare current and previous week graphs to find deteriorating edges.

    Returns:
        List of dicts describing declining pairs:
        {employee_a, employee_b, reason, current_sentiment, prev_sentiment}
    """
    declining = []

    for src, dst, data in current_G.edges(data=True):
        curr_sentiment = data.get("avg_sentiment", 0.0)
        curr_weight = data.get("weight", 0)

        if prev_G.has_edge(src, dst):
            prev_data = prev_G[src][dst]
            prev_sentiment = prev_data.get("avg_sentiment", 0.0)
            prev_weight = prev_data.get("weight", 1)

            sentiment_delta = curr_sentiment - prev_sentiment
            freq_ratio = curr_weight / max(prev_weight, 1)

            reasons = []
            if sentiment_delta < sentiment_threshold:
                reasons.append(f"sentiment_decline({sentiment_delta:.2f})")
            if freq_ratio < (1.0 - freq_threshold):
                reasons.append(f"frequency_drop({(1-freq_ratio)*100:.0f}%)")

            if reasons:
                declining.append({
                    "employee_a": src,
                    "employee_b": dst,
                    "reason": ", ".join(reasons),
                    "current_sentiment": curr_sentiment,
                    "prev_sentiment": prev_sentiment,
                    "freq_ratio": round(freq_ratio, 2),
                })

    return declining


# ─────────────────────────────────────────────────────────────
# Per-employee network health score
# ─────────────────────────────────────────────────────────────

def compute_employee_network_health(
    employee_id: int,
    degree: float,
    betweenness: float,
    clustering: float,
    is_isolated: bool,
    is_bottleneck: bool,
    declining_edges: int,
) -> float:
    """
    Aggregate network metrics into a 0.0–1.0 health score.

    High degree + moderate betweenness + high clustering = healthy.
    Isolation → low score. Bottleneck → moderate penalty (overloaded).
    """
    base = (
        min(degree * 5, 1.0) * 0.40   # degree contributes 40%
        + clustering * 0.30             # clustering 30%
        + (0.5 - abs(betweenness - 0.1)) * 0.20  # moderate betweenness is ideal
        + max(0.0, 0.10 - declining_edges * 0.03)  # 10% penalty for declining edges
    )
    if is_isolated:
        base *= 0.5
    if is_bottleneck:
        base *= 0.85  # slight penalty for being overloaded
    return round(max(0.0, min(1.0, base)), 4)


# ─────────────────────────────────────────────────────────────
# Full weekly network analysis
# ─────────────────────────────────────────────────────────────

def analyze_network_for_week(week: date) -> dict[int, dict]:
    """
    Run complete network analysis for a given week.

    Returns:
        Dict mapping employee_id → {
            degree_centrality, betweenness_centrality,
            clustering_coefficient, is_isolated, is_bottleneck,
            network_health_score, declining_edges
        }
    """
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        current_G = build_weekly_graph(week, session=session)
        prev_G = build_weekly_graph(week - timedelta(weeks=1), session=session)

    if current_G.number_of_nodes() == 0:
        logger.warning(f"Empty graph for week {week}")
        return {}

    degree = compute_degree_centrality(current_G)
    betweenness = compute_betweenness_centrality(current_G)
    clustering = compute_clustering_coefficient(current_G)

    isolated = set(identify_isolated_employees(degree))
    bottlenecks = set(identify_bottlenecks(betweenness))
    declining = detect_declining_relationships(current_G, prev_G)

    # Count declining edges per employee
    declining_count: dict[int, int] = {}
    for d in declining:
        for key in ("employee_a", "employee_b"):
            emp = d[key]
            declining_count[emp] = declining_count.get(emp, 0) + 1

    results: dict[int, dict] = {}
    for emp_id in current_G.nodes():
        d = degree.get(emp_id, 0.0)
        bw = betweenness.get(emp_id, 0.0)
        cl = clustering.get(emp_id, 0.0)
        is_iso = emp_id in isolated
        is_bot = emp_id in bottlenecks
        dec = declining_count.get(emp_id, 0)

        results[emp_id] = {
            "degree_centrality": round(d, 4),
            "betweenness_centrality": round(bw, 4),
            "clustering_coefficient": round(cl, 4),
            "is_isolated": is_iso,
            "is_bottleneck": is_bot,
            "declining_edges": dec,
            "network_health_score": compute_employee_network_health(
                emp_id, d, bw, cl, is_iso, is_bot, dec
            ),
        }

    logger.info(
        f"Network analysis week {week}: {len(results)} employees | "
        f"{len(isolated)} isolated | {len(bottlenecks)} bottlenecks | "
        f"{len(declining)} declining pairs"
    )
    return results
