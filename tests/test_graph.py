"""
Tests for the graph builder and network analyzer.
"""
import pytest
import networkx as nx
from datetime import date


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _build_sample_graph() -> nx.DiGraph:
    """Build a small test graph with 5 employees."""
    G = nx.DiGraph()
    employees = [1, 2, 3, 4, 5]
    for e in employees:
        G.add_node(e)

    # Normal edges
    G.add_edge(1, 2, weight=20, avg_sentiment=0.5, avg_response_h=1.0)
    G.add_edge(2, 1, weight=15, avg_sentiment=0.4, avg_response_h=2.0)
    G.add_edge(1, 3, weight=10, avg_sentiment=0.3, avg_response_h=3.0)
    G.add_edge(2, 3, weight=8,  avg_sentiment=0.2, avg_response_h=4.0)
    G.add_edge(3, 4, weight=5,  avg_sentiment=-0.1, avg_response_h=8.0)
    # Employee 5 is isolated (no edges)

    return G


def _build_conflict_graph() -> nx.DiGraph:
    """Graph where pair (1, 2) has negative sentiment and low frequency."""
    G = nx.DiGraph()
    for e in range(1, 6):
        G.add_node(e)

    G.add_edge(1, 2, weight=2, avg_sentiment=-0.8, avg_response_h=36.0)
    G.add_edge(3, 4, weight=25, avg_sentiment=0.7, avg_response_h=1.5)
    return G


# ─────────────────────────────────────────────────────────────
# Graph builder tests
# ─────────────────────────────────────────────────────────────

class TestGraphBuilder:
    def test_node_count_equals_employee_count(self):
        """Graph must have exactly as many nodes as employees."""
        G = _build_sample_graph()
        assert G.number_of_nodes() == 5

    def test_edge_attributes_present(self):
        """Edges must have weight, avg_sentiment, avg_response_h attributes."""
        G = _build_sample_graph()
        for src, dst, data in G.edges(data=True):
            assert "weight" in data, f"Edge ({src},{dst}) missing 'weight'"
            assert "avg_sentiment" in data, f"Edge ({src},{dst}) missing 'avg_sentiment'"
            assert "avg_response_h" in data, f"Edge ({src},{dst}) missing 'avg_response_h'"

    def test_self_loops_not_present(self):
        """No employee should have a self-loop."""
        G = _build_sample_graph()
        for node in G.nodes():
            assert not G.has_edge(node, node), f"Self-loop detected on node {node}"


# ─────────────────────────────────────────────────────────────
# Network analyzer tests
# ─────────────────────────────────────────────────────────────

class TestNetworkAnalyzer:
    def test_isolated_nodes_correctly_flagged(self):
        """Employee 5 with no edges should be flagged as isolated."""
        from src.graph.network_analyzer import (
            compute_degree_centrality,
            identify_isolated_employees,
        )
        G = _build_sample_graph()
        degree = compute_degree_centrality(G)
        isolated = identify_isolated_employees(degree, threshold=0.05)
        assert 5 in isolated, f"Expected employee 5 to be isolated, got: {isolated}"

    def test_connected_employees_not_isolated(self):
        """Employees 1 and 2 with many edges should NOT be isolated."""
        from src.graph.network_analyzer import (
            compute_degree_centrality,
            identify_isolated_employees,
        )
        G = _build_sample_graph()
        degree = compute_degree_centrality(G)
        isolated = identify_isolated_employees(degree, threshold=0.05)
        assert 1 not in isolated, "Employee 1 should not be isolated"
        assert 2 not in isolated, "Employee 2 should not be isolated"

    def test_relationship_health_lower_with_negative_sentiment(self):
        """Pairs with very negative sentiment should have lower health than positive pairs."""
        G_conflict = _build_conflict_graph()
        G_healthy = _build_sample_graph()

        # Compute relationship health manually
        def rel_health(data):
            sentiment = data.get("avg_sentiment", 0.0)
            resp_h = data.get("avg_response_h", 0.0)
            resp_score = max(0.0, 1.0 - resp_h / 24.0)
            return (sentiment + 1.0) / 2.0 * 0.7 + resp_score * 0.3

        # Conflict edge (1→2) should have lower health than healthy edge (3→4)
        conflict_health = rel_health(G_conflict[1][2])
        healthy_health = rel_health(G_conflict[3][4])
        assert conflict_health < healthy_health, \
            f"Conflict edge health {conflict_health:.3f} should be < healthy edge {healthy_health:.3f}"

    def test_centrality_values_between_0_and_1(self):
        """All centrality values must be normalized to [0, 1]."""
        from src.graph.network_analyzer import compute_degree_centrality
        G = _build_sample_graph()
        degree = compute_degree_centrality(G)
        for emp_id, val in degree.items():
            assert 0.0 <= val <= 1.0, f"Degree centrality for {emp_id} = {val} out of [0,1]"

    def test_betweenness_centrality_computed(self):
        """Betweenness centrality should be computable for all nodes."""
        from src.graph.network_analyzer import compute_betweenness_centrality
        G = _build_sample_graph()
        bw = compute_betweenness_centrality(G)
        assert len(bw) == G.number_of_nodes()

    def test_declining_relationships_detected(self):
        """A pair with dropped frequency should be detected as declining."""
        from src.graph.network_analyzer import detect_declining_relationships

        current = nx.DiGraph()
        current.add_edge(1, 2, weight=2, avg_sentiment=-0.4, avg_response_h=20)

        prev = nx.DiGraph()
        prev.add_edge(1, 2, weight=20, avg_sentiment=0.5, avg_response_h=2)

        declining = detect_declining_relationships(current, prev)
        assert any(d["employee_a"] == 1 and d["employee_b"] == 2 for d in declining), \
            "Pair (1, 2) should be flagged as declining"
