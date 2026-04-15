import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getTeamDashboard } from "../api/cogniClient";
import HealthScoreCard from "../components/HealthScoreCard";
import NetworkGraph from "../components/NetworkGraph";
import AlertBadge from "../components/AlertBadge";

export default function TeamHealth() {
  const { teamId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getTeamDashboard(teamId)
      .then(setData)
      .catch((err) => setError(err.response?.data?.detail || "Failed to load team data"))
      .finally(() => setLoading(false));
  }, [teamId]);

  if (loading) return <div className="p-8 text-gray-500 animate-pulse">Loading team data…</div>;
  if (error) return <div className="p-8 text-red-400">{error}</div>;

  const { members = [], top_alerts = [], avg_health_score, member_count, graph_data } = data || {};

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <button onClick={() => navigate("/")} className="btn-ghost text-sm">
          ← Back
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white">Team {teamId} Health</h1>
          <p className="text-gray-500 text-sm mt-1">
            {member_count} members · Avg score: {avg_health_score?.toFixed(0) ?? "—"}/100
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Members grid */}
        <div className="lg:col-span-2 space-y-6">
          <div>
            <h2 className="text-lg font-semibold text-white mb-4">Team Members</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {members.map((m) => (
                <HealthScoreCard key={m.employee_id} employee={{ id: m.employee_id, ...m }} />
              ))}
            </div>
          </div>

          {/* Network Graph */}
          <div className="card">
            <h2 className="text-lg font-semibold text-white mb-4">Communication Network</h2>
            <NetworkGraph graphData={graph_data} />
            <div className="flex gap-6 mt-3 text-xs text-gray-500">
              <span><span className="inline-block w-3 h-3 rounded-full bg-green-500 mr-1" />Healthy</span>
              <span><span className="inline-block w-3 h-3 rounded-full bg-yellow-500 mr-1" />Monitor</span>
              <span><span className="inline-block w-3 h-3 rounded-full bg-red-500 mr-1" />At Risk</span>
            </div>
          </div>
        </div>

        {/* Alerts sidebar */}
        <div>
          <h2 className="text-lg font-semibold text-white mb-4">Active Alerts</h2>
          <div className="space-y-3">
            {top_alerts.length === 0 ? (
              <div className="card text-center text-gray-500 text-sm py-8">
                No alerts for this team
              </div>
            ) : (
              top_alerts.map((a, i) => (
                <div key={i} className="card">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <span className="text-sm font-medium text-white capitalize">{a.type}</span>
                    <AlertBadge severity={a.severity} />
                  </div>
                  <p className="text-xs text-gray-400 mt-1 line-clamp-2">{a.description}</p>
                  <button
                    onClick={() => navigate(`/employee/${a.employee_id}`)}
                    className="text-xs text-cogni-400 mt-2 hover:text-cogni-300 transition-colors"
                  >
                    View employee →
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
