import React, { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { getOverview, listAlerts } from "../api/cogniClient";
import AlertBadge from "../components/AlertBadge";

const POLL_INTERVAL = 5 * 60 * 1000; // 5 minutes

function StatCard({ label, value, sub, color = "text-white" }) {
  return (
    <div className="card">
      <div className={`text-3xl font-bold ${color}`}>{value ?? "—"}</div>
      <div className="text-sm font-medium text-gray-300 mt-1">{label}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

export default function Dashboard() {
  const [overview, setOverview] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const navigate = useNavigate();

  const fetchData = useCallback(async () => {
    try {
      const [ov, al] = await Promise.all([getOverview(), listAlerts(false)]);
      setOverview(ov);
      setAlerts((al.alerts || []).slice(0, 3));
      setLastUpdated(new Date());
    } catch (err) {
      console.error("Dashboard fetch error:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-500 text-sm animate-pulse">Loading platform overview…</div>
      </div>
    );
  }

  const criticalCount = alerts.filter((a) => a.severity === "critical").length;

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Platform Overview</h1>
          <p className="text-gray-500 text-sm mt-1">
            AI-powered team health insights
            {lastUpdated && ` · Updated ${lastUpdated.toLocaleTimeString()}`}
          </p>
        </div>
        <button onClick={() => navigate("/chat")} className="btn-primary">
          Ask AI Assistant
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Total Employees"
          value={overview?.total_employees}
          sub="being monitored"
        />
        <StatCard
          label="Active Alerts"
          value={overview?.total_active_alerts}
          color={overview?.total_active_alerts > 0 ? "text-orange-400" : "text-green-400"}
          sub="unresolved"
        />
        <StatCard
          label="Critical Alerts"
          value={overview?.critical_alerts}
          color={overview?.critical_alerts > 0 ? "text-red-400" : "text-green-400"}
          sub="need immediate action"
        />
        <StatCard
          label="Teams Monitored"
          value={overview?.teams?.length ?? 0}
          sub="departments tracked"
        />
      </div>

      {/* Teams grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Team Health Cards */}
        <div className="lg:col-span-2">
          <h2 className="text-lg font-semibold text-white mb-4">Team Health Scores</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {(overview?.teams || []).map((team) => (
              <button
                key={team.team_id}
                onClick={() => navigate(`/team/${team.team_id}`)}
                className="card text-left hover:border-cogni-500 transition-colors group"
              >
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors">
                    Team {team.team_id}
                  </div>
                  <span
                    className={`text-xl font-bold ${
                      team.avg_health_score >= 70
                        ? "text-green-400"
                        : team.avg_health_score >= 40
                        ? "text-yellow-400"
                        : "text-red-400"
                    }`}
                  >
                    {team.avg_health_score.toFixed(0)}
                  </span>
                </div>
                <div className="mt-2 flex items-center gap-3 text-xs text-gray-500">
                  <span>{team.member_count} members</span>
                  {team.active_alerts > 0 && (
                    <span className="text-orange-400">{team.active_alerts} alerts</span>
                  )}
                </div>
                <div className="mt-2 w-full h-1 bg-gray-800 rounded-full">
                  <div
                    className={`h-full rounded-full ${
                      team.avg_health_score >= 70
                        ? "bg-green-500"
                        : team.avg_health_score >= 40
                        ? "bg-yellow-500"
                        : "bg-red-500"
                    }`}
                    style={{ width: `${team.avg_health_score}%` }}
                  />
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Top 3 Critical Alerts */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">Critical Alerts</h2>
            <button
              onClick={() => navigate("/alerts")}
              className="text-xs text-cogni-400 hover:text-cogni-300 transition-colors"
            >
              View all →
            </button>
          </div>
          <div className="space-y-3">
            {alerts.length === 0 ? (
              <div className="card text-center text-gray-500 text-sm py-8">
                No active alerts 🎉
              </div>
            ) : (
              alerts.map((alert) => (
                <div key={alert.id} className="card">
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="text-sm font-medium text-white capitalize">
                      {alert.alert_type} alert
                    </div>
                    <AlertBadge severity={alert.severity} />
                  </div>
                  <p className="text-xs text-gray-400 line-clamp-2">{alert.description}</p>
                  <button
                    onClick={() => navigate(`/employee/${alert.employee_id}`)}
                    className="mt-2 text-xs text-cogni-400 hover:text-cogni-300 transition-colors"
                  >
                    {alert.employee_name} →
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
