import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listAlerts, resolveAlert } from "../api/cogniClient";
import AlertBadge from "../components/AlertBadge";

const ALERT_TYPE_ICON = {
  burnout: "🔥",
  conflict: "⚡",
  attrition: "🚪",
  isolation: "🏝",
};

export default function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [showResolved, setShowResolved] = useState(false);
  const [loading, setLoading] = useState(true);
  const [resolving, setResolving] = useState(null);
  const navigate = useNavigate();

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      const data = await listAlerts(showResolved);
      setAlerts(data.alerts || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAlerts(); }, [showResolved]);

  const handleResolve = async (alertId) => {
    setResolving(alertId);
    try {
      await resolveAlert(alertId);
      setAlerts((prev) => prev.filter((a) => a.id !== alertId));
    } catch (err) {
      console.error(err);
    } finally {
      setResolving(null);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Alerts</h1>
          <p className="text-gray-500 text-sm mt-1">
            {alerts.length} {showResolved ? "resolved" : "active"} alerts
          </p>
        </div>
        <button
          onClick={() => setShowResolved((v) => !v)}
          className="btn-ghost text-sm"
        >
          {showResolved ? "Show active" : "Show resolved"}
        </button>
      </div>

      {loading ? (
        <div className="text-gray-500 animate-pulse">Loading alerts…</div>
      ) : alerts.length === 0 ? (
        <div className="card text-center text-gray-500 py-16">
          <div className="text-4xl mb-3">🎉</div>
          <div>{showResolved ? "No resolved alerts." : "No active alerts — team is healthy!"}</div>
        </div>
      ) : (
        <div className="space-y-3">
          {alerts.map((alert) => (
            <div key={alert.id} className="card">
              <div className="flex items-start gap-4">
                {/* Icon */}
                <div className="text-2xl mt-0.5 shrink-0">
                  {ALERT_TYPE_ICON[alert.alert_type] || "⚠️"}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2 mb-1">
                    <span className="font-semibold text-white capitalize">{alert.alert_type} Alert</span>
                    <AlertBadge severity={alert.severity} />
                    <span className="text-xs text-gray-500">
                      {new Date(alert.created_at).toLocaleDateString()}
                    </span>
                  </div>

                  <p className="text-sm text-gray-400 mb-2">{alert.description}</p>

                  {/* Employee link */}
                  <button
                    onClick={() => navigate(`/employee/${alert.employee_id}`)}
                    className="text-xs text-cogni-400 hover:text-cogni-300 transition-colors mb-2 block"
                  >
                    {alert.employee_name} — View profile →
                  </button>

                  {/* Recommendations */}
                  {Array.isArray(alert.recommendations) && alert.recommendations.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2">
                      {alert.recommendations.map((r, i) => (
                        <span key={i} className="text-xs bg-gray-800 text-gray-300 px-2.5 py-1 rounded-lg">
                          {r}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Resolve button */}
                {!showResolved && (
                  <button
                    onClick={() => handleResolve(alert.id)}
                    disabled={resolving === alert.id}
                    className="shrink-0 text-xs text-gray-500 hover:text-green-400 border border-gray-700 hover:border-green-700 px-3 py-1.5 rounded-xl transition-colors disabled:opacity-50"
                  >
                    {resolving === alert.id ? "…" : "Resolve"}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
