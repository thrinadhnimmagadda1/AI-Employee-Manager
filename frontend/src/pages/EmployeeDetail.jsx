import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getEmployee, getEmployeeRiskFactors } from "../api/cogniClient";
import SentimentChart from "../components/SentimentChart";
import RiskGauge from "../components/RiskGauge";
import AlertBadge from "../components/AlertBadge";

function Section({ title, children }) {
  return (
    <div className="card">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-widest mb-4">{title}</h3>
      {children}
    </div>
  );
}

function FeatureRow({ label, value, unit = "", color }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-800 last:border-0">
      <span className="text-sm text-gray-400">{label}</span>
      <span className={`text-sm font-semibold ${color ?? "text-white"}`}>
        {value != null ? `${value}${unit}` : "—"}
      </span>
    </div>
  );
}

export default function EmployeeDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const role = localStorage.getItem("cogniteam_role") || "manager";

  const [data, setData] = useState(null);
  const [riskFactors, setRiskFactors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([getEmployee(id), getEmployeeRiskFactors(id)])
      .then(([empData, rfData]) => {
        setData(empData);
        setRiskFactors(rfData.risk_factors || []);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed to load employee data"))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="p-8 text-gray-500 animate-pulse">Loading employee profile…</div>;
  if (error) return <div className="p-8 text-red-400">{error}</div>;

  const { employee, health_scores: hs, behavioral_features: bf, active_alerts, sentiment_trend } = data || {};

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <button onClick={() => navigate(-1)} className="btn-ghost text-sm">← Back</button>
        <div>
          <h1 className="text-2xl font-bold text-white">{employee?.name}</h1>
          <p className="text-gray-500 text-sm mt-1">
            {employee?.role} · {employee?.department} · {employee?.tenure_months}mo tenure
          </p>
        </div>
        <div className="ml-auto">
          <div
            className={`text-4xl font-bold ${
              (hs?.overall_score ?? 50) >= 70
                ? "text-green-400"
                : (hs?.overall_score ?? 50) >= 40
                ? "text-yellow-400"
                : "text-red-400"
            }`}
          >
            {hs?.overall_score ?? "—"}
            <span className="text-lg text-gray-500">/100</span>
          </div>
          <div className="text-xs text-gray-500 text-right">Overall Score</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column */}
        <div className="space-y-5">
          {/* Risk Gauges */}
          <Section title="Risk Scores">
            <div className="grid grid-cols-2 gap-4">
              <RiskGauge label="Burnout" value={hs?.burnout_risk ?? 0} />
              <RiskGauge label="Attrition 90d" value={hs?.attrition_risk_90d ?? 0} />
              <RiskGauge label="Conflict" value={hs?.conflict_risk ?? 0} />
              <RiskGauge label="Engagement" value={hs?.engagement_score ?? 0} />
            </div>
          </Section>

          {/* Active Alerts */}
          <Section title="Active Alerts">
            {!active_alerts?.length ? (
              <p className="text-sm text-gray-500">No active alerts</p>
            ) : (
              <div className="space-y-3">
                {active_alerts.map((a) => (
                  <div key={a.id} className="p-3 bg-gray-800 rounded-xl">
                    <div className="flex items-center gap-2 mb-1">
                      <AlertBadge severity={a.severity} />
                      <span className="text-sm font-medium text-white capitalize">{a.type}</span>
                    </div>
                    <p className="text-xs text-gray-400">{a.description}</p>
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* Behavioral Features (managers: visible; employees: hidden) */}
          {bf && (
            <Section title="Behavioral Signals">
              <FeatureRow
                label="Avg response time"
                value={bf.avg_response_hours?.toFixed(1)}
                unit="h"
                color={bf.avg_response_hours > 12 ? "text-yellow-400" : "text-white"}
              />
              <FeatureRow
                label="After-hours messages"
                value={bf.after_hours_count}
                color={bf.after_hours_count > 10 ? "text-red-400" : "text-white"}
              />
              <FeatureRow
                label="Messages this week"
                value={bf.message_count}
              />
              <FeatureRow
                label="Participation rate"
                value={bf.participation_rate != null ? (bf.participation_rate * 100).toFixed(0) : null}
                unit="%"
                color={bf.participation_rate < 0.4 ? "text-red-400" : "text-white"}
              />
              <FeatureRow
                label="Sentiment velocity"
                value={bf.sentiment_velocity?.toFixed(3)}
                color={bf.sentiment_velocity < -0.15 ? "text-red-400" : "text-white"}
              />
            </Section>
          )}
        </div>

        {/* Right: charts and SHAP */}
        <div className="lg:col-span-2 space-y-5">
          {/* Sentiment trend */}
          <Section title="12-Week Sentiment & Health Trend">
            <SentimentChart data={sentiment_trend || []} />
          </Section>

          {/* Attrition timeline */}
          <Section title="Attrition Risk Timeline">
            <div className="flex gap-4">
              {[
                { label: "30 days", val: hs?.attrition_risk_30d },
                { label: "60 days", val: hs?.attrition_risk_60d },
                { label: "90 days", val: hs?.attrition_risk_90d },
              ].map(({ label, val }) => (
                <div key={label} className="flex-1 bg-gray-800 rounded-xl p-4 text-center">
                  <div
                    className={`text-2xl font-bold ${
                      (val ?? 0) >= 0.5 ? "text-red-400" : (val ?? 0) >= 0.3 ? "text-yellow-400" : "text-green-400"
                    }`}
                  >
                    {val != null ? `${(val * 100).toFixed(0)}%` : "—"}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">{label}</div>
                </div>
              ))}
            </div>
          </Section>

          {/* SHAP / Risk Factors — HR sees full detail, manager sees flags */}
          <Section title={role === "hr" ? "SHAP Explainability (Top Risk Factors)" : "Risk Signals"}>
            {!riskFactors?.length ? (
              <p className="text-sm text-gray-500">No risk factor data available.</p>
            ) : (
              <div className="space-y-2">
                {riskFactors.map((f, i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-gray-800 rounded-xl">
                    <span className="text-sm text-gray-300">{f.feature || f.signal}</span>
                    {f.impact != null && (
                      <span className={`text-sm font-semibold ${f.direction === "increases_risk" ? "text-red-400" : "text-green-400"}`}>
                        {f.impact > 0 ? "+" : ""}{f.impact.toFixed(3)}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Section>
        </div>
      </div>
    </div>
  );
}
