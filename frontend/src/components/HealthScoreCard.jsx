import React from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";

function scoreColor(score) {
  if (score == null) return "border-gray-700 bg-gray-800/50";
  if (score >= 70) return "border-green-700 bg-green-950/30";
  if (score >= 40) return "border-yellow-700 bg-yellow-950/30";
  return "border-red-700 bg-red-950/30";
}

function scoreText(score) {
  if (score == null) return "text-gray-400";
  if (score >= 70) return "text-green-400";
  if (score >= 40) return "text-yellow-400";
  return "text-red-400";
}

export default function HealthScoreCard({ employee }) {
  const navigate = useNavigate();
  const { id, name, department, overall_score, burnout_risk, flags = [] } = employee;

  return (
    <button
      onClick={() => navigate(`/employee/${id}`)}
      className={clsx(
        "w-full text-left rounded-2xl border p-4 transition-all hover:scale-[1.02] hover:shadow-xl cursor-pointer",
        scoreColor(overall_score)
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="font-semibold text-white text-sm truncate">{name}</div>
          <div className="text-xs text-gray-500 mt-0.5 truncate">{department || "—"}</div>
        </div>
        <div className={clsx("text-2xl font-bold tabular-nums shrink-0", scoreText(overall_score))}>
          {overall_score ?? "—"}
        </div>
      </div>

      {/* Risk bar */}
      {burnout_risk != null && (
        <div className="mt-3">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Burnout risk</span>
            <span>{(burnout_risk * 100).toFixed(0)}%</span>
          </div>
          <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={clsx(
                "h-full rounded-full transition-all",
                burnout_risk >= 0.7 ? "bg-red-500" : burnout_risk >= 0.4 ? "bg-yellow-500" : "bg-green-500"
              )}
              style={{ width: `${(burnout_risk * 100).toFixed(0)}%` }}
            />
          </div>
        </div>
      )}

      {/* Top flag */}
      {flags.length > 0 && (
        <div className="mt-2 text-xs text-orange-400 truncate">⚠ {flags[0]}</div>
      )}
    </button>
  );
}
