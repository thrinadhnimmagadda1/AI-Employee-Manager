import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const val = payload[0]?.value;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-xs">
      <div className="text-gray-400 mb-1">{label}</div>
      <div className="font-semibold text-white">
        Sentiment: {val != null ? val.toFixed(2) : "N/A"}
      </div>
      {payload[1] && (
        <div className="text-blue-400">Score: {payload[1].value}</div>
      )}
    </div>
  );
};

export default function SentimentChart({ data = [] }) {
  const chartData = [...data].reverse().map((d) => ({
    week: d.week?.slice(5) ?? d.week,
    sentiment: d.sentiment != null ? parseFloat(d.sentiment.toFixed(3)) : null,
    score: d.overall_score ?? null,
  }));

  return (
    <div className="w-full h-56">
      <ResponsiveContainer>
        <AreaChart data={chartData} margin={{ top: 4, right: 8, left: -24, bottom: 0 }}>
          <defs>
            <linearGradient id="sentGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#4f6ef7" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#4f6ef7" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
          <XAxis
            dataKey="week"
            tick={{ fill: "#6b7280", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[-1, 1]}
            tick={{ fill: "#6b7280", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          {/* Reference lines for healthy / warning / critical */}
          <ReferenceLine y={0.3} stroke="#22c55e" strokeDasharray="4 4" strokeWidth={1} />
          <ReferenceLine y={-0.2} stroke="#eab308" strokeDasharray="4 4" strokeWidth={1} />
          <ReferenceLine y={-0.5} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1} />
          <Area
            type="monotone"
            dataKey="sentiment"
            stroke="#4f6ef7"
            strokeWidth={2}
            fill="url(#sentGrad)"
            dot={false}
            connectNulls
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
