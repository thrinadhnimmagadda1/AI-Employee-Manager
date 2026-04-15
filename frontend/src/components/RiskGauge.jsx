import React from "react";
import clsx from "clsx";

function arc(cx, cy, r, startAngle, endAngle) {
  const toRad = (d) => (d * Math.PI) / 180;
  const x1 = cx + r * Math.cos(toRad(startAngle));
  const y1 = cy + r * Math.sin(toRad(startAngle));
  const x2 = cx + r * Math.cos(toRad(endAngle));
  const y2 = cy + r * Math.sin(toRad(endAngle));
  const largeArc = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
}

export default function RiskGauge({ label, value = 0, size = 120 }) {
  const pct = Math.min(Math.max(value, 0), 1);
  const startAngle = 180;
  const endAngle = 180 + pct * 180;

  const color =
    pct >= 0.7 ? "#ef4444" : pct >= 0.4 ? "#eab308" : "#22c55e";

  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.38;
  const stroke = size * 0.1;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size / 2 + 16} viewBox={`0 0 ${size} ${size / 2 + 16}`}>
        {/* Track */}
        <path
          d={arc(cx, cy, r, 180, 360)}
          fill="none"
          stroke="#1f2937"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        {/* Value arc */}
        {pct > 0 && (
          <path
            d={arc(cx, cy, r, startAngle, endAngle)}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeLinecap="round"
          />
        )}
        {/* Value text */}
        <text
          x={cx}
          y={cy + 4}
          textAnchor="middle"
          fill="white"
          fontSize={size * 0.16}
          fontWeight="700"
          fontFamily="Inter"
        >
          {(pct * 100).toFixed(0)}%
        </text>
      </svg>
      <span className="text-xs text-gray-400 font-medium">{label}</span>
    </div>
  );
}
