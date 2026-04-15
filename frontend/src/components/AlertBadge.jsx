import React from "react";
import clsx from "clsx";

const CONFIG = {
  critical: { cls: "badge-critical", label: "CRITICAL" },
  high:     { cls: "badge-high",     label: "HIGH"     },
  medium:   { cls: "badge-medium",   label: "MEDIUM"   },
  low:      { cls: "badge-low",      label: "LOW"      },
};

export default function AlertBadge({ severity = "low" }) {
  const { cls, label } = CONFIG[severity] ?? CONFIG.low;
  return <span className={cls}>{label}</span>;
}
