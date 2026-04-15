import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "⬛" },
  { to: "/alerts", label: "Alerts", icon: "🔔" },
  { to: "/chat", label: "AI Chat", icon: "💬" },
];

export default function Navbar({ onLogout }) {
  const role = localStorage.getItem("cogniteam_role") || "manager";

  return (
    <aside className="fixed left-0 top-0 h-full w-64 bg-gray-900 border-r border-gray-800 flex flex-col z-50">
      {/* Brand */}
      <div className="px-6 py-5 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-cogni-500 flex items-center justify-center text-white font-bold text-lg">
            C
          </div>
          <div>
            <div className="font-bold text-white leading-tight">CogniTeam</div>
            <div className="text-xs text-gray-500 capitalize">{role} view</div>
          </div>
        </div>
      </div>

      {/* Nav links */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition-colors",
                isActive
                  ? "bg-cogni-500/20 text-cogni-400"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
              )
            }
          >
            <span className="text-base">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-gray-800">
        <button
          onClick={onLogout}
          className="w-full text-left text-sm text-gray-500 hover:text-red-400 transition-colors px-4 py-2 rounded-xl hover:bg-red-950/30"
        >
          Sign out
        </button>
      </div>
    </aside>
  );
}
