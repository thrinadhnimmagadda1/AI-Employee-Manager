import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api/cogniClient";

const DEMO_CREDENTIALS = [
  { role: "HR Admin", email: "hr@cogniteam.ai", password: "hr123" },
  { role: "Manager", email: "manager@cogniteam.ai", password: "manager123" },
  { role: "Employee", email: "employee@cogniteam.ai", password: "employee123" },
];

export default function Login({ onLogin }) {
  const [email, setEmail] = useState("manager@cogniteam.ai");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const data = await login(email, password);
      onLogin(data.access_token, data.role);
      navigate("/");
    } catch (err) {
      setError(err.response?.data?.detail || "Invalid credentials");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950">
      <div className="w-full max-w-sm">
        {/* Brand */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-cogni-500 text-white font-bold text-2xl mb-4">
            C
          </div>
          <h1 className="text-2xl font-bold text-white">CogniTeam</h1>
          <p className="text-gray-500 mt-1 text-sm">AI Organizational Intelligence</p>
        </div>

        <form onSubmit={handleSubmit} className="card space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1.5">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-white text-sm focus:outline-none focus:border-cogni-500 transition-colors"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1.5">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-white text-sm focus:outline-none focus:border-cogni-500 transition-colors"
              required
            />
          </div>

          {error && (
            <div className="text-red-400 text-sm bg-red-950/30 border border-red-800 rounded-xl px-4 py-2.5">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>

        {/* Demo credentials box */}
        <div className="mt-5 rounded-xl border border-blue-800/60 bg-blue-950/40 px-4 py-4">
          <p className="text-xs font-semibold text-blue-300 uppercase tracking-wide mb-3">
            Demo Credentials
          </p>
          <div className="space-y-2">
            {DEMO_CREDENTIALS.map(({ role, email: demoEmail, password: demoPass }) => (
              <button
                key={demoEmail}
                type="button"
                onClick={() => {
                  setEmail(demoEmail);
                  setPassword(demoPass);
                  setError("");
                }}
                className="w-full text-left flex items-center justify-between rounded-lg bg-blue-900/30 hover:bg-blue-900/60 border border-blue-800/40 px-3 py-2 transition-colors group"
              >
                <span className="text-xs font-medium text-blue-200 w-24">{role}</span>
                <span className="text-xs text-gray-400 flex-1 truncate">{demoEmail}</span>
                <span className="text-xs font-mono text-blue-300 ml-2 opacity-70 group-hover:opacity-100">
                  {demoPass}
                </span>
              </button>
            ))}
          </div>
          <p className="text-xs text-blue-400/60 mt-3 text-center">
            Click a row to auto-fill credentials
          </p>
        </div>

        <p className="text-center text-xs text-gray-600 mt-5">
          CogniTeam — All communication data is anonymized and privacy-protected.
        </p>
      </div>
    </div>
  );
}
