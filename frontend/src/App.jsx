import React, { useState, useEffect, useCallback } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import TeamHealth from "./pages/TeamHealth";
import EmployeeDetail from "./pages/EmployeeDetail";
import Alerts from "./pages/Alerts";
import Chat from "./pages/Chat";
import Login from "./pages/Login";
import { logout as apiLogout, refreshToken } from "./api/cogniClient";

// ── PrivateRoute ─────────────────────────────────────────────
// Checks localStorage so it works correctly after a silent token
// refresh (which updates localStorage without re-rendering App).
function PrivateRoute({ children }) {
  const token = localStorage.getItem("cogniteam_token");
  return token ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const [token,        setToken]        = useState(localStorage.getItem("cogniteam_token"));
  const [initializing, setInitializing] = useState(true);

  // ── Clear React auth state ──────────────────────────────────
  const clearSession = useCallback(() => {
    localStorage.removeItem("cogniteam_token");
    localStorage.removeItem("cogniteam_role");
    setToken(null);
  }, []);

  // ── Login callback (called by Login.jsx) ──────────────────
  const handleLogin = useCallback((newToken, role) => {
    localStorage.setItem("cogniteam_token", newToken);
    localStorage.setItem("cogniteam_role",  role);
    setToken(newToken);
  }, []);

  // ── Logout: revoke refresh token server-side then clear state
  const handleLogout = useCallback(async () => {
    try {
      await apiLogout();          // POST /auth/logout — revokes cookie + DB record
    } catch {
      // Network error or already-expired token: clear local state regardless
    } finally {
      clearSession();
    }
  }, [clearSession]);

  // ── Silent refresh on app load ────────────────────────────
  // If the user reopens the tab and the stored access token is expired
  // (or absent), attempt a silent refresh via the httpOnly cookie.
  // This means users never see the login page just because they were
  // away for 30+ minutes — as long as their 7-day refresh token is valid.
  useEffect(() => {
    const storedToken = localStorage.getItem("cogniteam_token");

    if (storedToken) {
      // Access token present: no need to refresh immediately.
      // The axios interceptor will handle expiry transparently.
      setInitializing(false);
      return;
    }

    // No access token — attempt a silent refresh from the httpOnly cookie
    refreshToken()
      .then(({ access_token, role }) => {
        localStorage.setItem("cogniteam_token", access_token);
        if (role) localStorage.setItem("cogniteam_role", role);
        setToken(access_token);
      })
      .catch(() => {
        // Cookie absent / expired — user must log in
      })
      .finally(() => {
        setInitializing(false);
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Listen for interceptor-forced logout ─────────────────
  // The axios response interceptor dispatches this event when a
  // /auth/refresh call fails, so React re-renders the Navbar away.
  useEffect(() => {
    const onExpired = () => clearSession();
    window.addEventListener("cogniteam:session-expired", onExpired);
    return () => window.removeEventListener("cogniteam:session-expired", onExpired);
  }, [clearSession]);

  // ── Splash screen while we check the refresh cookie ──────
  if (initializing) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-gray-500 text-sm animate-pulse">Loading CogniTeam…</div>
      </div>
    );
  }

  return (
    <BrowserRouter>
      {token && <Navbar onLogout={handleLogout} />}
      <div className={token ? "ml-64 min-h-screen" : "min-h-screen"}>
        <Routes>
          <Route path="/login" element={<Login onLogin={handleLogin} />} />
          <Route
            path="/"
            element={
              <PrivateRoute>
                <Dashboard />
              </PrivateRoute>
            }
          />
          <Route
            path="/team/:teamId"
            element={
              <PrivateRoute>
                <TeamHealth />
              </PrivateRoute>
            }
          />
          <Route
            path="/employee/:id"
            element={
              <PrivateRoute>
                <EmployeeDetail />
              </PrivateRoute>
            }
          />
          <Route
            path="/alerts"
            element={
              <PrivateRoute>
                <Alerts />
              </PrivateRoute>
            }
          />
          <Route
            path="/chat"
            element={
              <PrivateRoute>
                <Chat />
              </PrivateRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
