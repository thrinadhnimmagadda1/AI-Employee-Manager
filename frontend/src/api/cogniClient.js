/**
 * CogniTeam API client
 * ====================
 * Axios instance with:
 *   • Access token attached to every request (Authorization: Bearer)
 *   • withCredentials: true — browser sends httpOnly refresh-token cookie
 *     automatically on /auth/* requests
 *   • Transparent 401 recovery:
 *       1. On first 401, pause the failing request
 *       2. POST /auth/refresh  (cookie sent automatically by browser)
 *       3. Update stored access token
 *       4. Drain the queue of any other requests that arrived mid-refresh
 *       5. Retry all paused requests with the new token
 *       6. If refresh also fails → emit session-expired event + redirect
 *
 * Session-expired event:
 *   window dispatches "cogniteam:session-expired"
 *   App.jsx listens and clears React auth state cleanly.
 */

import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Refresh-queue state ───────────────────────────────────────
// Only one token refresh is allowed in-flight at a time.
// All other 401ing requests queue up and are retried once the new token arrives.
let _isRefreshing  = false;
let _pendingQueue  = [];   // [{ resolve, reject }]

function _drainQueue(error, newToken = null) {
  _pendingQueue.forEach(({ resolve, reject }) =>
    error ? reject(error) : resolve(newToken)
  );
  _pendingQueue = [];
}

function _clearLocalAuth() {
  localStorage.removeItem("cogniteam_token");
  localStorage.removeItem("cogniteam_role");
}

// ── Axios instance ────────────────────────────────────────────
const client = axios.create({
  baseURL:      BASE_URL,
  timeout:      30_000,
  headers:      { "Content-Type": "application/json" },
  // Sends cookies (incl. httpOnly refresh-token) cross-origin.
  // Requires CORS allow_credentials=True + explicit origin on the server.
  withCredentials: true,
});

// ── Request interceptor: attach access token ──────────────────
client.interceptors.request.use((config) => {
  const token = localStorage.getItem("cogniteam_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ── Response interceptor: transparent token refresh ───────────
client.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config;

    // ── Only intercept 401s on non-auth, first-time failures ──
    const isAuthEndpoint = original?.url?.startsWith("/auth/");
    const alreadyRetried = original?._retried === true;

    if (error.response?.status !== 401 || isAuthEndpoint || alreadyRetried) {
      return Promise.reject(error);
    }

    // ── If a refresh is already in flight, queue this request ─
    if (_isRefreshing) {
      return new Promise((resolve, reject) => {
        _pendingQueue.push({ resolve, reject });
      }).then((newToken) => {
        original.headers.Authorization = `Bearer ${newToken}`;
        return client(original);
      });
    }

    // ── Attempt token refresh ─────────────────────────────────
    original._retried = true;
    _isRefreshing      = true;

    try {
      // The refresh-token httpOnly cookie is sent automatically by the browser
      // because withCredentials: true is set on the client AND the server
      // sets the cookie with path="/auth" (so it's sent to /auth/* only).
      const { data } = await client.post("/auth/refresh");
      const newToken = data.access_token;

      localStorage.setItem("cogniteam_token", newToken);
      if (data.role) localStorage.setItem("cogniteam_role", data.role);

      // Update default header for new requests
      client.defaults.headers.common.Authorization = `Bearer ${newToken}`;

      // Patch the original request and drain all queued requests
      original.headers.Authorization = `Bearer ${newToken}`;
      _drainQueue(null, newToken);

      return client(original);
    } catch (refreshError) {
      // Refresh failed — session is genuinely over
      _drainQueue(refreshError);
      _clearLocalAuth();

      // Signal App.jsx to clear React auth state
      window.dispatchEvent(new Event("cogniteam:session-expired"));

      // Hard redirect — no back-navigation to the protected page
      window.location.replace("/login");

      return Promise.reject(refreshError);
    } finally {
      _isRefreshing = false;
    }
  }
);

// ── Auth ──────────────────────────────────────────────────────

/**
 * Login — returns { access_token, role, expires_in }.
 * Server also sets httpOnly refresh-token cookie automatically.
 */
export const login = (email, password) =>
  client.post("/auth/login", { email, password }).then((r) => r.data);

/**
 * Manually refresh the access token.
 * Normally done automatically by the interceptor; exposed here so
 * App.jsx can attempt a silent refresh on initial page load.
 */
export const refreshToken = () =>
  client.post("/auth/refresh").then((r) => r.data);

/**
 * Logout — revokes the refresh token server-side and clears the cookie.
 * Always clears local state even if the API call fails.
 */
export const logout = () =>
  client
    .post("/auth/logout")
    .then((r) => r.data)
    .finally(_clearLocalAuth);

// ── Dashboard ─────────────────────────────────────────────────
export const getOverview = () =>
  client.get("/dashboard/overview").then((r) => r.data);

export const getTeamDashboard = (teamId) =>
  client.get(`/dashboard/team/${teamId}`).then((r) => r.data);

export const getEmployeeTrends = (employeeId, weeks = 12) =>
  client
    .get(`/dashboard/trends/${employeeId}`, { params: { weeks } })
    .then((r) => r.data);

// ── Employees ─────────────────────────────────────────────────
export const listEmployees = () =>
  client.get("/employees").then((r) => r.data);

export const getEmployee = (id) =>
  client.get(`/employees/${id}`).then((r) => r.data);

export const getEmployeeRiskFactors = (id) =>
  client.get(`/employees/${id}/risk-factors`).then((r) => r.data);

export const getEmployeeHistory = (id, weeks = 12) =>
  client.get(`/employees/${id}/history`, { params: { weeks } }).then((r) => r.data);

// ── Alerts ────────────────────────────────────────────────────
export const listAlerts = (resolved = false, severity = null) =>
  client
    .get("/alerts", { params: { resolved, severity } })
    .then((r) => r.data);

export const getAlert = (id) =>
  client.get(`/alerts/${id}`).then((r) => r.data);

export const resolveAlert = (id, note = "") =>
  client
    .patch(`/alerts/${id}/resolve`, { resolved: true, resolution_note: note })
    .then((r) => r.data);

// ── Chat ──────────────────────────────────────────────────────
export const sendChat = (question, employeeId = null, teamId = null, useRag = true) =>
  client
    .post("/chat", {
      question,
      employee_id: employeeId,
      team_id:     teamId,
      use_rag:     useRag,
    })
    .then((r) => r.data);

// ── Graph ─────────────────────────────────────────────────────
export const getGraphForWeek = (week = "current") =>
  client.get(`/graph/${week}`).then((r) => r.data);

export const listGraphWeeks = () =>
  client.get("/graph/weeks").then((r) => r.data);

// ── System ────────────────────────────────────────────────────
export const getSystemInfo = () =>
  client.get("/api/info").then((r) => r.data);

// ── Admin (HR only) ───────────────────────────────────────────
export const getAuditLog = (params = {}) =>
  client.get("/audit-log", { params }).then((r) => r.data);

export const runPipeline = (week = "current") =>
  client.post("/admin/run-pipeline", { week }).then((r) => r.data);

export const trainModels = (skipBert = true) =>
  client.post("/admin/train-models", { skip_bert: skipBert }).then((r) => r.data);

export const getJobStatus = (jobId) =>
  client.get(`/admin/jobs/${jobId}`).then((r) => r.data);

export default client;
