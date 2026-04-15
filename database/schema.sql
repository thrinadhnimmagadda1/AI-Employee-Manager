-- ─────────────────────────────────────────────────────────
-- CogniTeam — PostgreSQL Schema
-- ─────────────────────────────────────────────────────────

-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Table 1: employees ────────────────────────────────────
CREATE TABLE IF NOT EXISTS employees (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    department      VARCHAR(100),
    role            VARCHAR(100),
    manager_id      INTEGER REFERENCES employees(id) ON DELETE SET NULL,
    tenure_months   INTEGER DEFAULT 0,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_employees_manager ON employees(manager_id);
CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department);

-- ── Table 2: message_metadata ─────────────────────────────
-- NOTE: Raw message content is NEVER stored — only metadata
CREATE TABLE IF NOT EXISTS message_metadata (
    id                  SERIAL PRIMARY KEY,
    sender_id           INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    receiver_id         INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    channel             VARCHAR(50) NOT NULL DEFAULT 'email',
    timestamp           TIMESTAMP WITH TIME ZONE NOT NULL,
    word_count          INTEGER DEFAULT 0,
    is_after_hours      BOOLEAN DEFAULT FALSE,
    response_time_hours FLOAT
);

CREATE INDEX IF NOT EXISTS idx_msg_sender ON message_metadata(sender_id);
CREATE INDEX IF NOT EXISTS idx_msg_receiver ON message_metadata(receiver_id);
CREATE INDEX IF NOT EXISTS idx_msg_timestamp ON message_metadata(timestamp);

-- ── Table 3: sentiment_scores ─────────────────────────────
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id          SERIAL PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    week        DATE NOT NULL,
    sentiment   FLOAT CHECK (sentiment >= -1.0 AND sentiment <= 1.0),
    emotion     VARCHAR(50),
    confidence  FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (employee_id, week)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_employee ON sentiment_scores(employee_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_week ON sentiment_scores(week);

-- ── Table 4: behavioral_features ─────────────────────────
CREATE TABLE IF NOT EXISTS behavioral_features (
    id                  SERIAL PRIMARY KEY,
    employee_id         INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    week                DATE NOT NULL,
    avg_response_hours  FLOAT,
    after_hours_count   INTEGER DEFAULT 0,
    message_count       INTEGER DEFAULT 0,
    avg_message_length  FLOAT,
    participation_rate  FLOAT CHECK (participation_rate >= 0.0 AND participation_rate <= 1.0),
    manager_comm_ratio  FLOAT CHECK (manager_comm_ratio >= 0.0 AND manager_comm_ratio <= 1.0),
    sentiment_velocity  FLOAT,
    UNIQUE (employee_id, week)
);

CREATE INDEX IF NOT EXISTS idx_bf_employee ON behavioral_features(employee_id);
CREATE INDEX IF NOT EXISTS idx_bf_week ON behavioral_features(week);

-- ── Table 5: health_scores ────────────────────────────────
CREATE TABLE IF NOT EXISTS health_scores (
    id                  SERIAL PRIMARY KEY,
    employee_id         INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    team_id             INTEGER,
    week                DATE NOT NULL,
    burnout_risk        FLOAT CHECK (burnout_risk >= 0.0 AND burnout_risk <= 1.0),
    attrition_risk_30d  FLOAT CHECK (attrition_risk_30d >= 0.0 AND attrition_risk_30d <= 1.0),
    attrition_risk_60d  FLOAT CHECK (attrition_risk_60d >= 0.0 AND attrition_risk_60d <= 1.0),
    attrition_risk_90d  FLOAT CHECK (attrition_risk_90d >= 0.0 AND attrition_risk_90d <= 1.0),
    conflict_risk       FLOAT CHECK (conflict_risk >= 0.0 AND conflict_risk <= 1.0),
    engagement_score    FLOAT CHECK (engagement_score >= 0.0 AND engagement_score <= 1.0),
    overall_score       INTEGER CHECK (overall_score >= 0 AND overall_score <= 100),
    shap_values         JSONB,
    flags               JSONB DEFAULT '[]',
    computed_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (employee_id, week)
);

CREATE INDEX IF NOT EXISTS idx_hs_employee ON health_scores(employee_id);
CREATE INDEX IF NOT EXISTS idx_hs_week ON health_scores(week);
CREATE INDEX IF NOT EXISTS idx_hs_team ON health_scores(team_id);

-- ── Table 6: alerts ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id              SERIAL PRIMARY KEY,
    employee_id     INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    alert_type      VARCHAR(50) NOT NULL CHECK (alert_type IN ('burnout','conflict','attrition','isolation')),
    severity        VARCHAR(20) NOT NULL CHECK (severity IN ('low','medium','high','critical')),
    description     TEXT,
    recommendations JSONB DEFAULT '[]',
    resolved        BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_employee ON alerts(employee_id);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

-- ── Table 7: comm_graph ───────────────────────────────────
CREATE TABLE IF NOT EXISTS comm_graph (
    id                  SERIAL PRIMARY KEY,
    employee_a          INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    employee_b          INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    week                DATE NOT NULL,
    message_count       INTEGER DEFAULT 0,
    avg_sentiment       FLOAT,
    avg_response_hours  FLOAT,
    relationship_health FLOAT CHECK (relationship_health >= 0.0 AND relationship_health <= 1.0),
    UNIQUE (employee_a, employee_b, week)
);

CREATE INDEX IF NOT EXISTS idx_graph_employee_a ON comm_graph(employee_a);
CREATE INDEX IF NOT EXISTS idx_graph_employee_b ON comm_graph(employee_b);
CREATE INDEX IF NOT EXISTS idx_graph_week ON comm_graph(week);

-- ── Table 8: audit_log ────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER,
    action      VARCHAR(100) NOT NULL,
    target      VARCHAR(100),
    timestamp   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address  VARCHAR(45)
);

CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);

-- ── Table 9: users (auth) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role            VARCHAR(20) NOT NULL DEFAULT 'manager' CHECK (role IN ('employee','manager','hr')),
    employee_id     INTEGER REFERENCES employees(id) ON DELETE SET NULL,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ── Table 10: refresh_tokens ───────────────────────────────
-- Server-side refresh token store for token rotation + revocation.
-- Raw tokens are NEVER stored — only their SHA-256 hash.
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  VARCHAR(64) NOT NULL UNIQUE,   -- SHA-256 hex digest of raw token
    expires_at  TIMESTAMP WITH TIME ZONE NOT NULL,
    revoked     BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rt_user   ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_rt_hash   ON refresh_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_rt_expiry ON refresh_tokens(expires_at);
