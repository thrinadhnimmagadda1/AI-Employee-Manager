"""
SQLAlchemy ORM models for CogniTeam.

These models mirror database/schema.sql exactly and are used by:
  - Alembic autogenerate (future schema diffs)
  - Type-safe queries where raw SQL isn't needed

Raw SQL via sqlalchemy.text() is still used in the existing API routes;
these models exist alongside that pattern without breaking anything.
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ── Table 1: employees ──────────────────────────────────────────────────────

class Employee(Base):
    __tablename__ = "employees"
    __table_args__ = (
        Index("idx_employees_manager", "manager_id"),
        Index("idx_employees_department", "department"),
    )

    id             = Column(Integer, primary_key=True)
    name           = Column(String(100), nullable=False)
    department     = Column(String(100))
    role           = Column(String(100))
    manager_id     = Column(Integer, ForeignKey("employees.id", ondelete="SET NULL"))
    tenure_months  = Column(Integer, default=0)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())


# ── Table 2: message_metadata ───────────────────────────────────────────────

class MessageMetadata(Base):
    __tablename__ = "message_metadata"
    __table_args__ = (
        Index("idx_msg_sender",    "sender_id"),
        Index("idx_msg_receiver",  "receiver_id"),
        Index("idx_msg_timestamp", "timestamp"),
    )

    id                  = Column(Integer, primary_key=True)
    sender_id           = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    receiver_id         = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    channel             = Column(String(50), nullable=False, default="email")
    timestamp           = Column(DateTime(timezone=True), nullable=False)
    word_count          = Column(Integer, default=0)
    is_after_hours      = Column(Boolean, default=False)
    response_time_hours = Column(Float)


# ── Table 3: sentiment_scores ───────────────────────────────────────────────

class SentimentScore(Base):
    __tablename__ = "sentiment_scores"
    __table_args__ = (
        UniqueConstraint("employee_id", "week", name="uq_sentiment_employee_week"),
        Index("idx_sentiment_employee", "employee_id"),
        Index("idx_sentiment_week",     "week"),
        CheckConstraint("sentiment  >= -1.0 AND sentiment  <= 1.0", name="ck_sentiment_range"),
        CheckConstraint("confidence >= 0.0  AND confidence <= 1.0", name="ck_confidence_range"),
    )

    id          = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    week        = Column(Date, nullable=False)
    sentiment   = Column(Float)
    emotion     = Column(String(50))
    confidence  = Column(Float)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())


# ── Table 4: behavioral_features ────────────────────────────────────────────

class BehavioralFeature(Base):
    __tablename__ = "behavioral_features"
    __table_args__ = (
        UniqueConstraint("employee_id", "week", name="uq_bf_employee_week"),
        Index("idx_bf_employee", "employee_id"),
        Index("idx_bf_week",     "week"),
        CheckConstraint(
            "participation_rate >= 0.0 AND participation_rate <= 1.0",
            name="ck_participation_rate",
        ),
        CheckConstraint(
            "manager_comm_ratio >= 0.0 AND manager_comm_ratio <= 1.0",
            name="ck_manager_comm_ratio",
        ),
    )

    id                 = Column(Integer, primary_key=True)
    employee_id        = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    week               = Column(Date, nullable=False)
    avg_response_hours = Column(Float)
    after_hours_count  = Column(Integer, default=0)
    message_count      = Column(Integer, default=0)
    avg_message_length = Column(Float)
    participation_rate = Column(Float)
    manager_comm_ratio = Column(Float)
    sentiment_velocity = Column(Float)


# ── Table 5: health_scores ───────────────────────────────────────────────────

class HealthScore(Base):
    __tablename__ = "health_scores"
    __table_args__ = (
        UniqueConstraint("employee_id", "week", name="uq_hs_employee_week"),
        Index("idx_hs_employee", "employee_id"),
        Index("idx_hs_week",     "week"),
        Index("idx_hs_team",     "team_id"),
        CheckConstraint("burnout_risk       >= 0.0 AND burnout_risk       <= 1.0", name="ck_burnout_risk"),
        CheckConstraint("attrition_risk_30d >= 0.0 AND attrition_risk_30d <= 1.0", name="ck_attrition_30d"),
        CheckConstraint("attrition_risk_60d >= 0.0 AND attrition_risk_60d <= 1.0", name="ck_attrition_60d"),
        CheckConstraint("attrition_risk_90d >= 0.0 AND attrition_risk_90d <= 1.0", name="ck_attrition_90d"),
        CheckConstraint("conflict_risk      >= 0.0 AND conflict_risk      <= 1.0", name="ck_conflict_risk"),
        CheckConstraint("engagement_score   >= 0.0 AND engagement_score   <= 1.0", name="ck_engagement_score"),
        CheckConstraint("overall_score >= 0 AND overall_score <= 100",             name="ck_overall_score"),
    )

    id                 = Column(Integer, primary_key=True)
    employee_id        = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    team_id            = Column(Integer)
    week               = Column(Date, nullable=False)
    burnout_risk       = Column(Float)
    attrition_risk_30d = Column(Float)
    attrition_risk_60d = Column(Float)
    attrition_risk_90d = Column(Float)
    conflict_risk      = Column(Float)
    engagement_score   = Column(Float)
    overall_score      = Column(Integer)
    shap_values        = Column(JSONB)
    flags              = Column(JSONB, server_default="'[]'")
    computed_at        = Column(DateTime(timezone=True), server_default=func.now())


# ── Table 6: alerts ──────────────────────────────────────────────────────────

class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        Index("idx_alerts_employee", "employee_id"),
        Index("idx_alerts_resolved", "resolved"),
        Index("idx_alerts_severity", "severity"),
        CheckConstraint(
            "alert_type IN ('burnout','conflict','attrition','isolation')",
            name="ck_alert_type",
        ),
        CheckConstraint(
            "severity IN ('low','medium','high','critical')",
            name="ck_severity",
        ),
    )

    id              = Column(Integer, primary_key=True)
    employee_id     = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    alert_type      = Column(String(50), nullable=False)
    severity        = Column(String(20), nullable=False)
    description     = Column(Text)
    recommendations = Column(JSONB, server_default="'[]'")
    resolved        = Column(Boolean, default=False)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


# ── Table 7: comm_graph ──────────────────────────────────────────────────────

class CommGraph(Base):
    __tablename__ = "comm_graph"
    __table_args__ = (
        UniqueConstraint("employee_a", "employee_b", "week", name="uq_graph_pair_week"),
        Index("idx_graph_employee_a", "employee_a"),
        Index("idx_graph_employee_b", "employee_b"),
        Index("idx_graph_week",       "week"),
        CheckConstraint(
            "relationship_health >= 0.0 AND relationship_health <= 1.0",
            name="ck_relationship_health",
        ),
    )

    id                  = Column(Integer, primary_key=True)
    employee_a          = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    employee_b          = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    week                = Column(Date, nullable=False)
    message_count       = Column(Integer, default=0)
    avg_sentiment       = Column(Float)
    avg_response_hours  = Column(Float)
    relationship_health = Column(Float)


# ── Table 8: audit_log ───────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("idx_audit_user",      "user_id"),
        Index("idx_audit_timestamp", "timestamp"),
    )

    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer)
    action     = Column(String(100), nullable=False)
    target     = Column(String(100))
    timestamp  = Column(DateTime(timezone=True), server_default=func.now())
    ip_address = Column(String(45))


# ── Table 9: users ───────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("idx_users_email", "email"),
        CheckConstraint(
            "role IN ('employee','manager','hr')",
            name="ck_user_role",
        ),
    )

    id              = Column(Integer, primary_key=True)
    email           = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role            = Column(String(20), nullable=False, default="manager")
    employee_id     = Column(Integer, ForeignKey("employees.id", ondelete="SET NULL"))
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


# ── Table 10: refresh_tokens ─────────────────────────────────────────────────

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    __table_args__ = (
        Index("idx_rt_user",   "user_id"),
        Index("idx_rt_hash",   "token_hash"),
        Index("idx_rt_expiry", "expires_at"),
    )

    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(64), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked    = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
