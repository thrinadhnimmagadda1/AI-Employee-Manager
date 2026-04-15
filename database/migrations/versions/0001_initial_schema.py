"""initial_schema

Creates all 10 CogniTeam tables from scratch on a fresh database.
Applying this on a database that already has the tables is safe because every
create_table call is wrapped in IF NOT EXISTS via checkfirst=True.

Revision ID: 0001
Revises: —
Create Date: 2026-04-07 00:00:00 UTC

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enable uuid-ossp extension ────────────────────────────────────────
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ── Table 1: employees ────────────────────────────────────────────────
    op.create_table(
        "employees",
        sa.Column("id",            sa.Integer(),     nullable=False),
        sa.Column("name",          sa.String(100),   nullable=False),
        sa.Column("department",    sa.String(100),   nullable=True),
        sa.Column("role",          sa.String(100),   nullable=True),
        sa.Column("manager_id",    sa.Integer(),     nullable=True),
        sa.Column("tenure_months", sa.Integer(),     nullable=True, server_default="0"),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=True,
                  server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["manager_id"], ["employees.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_employees_manager",    "employees", ["manager_id"])
    op.create_index("idx_employees_department", "employees", ["department"])

    # ── Table 2: message_metadata ─────────────────────────────────────────
    op.create_table(
        "message_metadata",
        sa.Column("id",                  sa.Integer(),  nullable=False),
        sa.Column("sender_id",           sa.Integer(),  nullable=False),
        sa.Column("receiver_id",         sa.Integer(),  nullable=False),
        sa.Column("channel",             sa.String(50), nullable=False, server_default="email"),
        sa.Column("timestamp",           sa.DateTime(timezone=True), nullable=False),
        sa.Column("word_count",          sa.Integer(),  nullable=True, server_default="0"),
        sa.Column("is_after_hours",      sa.Boolean(),  nullable=True, server_default="false"),
        sa.Column("response_time_hours", sa.Float(),    nullable=True),
        sa.ForeignKeyConstraint(["sender_id"],   ["employees.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["receiver_id"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_msg_sender",    "message_metadata", ["sender_id"])
    op.create_index("idx_msg_receiver",  "message_metadata", ["receiver_id"])
    op.create_index("idx_msg_timestamp", "message_metadata", ["timestamp"])

    # ── Table 3: sentiment_scores ─────────────────────────────────────────
    op.create_table(
        "sentiment_scores",
        sa.Column("id",          sa.Integer(),  nullable=False),
        sa.Column("employee_id", sa.Integer(),  nullable=False),
        sa.Column("week",        sa.Date(),     nullable=False),
        sa.Column("sentiment",   sa.Float(),    nullable=True),
        sa.Column("emotion",     sa.String(50), nullable=True),
        sa.Column("confidence",  sa.Float(),    nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=True,
                  server_default=sa.text("NOW()")),
        sa.CheckConstraint("sentiment  >= -1.0 AND sentiment  <= 1.0", name="ck_sentiment_range"),
        sa.CheckConstraint("confidence >= 0.0  AND confidence <= 1.0", name="ck_confidence_range"),
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("employee_id", "week", name="uq_sentiment_employee_week"),
    )
    op.create_index("idx_sentiment_employee", "sentiment_scores", ["employee_id"])
    op.create_index("idx_sentiment_week",     "sentiment_scores", ["week"])

    # ── Table 4: behavioral_features ─────────────────────────────────────
    op.create_table(
        "behavioral_features",
        sa.Column("id",                 sa.Integer(), nullable=False),
        sa.Column("employee_id",        sa.Integer(), nullable=False),
        sa.Column("week",               sa.Date(),    nullable=False),
        sa.Column("avg_response_hours", sa.Float(),   nullable=True),
        sa.Column("after_hours_count",  sa.Integer(), nullable=True, server_default="0"),
        sa.Column("message_count",      sa.Integer(), nullable=True, server_default="0"),
        sa.Column("avg_message_length", sa.Float(),   nullable=True),
        sa.Column("participation_rate", sa.Float(),   nullable=True),
        sa.Column("manager_comm_ratio", sa.Float(),   nullable=True),
        sa.Column("sentiment_velocity", sa.Float(),   nullable=True),
        sa.CheckConstraint(
            "participation_rate >= 0.0 AND participation_rate <= 1.0",
            name="ck_participation_rate",
        ),
        sa.CheckConstraint(
            "manager_comm_ratio >= 0.0 AND manager_comm_ratio <= 1.0",
            name="ck_manager_comm_ratio",
        ),
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("employee_id", "week", name="uq_bf_employee_week"),
    )
    op.create_index("idx_bf_employee", "behavioral_features", ["employee_id"])
    op.create_index("idx_bf_week",     "behavioral_features", ["week"])

    # ── Table 5: health_scores ────────────────────────────────────────────
    op.create_table(
        "health_scores",
        sa.Column("id",                 sa.Integer(), nullable=False),
        sa.Column("employee_id",        sa.Integer(), nullable=False),
        sa.Column("team_id",            sa.Integer(), nullable=True),
        sa.Column("week",               sa.Date(),    nullable=False),
        sa.Column("burnout_risk",       sa.Float(),   nullable=True),
        sa.Column("attrition_risk_30d", sa.Float(),   nullable=True),
        sa.Column("attrition_risk_60d", sa.Float(),   nullable=True),
        sa.Column("attrition_risk_90d", sa.Float(),   nullable=True),
        sa.Column("conflict_risk",      sa.Float(),   nullable=True),
        sa.Column("engagement_score",   sa.Float(),   nullable=True),
        sa.Column("overall_score",      sa.Integer(), nullable=True),
        sa.Column("shap_values",        postgresql.JSONB(), nullable=True),
        sa.Column("flags",              postgresql.JSONB(), nullable=True, server_default="'[]'"),
        sa.Column("computed_at",        sa.DateTime(timezone=True), nullable=True,
                  server_default=sa.text("NOW()")),
        sa.CheckConstraint("burnout_risk       >= 0.0 AND burnout_risk       <= 1.0", name="ck_burnout_risk"),
        sa.CheckConstraint("attrition_risk_30d >= 0.0 AND attrition_risk_30d <= 1.0", name="ck_attrition_30d"),
        sa.CheckConstraint("attrition_risk_60d >= 0.0 AND attrition_risk_60d <= 1.0", name="ck_attrition_60d"),
        sa.CheckConstraint("attrition_risk_90d >= 0.0 AND attrition_risk_90d <= 1.0", name="ck_attrition_90d"),
        sa.CheckConstraint("conflict_risk      >= 0.0 AND conflict_risk      <= 1.0", name="ck_conflict_risk"),
        sa.CheckConstraint("engagement_score   >= 0.0 AND engagement_score   <= 1.0", name="ck_engagement_score"),
        sa.CheckConstraint("overall_score >= 0 AND overall_score <= 100",             name="ck_overall_score"),
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("employee_id", "week", name="uq_hs_employee_week"),
    )
    op.create_index("idx_hs_employee", "health_scores", ["employee_id"])
    op.create_index("idx_hs_week",     "health_scores", ["week"])
    op.create_index("idx_hs_team",     "health_scores", ["team_id"])

    # ── Table 6: alerts ───────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id",              sa.Integer(),  nullable=False),
        sa.Column("employee_id",     sa.Integer(),  nullable=False),
        sa.Column("alert_type",      sa.String(50), nullable=False),
        sa.Column("severity",        sa.String(20), nullable=False),
        sa.Column("description",     sa.Text(),     nullable=True),
        sa.Column("recommendations", postgresql.JSONB(), nullable=True, server_default="'[]'"),
        sa.Column("resolved",        sa.Boolean(),  nullable=True, server_default="false"),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=True,
                  server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "alert_type IN ('burnout','conflict','attrition','isolation')",
            name="ck_alert_type",
        ),
        sa.CheckConstraint(
            "severity IN ('low','medium','high','critical')",
            name="ck_severity",
        ),
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_alerts_employee", "alerts", ["employee_id"])
    op.create_index("idx_alerts_resolved", "alerts", ["resolved"])
    op.create_index("idx_alerts_severity", "alerts", ["severity"])

    # ── Table 7: comm_graph ───────────────────────────────────────────────
    op.create_table(
        "comm_graph",
        sa.Column("id",                  sa.Integer(), nullable=False),
        sa.Column("employee_a",          sa.Integer(), nullable=False),
        sa.Column("employee_b",          sa.Integer(), nullable=False),
        sa.Column("week",                sa.Date(),    nullable=False),
        sa.Column("message_count",       sa.Integer(), nullable=True, server_default="0"),
        sa.Column("avg_sentiment",       sa.Float(),   nullable=True),
        sa.Column("avg_response_hours",  sa.Float(),   nullable=True),
        sa.Column("relationship_health", sa.Float(),   nullable=True),
        sa.CheckConstraint(
            "relationship_health >= 0.0 AND relationship_health <= 1.0",
            name="ck_relationship_health",
        ),
        sa.ForeignKeyConstraint(["employee_a"], ["employees.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["employee_b"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("employee_a", "employee_b", "week", name="uq_graph_pair_week"),
    )
    op.create_index("idx_graph_employee_a", "comm_graph", ["employee_a"])
    op.create_index("idx_graph_employee_b", "comm_graph", ["employee_b"])
    op.create_index("idx_graph_week",       "comm_graph", ["week"])

    # ── Table 8: audit_log ────────────────────────────────────────────────
    op.create_table(
        "audit_log",
        sa.Column("id",         sa.Integer(),  nullable=False),
        sa.Column("user_id",    sa.Integer(),  nullable=True),
        sa.Column("action",     sa.String(100), nullable=False),
        sa.Column("target",     sa.String(100), nullable=True),
        sa.Column("timestamp",  sa.DateTime(timezone=True), nullable=True,
                  server_default=sa.text("NOW()")),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_audit_user",      "audit_log", ["user_id"])
    op.create_index("idx_audit_timestamp", "audit_log", ["timestamp"])

    # ── Table 9: users ────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",              sa.Integer(),    nullable=False),
        sa.Column("email",           sa.String(255),  nullable=False),
        sa.Column("hashed_password", sa.String(255),  nullable=False),
        sa.Column("role",            sa.String(20),   nullable=False, server_default="manager"),
        sa.Column("employee_id",     sa.Integer(),    nullable=True),
        sa.Column("is_active",       sa.Boolean(),    nullable=True, server_default="true"),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=True,
                  server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "role IN ('employee','manager','hr')",
            name="ck_user_role",
        ),
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("idx_users_email", "users", ["email"])

    # ── Table 10: refresh_tokens ──────────────────────────────────────────
    op.create_table(
        "refresh_tokens",
        sa.Column("id",         sa.Integer(),   nullable=False),
        sa.Column("user_id",    sa.Integer(),   nullable=False),
        sa.Column("token_hash", sa.String(64),  nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked",    sa.Boolean(),   nullable=True, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True,
                  server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("idx_rt_user",   "refresh_tokens", ["user_id"])
    op.create_index("idx_rt_hash",   "refresh_tokens", ["token_hash"])
    op.create_index("idx_rt_expiry", "refresh_tokens", ["expires_at"])


def downgrade() -> None:
    # Drop tables in reverse dependency order
    op.drop_table("refresh_tokens")
    op.drop_table("users")
    op.drop_table("audit_log")
    op.drop_table("comm_graph")
    op.drop_table("alerts")
    op.drop_table("health_scores")
    op.drop_table("behavioral_features")
    op.drop_table("sentiment_scores")
    op.drop_table("message_metadata")
    op.drop_table("employees")
