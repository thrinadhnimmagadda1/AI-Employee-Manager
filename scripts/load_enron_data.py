#!/usr/bin/env python3
"""
CogniTeam — Enron Data Loader & Full Pipeline
================================================
The most important script in the project.
Running this once transforms CogniTeam from a demo with seeded data
into a platform powered by real Enron workplace communication patterns.

What it does:
  1. Checks / cleans the raw Enron CSV
  2. Inserts ~517K email records into message_metadata
  3. Runs 12 weeks of feature engineering   → behavioral_features
  4. Builds 12 weekly communication graphs  → comm_graph
  5. Runs network analysis for each week
  6. Runs ML models on every employee       → health_scores
  7. Generates alerts for high-risk employees → alerts

Usage:
    python scripts/load_enron_data.py
    python scripts/load_enron_data.py --limit 50000      # quick test run
    python scripts/load_enron_data.py --weeks 4          # last 4 weeks only
    python scripts/load_enron_data.py --skip-cleaning    # skip cleaning step
    python scripts/load_enron_data.py --skip-loading     # skip email load (data already in DB)
    python scripts/load_enron_data.py --pipeline-only    # only re-run the analytics pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── repo root ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_RAW       = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
MODELS_DIR     = REPO_ROOT / "models"

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam"
)

# ── risk thresholds for alert generation ────────────────────────────────────
BURNOUT_CRITICAL    = 0.85
BURNOUT_HIGH        = 0.70
ATTRITION_HIGH      = 0.65
ATTRITION_MEDIUM    = 0.45
CONFLICT_HIGH       = 0.60

# ── pretty helpers ────────────────────────────────────────────────────────────

_WIDE = "═" * 60
_THIN = "─" * 60


def _banner(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}", flush=True)


def _ok(msg: str) -> None:
    print(f"  ✅  {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"  ℹ️   {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"  ⚠️   {msg}", flush=True)


def _err(msg: str) -> None:
    print(f"  ❌  {msg}", flush=True)


def _progress(current: int, total: int, label: str = "") -> None:
    pct = current / max(total, 1) * 100
    bar_len = 30
    filled = int(bar_len * current / max(total, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
    suffix = f"  {label}" if label else ""
    print(f"\r  [{bar}] {pct:5.1f}%  {current:,}/{total:,}{suffix}",
          end="", flush=True)


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_engine():
    from sqlalchemy import create_engine
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def _scalar(sql: str, params: dict | None = None) -> int:
    from sqlalchemy import text
    from sqlalchemy.orm import Session
    with Session(_get_engine()) as s:
        return s.execute(text(sql), params or {}).scalar() or 0


def _rows(sql: str, params: dict | None = None) -> list:
    from sqlalchemy import text
    from sqlalchemy.orm import Session
    with Session(_get_engine()) as s:
        return s.execute(text(sql), params or {}).fetchall()


# ── Step 1: pre-flight check ─────────────────────────────────────────────────

def check_enron_csv() -> Path:
    """Verify the Enron CSV exists. Exit with instructions if not."""
    path = DATA_RAW / "enron_emails.csv"
    if path.exists():
        rows = sum(1 for _ in open(path)) - 1
        _ok(f"data/raw/enron_emails.csv found  ({rows:,} rows)")
        return path

    _err("data/raw/enron_emails.csv NOT FOUND")
    print()
    print(f"  {_WIDE}")
    print("  MANUAL DOWNLOAD REQUIRED FOR ENRON DATA")
    print(f"  {_WIDE}")
    print("  1. Go to: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset")
    print("  2. Sign in with a free Kaggle account and click Download")
    print("  3. Unzip the downloaded archive")
    print("  4. Rename the CSV to:  enron_emails.csv")
    print(f"  5. Place it at:        {path}")
    print("  File size: approximately 1.7 GB, ~517,000 rows")
    print()
    print("  Then re-run:")
    print("    python scripts/load_enron_data.py")
    print(f"  {_WIDE}")
    sys.exit(1)


# ── Step 2: data cleaning ─────────────────────────────────────────────────────

def clean_enron_emails(raw_path: Path, force: bool = False) -> Path:
    """
    Run data_cleaner.clean_emails() to produce emails_clean.csv.
    Skips if cleaned file already exists (unless force=True).
    """
    _banner("Step 2 / 7 — Data Cleaning")
    output_path = DATA_PROCESSED / "emails_clean.csv"

    if output_path.exists() and not force:
        rows = sum(1 for _ in open(output_path)) - 1
        _ok(f"data/processed/emails_clean.csv already exists  ({rows:,} rows) — skipping.")
        _info("Use --force-clean to re-run cleaning.")
        return output_path

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"  Cleaning {raw_path.name}…  (deduplication, PII removal, signature stripping)")
    t0 = time.time()

    from src.ingestion.data_cleaner import clean_emails  # noqa: PLC0415
    clean_emails(input_path=raw_path, output_path=output_path)

    elapsed = time.time() - t0
    rows = sum(1 for _ in open(output_path)) - 1
    _ok(f"Cleaned file written: {rows:,} rows  ({_elapsed(elapsed)})")
    return output_path


# ── Step 3: email loading with live progress ──────────────────────────────────

def _count_enron_rows(csv_path: Path) -> int:
    """Count total rows in CSV quickly."""
    try:
        with open(csv_path, "rb") as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return 0


def load_enron_emails(
    csv_path: Path,
    limit: Optional[int] = None,
) -> tuple[int, int]:
    """
    Load email metadata into message_metadata + employees tables.
    Prints progress every 10,000 emails.

    Returns:
        (emails_inserted, employees_created)
    """
    _banner("Step 3 / 7 — Loading Enron Emails → Database")

    total_rows = min(limit, _count_enron_rows(csv_path)) if limit else _count_enron_rows(csv_path)
    _info(f"Target: {total_rows:,} rows from {csv_path.name}")
    _info("Progress is logged every 10,000 emails.")
    print()

    # Check how many are already loaded
    already = _scalar("SELECT COUNT(*) FROM message_metadata")
    if already > 0:
        _warn(f"{already:,} rows already in message_metadata.")
        _info("Will add new records only (ON CONFLICT DO NOTHING).")
    print()

    t0 = time.time()

    # Patch load_enron_emails to intercept its internal progress logging
    # and print our own formatted lines
    from loguru import logger as loguru_logger  # noqa: PLC0415
    original_info = loguru_logger.info

    last_printed: list[int] = [0]

    def _patched_info(msg: str, *args, **kwargs):
        if "Processed" in str(msg) and "rows" in str(msg).lower():
            # Extract the processed count from the log message
            import re
            m = re.search(r"Processed ([\d,]+) rows", str(msg))
            if m:
                n = int(m.group(1).replace(",", ""))
                if n - last_printed[0] >= 10_000:
                    pct = n / max(total_rows, 1) * 100
                    elapsed_so_far = time.time() - t0
                    rate = n / max(elapsed_so_far, 1)
                    eta = (total_rows - n) / max(rate, 1)
                    print(
                        f"  Loaded {n:>9,} / {total_rows:,} emails"
                        f"  ({pct:4.1f}%)  ETA: {_elapsed(eta)}",
                        flush=True,
                    )
                    last_printed[0] = n
        original_info(msg, *args, **kwargs)

    try:
        loguru_logger.info = _patched_info  # type: ignore[method-assign]

        from src.ingestion.enron_loader import load_enron_emails as _load  # noqa: PLC0415
        _load(csv_path=csv_path, limit=limit)

    finally:
        loguru_logger.info = original_info  # type: ignore[method-assign]

    elapsed = time.time() - t0
    after = _scalar("SELECT COUNT(*) FROM message_metadata")
    emp_count = _scalar("SELECT COUNT(*) FROM employees")
    inserted = after - already

    print()
    _ok(
        f"Email load complete in {_elapsed(elapsed)}\n"
        f"       message_metadata rows: {after:,}  ({inserted:,} new)\n"
        f"       employees table:       {emp_count:,} unique employees"
    )
    return inserted, emp_count


# ── Step 4a: feature engineering ─────────────────────────────────────────────

def run_feature_engineering(weeks: list[date]) -> int:
    """Compute behavioral features for every active employee in each week."""
    from src.ml.feature_engineering import compute_features_for_week  # noqa: PLC0415

    total = 0
    for i, week in enumerate(weeks, 1):
        t0 = time.time()
        n = compute_features_for_week(week)
        elapsed = time.time() - t0
        print(
            f"  Week {i:>2}/{len(weeks)}  {week}  →  {n} employees  ({_elapsed(elapsed)})",
            flush=True,
        )
        total += n

    return total


# ── Step 4b: graph building ───────────────────────────────────────────────────

def run_graph_building(weeks: list[date]) -> dict[date, int]:
    """Build and persist communication graph for each week."""
    from src.graph.graph_builder import build_and_persist_weekly_graph  # noqa: PLC0415

    edges_by_week: dict[date, int] = {}
    for i, week in enumerate(weeks, 1):
        t0 = time.time()
        try:
            G = build_and_persist_weekly_graph(week)
            n_edges = G.number_of_edges()
            elapsed = time.time() - t0
            print(
                f"  Week {i:>2}/{len(weeks)}  {week}  →  "
                f"{G.number_of_nodes()} nodes, {n_edges} edges  ({_elapsed(elapsed)})",
                flush=True,
            )
            edges_by_week[week] = n_edges
        except Exception as exc:
            _warn(f"Graph build failed for week {week}: {exc}")
            edges_by_week[week] = 0

    return edges_by_week


# ── Step 4c: network analysis ─────────────────────────────────────────────────

def run_network_analysis(weeks: list[date]) -> dict[date, int]:
    """Run centrality + isolation analysis for each week."""
    from src.graph.network_analyzer import analyze_network_for_week  # noqa: PLC0415

    analyzed: dict[date, int] = {}
    for i, week in enumerate(weeks, 1):
        t0 = time.time()
        try:
            metrics = analyze_network_for_week(week)
            elapsed = time.time() - t0
            isolated = sum(1 for m in metrics.values() if m.get("is_isolated"))
            print(
                f"  Week {i:>2}/{len(weeks)}  {week}  →  "
                f"{len(metrics)} employees  ({isolated} isolated)  ({_elapsed(elapsed)})",
                flush=True,
            )
            analyzed[week] = len(metrics)
        except Exception as exc:
            _warn(f"Network analysis failed for week {week}: {exc}")
            analyzed[week] = 0

    return analyzed


# ── Step 4d: ML predictions → health_scores ───────────────────────────────────

def _map_behavioral_to_hr_features(bf: dict, tenure_months: int) -> dict:
    """
    Map communication behavioral features → HR Analytics feature space.

    The ML models (burnout, attrition) were trained on HR Analytics columns.
    Enron data gives us communication metadata, so we derive proxy values.

    Mapping rationale:
      satisfaction_level   ← participation × 0.6 + sentiment_vel direction × 0.3 + overwork penalty
      last_evaluation      ← participation × 0.7 + sentiment positive component × 0.3
      average_montly_hours ← 160 base + after_hours × 3 + message_count × 0.4
      number_project       ← bucketed message_count proxy
      time_spend_company   ← tenure_months / 12 (capped)
      work_accident        ← 0 (not in Enron data)
      promotion_last_5years← 0 (not in Enron data)
    """
    ah     = float(bf.get("after_hours_count") or 0)
    prate  = float(bf.get("participation_rate") or 1.0)
    svel   = float(bf.get("sentiment_velocity") or 0)
    msgs   = float(bf.get("message_count") or 20)
    resp_h = float(bf.get("avg_response_hours") or 2)

    # satisfaction: high participation + improving sentiment → high satisfaction
    overwork_penalty = min(0.3, ah / 30)
    satisfaction = max(0.05, min(0.95,
        prate * 0.6
        + max(0.0, svel) * 0.3
        - overwork_penalty
    ))

    # last_evaluation proxy: how engaged/responsive the person is
    response_penalty = min(0.2, max(0, resp_h - 4) / 48)
    last_eval = max(0.2, min(1.0, prate * 0.7 + max(0, svel * 2) * 0.15 - response_penalty))

    # monthly hours proxy
    monthly_hours = int(min(310, 160 + ah * 3.5 + msgs * 0.4))

    # number of projects proxy (bucketed by message volume)
    if msgs <= 15:
        n_proj = 2
    elif msgs <= 30:
        n_proj = 3
    elif msgs <= 50:
        n_proj = 4
    elif msgs <= 80:
        n_proj = 5
    else:
        n_proj = 6

    return {
        "satisfaction_level":    round(satisfaction, 3),
        "last_evaluation":       round(last_eval, 3),
        "number_project":        n_proj,
        "average_montly_hours":  monthly_hours,
        "time_spend_company":    max(1, min(10, tenure_months // 12)),
        "work_accident":         0,
        "promotion_last_5years": 0,
        "salary":                1,  # encoded as 'medium'
    }


def _compute_overall_score(
    burnout: float,
    attrition_60d: float,
    conflict: float,
    engagement: float,
) -> int:
    """Combine all risk signals into a 0-100 health score (100 = healthiest)."""
    risk_component = (
        burnout       * 40
        + attrition_60d * 30
        + conflict      * 15
        + (1 - engagement) * 15
    )
    return max(0, min(100, int(100 - risk_component)))


def run_ml_predictions(weeks: list[date]) -> int:
    """
    For every employee that has behavioral features, run the ML models
    and upsert results into health_scores.

    Returns: number of health_scores rows written.
    """
    from sqlalchemy import text
    from sqlalchemy.orm import Session
    import src.ml.burnout_predictor  as bp  # noqa: PLC0415
    import src.ml.attrition_model    as am  # noqa: PLC0415
    import src.ml.conflict_detector  as cd  # noqa: PLC0415

    # Pre-load models once (avoid reloading on every employee)
    try:
        bp._load_artifact()
        am._load_artifact()
        cd._load_artifact()
        _ok("ML models loaded from disk.")
    except FileNotFoundError as exc:
        _err(f"Model file missing: {exc}")
        _err("Run:  python scripts/train_all_models.py --skip-bert")
        return 0

    engine = _get_engine()
    total_written = 0

    for week_idx, week in enumerate(weeks, 1):
        t0 = time.time()

        with Session(engine) as session:
            # Fetch all employees with behavioral features for this week
            bf_rows = session.execute(
                text("""
                    SELECT
                        bf.employee_id,
                        bf.avg_response_hours,
                        bf.after_hours_count,
                        bf.message_count,
                        bf.avg_message_length,
                        bf.participation_rate,
                        bf.manager_comm_ratio,
                        bf.sentiment_velocity,
                        e.tenure_months,
                        hs.team_id
                    FROM behavioral_features bf
                    JOIN employees e ON e.id = bf.employee_id
                    LEFT JOIN health_scores hs
                        ON hs.employee_id = bf.employee_id
                        AND hs.week = (
                            SELECT MAX(week) FROM health_scores
                            WHERE employee_id = bf.employee_id
                        )
                    WHERE bf.week = :week
                """),
                {"week": week},
            ).fetchall()

            if not bf_rows:
                # Fall back: find employees active that week from message_metadata
                bf_rows = session.execute(
                    text("""
                        SELECT
                            e.id,
                            2.0, 0, 20, 50.0, 0.8, 0.2, 0.0,
                            e.tenure_months,
                            NULL
                        FROM employees e
                        WHERE e.id IN (
                            SELECT DISTINCT sender_id
                            FROM message_metadata
                            WHERE DATE_TRUNC('week', timestamp)::date = :week
                        )
                    """),
                    {"week": week},
                ).fetchall()

            # Fetch network metrics for this week
            network_rows = session.execute(
                text("""
                    SELECT employee_a, employee_b, relationship_health,
                           avg_sentiment
                    FROM comm_graph WHERE week = :week
                """),
                {"week": week},
            ).fetchall()

        # Build per-employee conflict proxy from comm_graph
        conflict_by_emp: dict[int, float] = {}
        for nr in network_rows:
            emp_a, emp_b, rel_health, avg_sent = nr
            rel_health = float(rel_health or 1.0)
            avg_sent   = float(avg_sent or 0.0)
            # Low relationship health + negative sentiment → conflict proxy
            raw_conflict = max(0.0, (1 - rel_health) * 0.6 + max(0, -avg_sent) * 0.4)
            for emp in (emp_a, emp_b):
                conflict_by_emp[emp] = max(conflict_by_emp.get(emp, 0.0), raw_conflict)

        written_this_week = 0
        with Session(engine) as session:
            for row in bf_rows:
                (
                    emp_id,
                    avg_resp, after_h, msg_cnt, avg_len,
                    prate, mgr_ratio, svel,
                    tenure, team_id,
                ) = row

                bf_dict = {
                    "avg_response_hours": float(avg_resp or 2),
                    "after_hours_count":  int(after_h or 0),
                    "message_count":      int(msg_cnt or 0),
                    "avg_message_length": float(avg_len or 50),
                    "participation_rate": float(prate or 0.8),
                    "manager_comm_ratio": float(mgr_ratio or 0.2),
                    "sentiment_velocity": float(svel or 0),
                }

                hr_feats = _map_behavioral_to_hr_features(bf_dict, int(tenure or 12))

                try:
                    b_result = bp.predict(hr_feats)
                    a_result = am.predict(hr_feats)
                except Exception as exc:
                    _warn(f"Prediction failed for emp {emp_id}: {exc}")
                    continue

                burnout_risk    = b_result["burnout_risk"]
                attrition_30d   = a_result["attrition_risk_30d"]
                attrition_60d   = a_result["attrition_risk_60d"]
                attrition_90d   = a_result["attrition_risk_90d"]
                conflict_risk   = min(1.0, conflict_by_emp.get(emp_id, 0.0))
                engagement      = max(0.0, min(1.0,
                    float(prate or 0.8) * 0.7 + max(0, float(svel or 0)) * 0.3
                ))
                overall         = _compute_overall_score(
                    burnout_risk, attrition_60d, conflict_risk, engagement
                )

                # Build flags list
                flags: list[str] = []
                if burnout_risk >= BURNOUT_CRITICAL:
                    flags.append("CRITICAL_BURNOUT")
                elif burnout_risk >= BURNOUT_HIGH:
                    flags.append("HIGH_BURNOUT")
                if attrition_60d >= ATTRITION_HIGH:
                    flags.append("FLIGHT_RISK")
                elif attrition_60d >= ATTRITION_MEDIUM:
                    flags.append("ATTRITION_WATCH")
                if conflict_risk >= CONFLICT_HIGH:
                    flags.append("CONFLICT_RISK")
                if int(after_h or 0) >= 15:
                    flags.append("OVERWORK")
                if float(prate or 1.0) < 0.4:
                    flags.append("DISENGAGED")

                shap_values = b_result.get("shap_values", {})

                try:
                    session.execute(
                        text("""
                            INSERT INTO health_scores
                                (employee_id, team_id, week, burnout_risk,
                                 attrition_risk_30d, attrition_risk_60d,
                                 attrition_risk_90d, conflict_risk,
                                 engagement_score, overall_score,
                                 shap_values, flags)
                            VALUES
                                (:eid, :tid, :week, :burnout,
                                 :attr30, :attr60, :attr90, :conflict,
                                 :engage, :overall,
                                 cast(:shap as jsonb), cast(:flags as jsonb))
                            ON CONFLICT (employee_id, week)
                            DO UPDATE SET
                                burnout_risk       = EXCLUDED.burnout_risk,
                                attrition_risk_30d = EXCLUDED.attrition_risk_30d,
                                attrition_risk_60d = EXCLUDED.attrition_risk_60d,
                                attrition_risk_90d = EXCLUDED.attrition_risk_90d,
                                conflict_risk      = EXCLUDED.conflict_risk,
                                engagement_score   = EXCLUDED.engagement_score,
                                overall_score      = EXCLUDED.overall_score,
                                shap_values        = EXCLUDED.shap_values,
                                flags              = EXCLUDED.flags,
                                computed_at        = now()
                        """),
                        {
                            "eid":     emp_id,
                            "tid":     team_id,
                            "week":    week,
                            "burnout": round(burnout_risk, 4),
                            "attr30":  round(attrition_30d, 4),
                            "attr60":  round(attrition_60d, 4),
                            "attr90":  round(attrition_90d, 4),
                            "conflict":round(conflict_risk, 4),
                            "engage":  round(engagement, 4),
                            "overall": overall,
                            "shap":    json.dumps(shap_values),
                            "flags":   json.dumps(flags),
                        },
                    )
                    written_this_week += 1
                except Exception as exc:
                    _warn(f"health_scores upsert failed for emp {emp_id}: {exc}")
                    session.rollback()
                    continue

            session.commit()

        elapsed = time.time() - t0
        print(
            f"  Week {week_idx:>2}/{len(weeks)}  {week}  →  "
            f"{written_this_week} health scores  ({_elapsed(elapsed)})",
            flush=True,
        )
        total_written += written_this_week

    return total_written


# ── Step 4e: alert generation ─────────────────────────────────────────────────

def generate_alerts(weeks: list[date]) -> int:
    """
    Scan health_scores for each week and create alerts for employees
    above risk thresholds. Only generates alerts for the most recent week
    to avoid duplicate flooding.
    """
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    # Only generate fresh alerts for the most recent week
    latest_week = max(weeks)

    engine  = _get_engine()
    created = 0

    with Session(engine) as session:
        rows = session.execute(
            text("""
                SELECT employee_id, burnout_risk, attrition_risk_60d,
                       attrition_risk_90d, conflict_risk, flags
                FROM health_scores
                WHERE week = :week
            """),
            {"week": latest_week},
        ).fetchall()

        for row in rows:
            emp_id, burnout, attr60, attr90, conflict, flags = row
            burnout  = float(burnout  or 0)
            attr60   = float(attr60   or 0)
            conflict = float(conflict or 0)

            # Burnout alert
            if burnout >= BURNOUT_HIGH:
                sev  = "critical" if burnout >= BURNOUT_CRITICAL else "high"
                desc = (
                    f"Burnout risk reached {burnout:.0%} — "
                    f"{'critical: immediate intervention needed' if sev == 'critical' else 'elevated: schedule check-in this week'}."
                )
                try:
                    session.execute(
                        text("""
                            INSERT INTO alerts
                                (employee_id, alert_type, severity, description, recommendations)
                            VALUES (:eid, 'burnout', :sev, :desc, cast(:recs as jsonb))
                            ON CONFLICT DO NOTHING
                        """),
                        {
                            "eid":  emp_id,
                            "sev":  sev,
                            "desc": desc,
                            "recs": json.dumps([
                                "Schedule 1-on-1 within 24 hours",
                                "Review and reduce workload immediately",
                                "Share EAP resources",
                            ]),
                        },
                    )
                    created += 1
                except Exception:
                    session.rollback()

            # Attrition alert
            if attr60 >= ATTRITION_HIGH:
                try:
                    session.execute(
                        text("""
                            INSERT INTO alerts
                                (employee_id, alert_type, severity, description, recommendations)
                            VALUES (:eid, 'attrition', 'high', :desc, cast(:recs as jsonb))
                            ON CONFLICT DO NOTHING
                        """),
                        {
                            "eid":  emp_id,
                            "desc": (
                                f"Flight risk at {attr60:.0%} within 60 days "
                                f"({attr90:.0%} within 90 days). "
                                f"Retention conversation needed."
                            ),
                            "recs": json.dumps([
                                "Have a career growth conversation this week",
                                "Consider compensation review",
                                "Increase manager visibility and recognition",
                            ]),
                        },
                    )
                    created += 1
                except Exception:
                    session.rollback()

            # Conflict alert
            if conflict >= CONFLICT_HIGH:
                try:
                    session.execute(
                        text("""
                            INSERT INTO alerts
                                (employee_id, alert_type, severity, description, recommendations)
                            VALUES (:eid, 'conflict', 'high', :desc, cast(:recs as jsonb))
                            ON CONFLICT DO NOTHING
                        """),
                        {
                            "eid":  emp_id,
                            "desc": (
                                f"Conflict risk {conflict:.0%} detected in "
                                f"communication graph patterns."
                            ),
                            "recs": json.dumps([
                                "Observe team communication dynamics",
                                "Consider structured team alignment session",
                                "One-on-one check-in to identify interpersonal friction",
                            ]),
                        },
                    )
                    created += 1
                except Exception:
                    session.rollback()

        session.commit()

    return created


# ── Step 5: final summary ─────────────────────────────────────────────────────

def print_final_summary(results: dict) -> None:
    print()
    print(f"\n{_WIDE}")
    print(f"  {'ENRON PIPELINE COMPLETE':^58}")
    print(_WIDE)

    rows = [
        ("Employees loaded",         results.get("employees",          "—")),
        ("Emails processed",         results.get("emails_processed",   "—")),
        ("Behavioral feature rows",  results.get("behavioral_rows",    "—")),
        ("Graph edges created",      results.get("graph_edges",        "—")),
        ("Network analyses run",     results.get("network_analyzed",   "—")),
        ("Health scores computed",   results.get("health_scores",      "—")),
        ("New alerts generated",     results.get("alerts_generated",   "—")),
        ("Total pipeline time",      results.get("total_elapsed",      "—")),
    ]

    for label, value in rows:
        value_str = f"{value:,}" if isinstance(value, int) else str(value)
        print(f"  {label:<32}  {value_str}")

    print(_WIDE)
    print()
    print("  Dashboard now shows real Enron communication data.")
    print("  Open:  http://localhost:3000")
    print()

    # Highlight any high-risk findings
    n_crit = results.get("critical_alerts", 0)
    n_high = results.get("high_alerts", 0)
    if n_crit:
        print(f"  🔴  {n_crit} CRITICAL alerts generated — immediate attention needed.")
    if n_high:
        print(f"  🟠  {n_high} HIGH alerts generated.")
    if not n_crit and not n_high:
        print("  🟢  No critical alerts — workforce appears in healthy range.")

    print()
    print("  Run again anytime to refresh with latest data:")
    print("    python scripts/load_enron_data.py --pipeline-only")
    print(_WIDE)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _get_weeks(n_weeks: int) -> list[date]:
    """Return the last n_weeks Monday dates in ascending order."""
    today = date.today()
    # Most recent Monday
    last_monday = today - timedelta(days=today.weekday())
    weeks = [last_monday - timedelta(weeks=i) for i in range(n_weeks)]
    return sorted(weeks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load Enron data and run the full CogniTeam analytics pipeline"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of emails loaded (for quick tests, e.g. 50000)",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=12,
        help="Number of weeks of analytics to run (default: 12)",
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip the data-cleaning step",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Re-run cleaning even if emails_clean.csv already exists",
    )
    parser.add_argument(
        "--skip-loading",
        action="store_true",
        help="Skip the email loading step (assumes message_metadata is already populated)",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Skip cleaning + loading; only re-run feature eng / graph / ML pipeline",
    )
    args = parser.parse_args()

    pipeline_only = args.pipeline_only
    skip_loading  = args.skip_loading or pipeline_only
    skip_cleaning = args.skip_cleaning or pipeline_only

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        CogniTeam — Enron Data Load & Pipeline           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Database:  {DATABASE_URL.split('@')[-1]}")
    print(f"  Weeks:     {args.weeks}")
    if args.limit:
        print(f"  Email cap: {args.limit:,} rows (test mode)")
    print()

    t_total = time.time()
    results: dict = {}

    # ── Step 1: check CSV ───────────────────────────────────────────────────
    if not skip_loading:
        _banner("Step 1 / 7 — Pre-flight Check")
        raw_path = check_enron_csv()
    else:
        raw_path = DATA_RAW / "enron_emails.csv"
        _info("Skipping email load (--skip-loading / --pipeline-only).")
        msg_count = _scalar("SELECT COUNT(*) FROM message_metadata")
        if msg_count == 0:
            _err("message_metadata is empty. Run without --skip-loading first.")
            sys.exit(1)
        _ok(f"message_metadata has {msg_count:,} rows — using existing data.")

    # ── Step 2: clean ───────────────────────────────────────────────────────
    if not skip_cleaning:
        clean_enron_emails(raw_path, force=args.force_clean)
    else:
        _info("Skipping data cleaning.")

    # ── Step 3: load emails ─────────────────────────────────────────────────
    if not skip_loading:
        inserted, emp_count = load_enron_emails(raw_path, limit=args.limit)
        results["emails_processed"] = inserted
        results["employees"]        = emp_count
    else:
        results["emails_processed"] = _scalar("SELECT COUNT(*) FROM message_metadata")
        results["employees"]        = _scalar("SELECT COUNT(*) FROM employees")

    # ── Step 4: analytics pipeline ──────────────────────────────────────────
    weeks = _get_weeks(args.weeks)
    _info(f"Running pipeline for {len(weeks)} weeks: {weeks[0]} → {weeks[-1]}")

    _banner("Step 4a / 7 — Behavioral Feature Engineering")
    bf_total = run_feature_engineering(weeks)
    results["behavioral_rows"] = bf_total
    _ok(f"Behavioral features written: {bf_total:,} rows")

    _banner("Step 4b / 7 — Communication Graph Building")
    edges_by_week = run_graph_building(weeks)
    total_edges = sum(edges_by_week.values())
    results["graph_edges"] = total_edges
    _ok(f"Graph edges written: {total_edges:,} total across {len(weeks)} weeks")

    _banner("Step 4c / 7 — Network Analysis")
    analyzed = run_network_analysis(weeks)
    results["network_analyzed"] = sum(analyzed.values())
    _ok(f"Network metrics computed for {results['network_analyzed']} employee-weeks")

    _banner("Step 4d / 7 — ML Predictions → health_scores")
    hs_count = run_ml_predictions(weeks)
    results["health_scores"] = hs_count
    _ok(f"Health scores written: {hs_count:,} rows")

    _banner("Step 4e / 7 — Alert Generation")
    alert_count = generate_alerts(weeks)
    results["alerts_generated"] = alert_count
    _ok(f"Alerts generated: {alert_count}")

    # Count alert severities for summary
    results["critical_alerts"] = _scalar(
        "SELECT COUNT(*) FROM alerts WHERE severity = 'critical' AND resolved = FALSE"
    )
    results["high_alerts"] = _scalar(
        "SELECT COUNT(*) FROM alerts WHERE severity = 'high' AND resolved = FALSE"
    )

    results["total_elapsed"] = _elapsed(time.time() - t_total)

    # ── Step 5: final summary ───────────────────────────────────────────────
    _banner("Step 5 / 7 — Summary")
    print_final_summary(results)


if __name__ == "__main__":
    main()
