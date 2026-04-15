"""
Seed the database with realistic demo data so the dashboard
has something to display when started fresh.
"""
import os
import sys
import json
import random
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from passlib.context import CryptContext

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")
engine = create_engine(DATABASE_URL)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

random.seed(42)

DEPARTMENTS = ["Engineering", "Product", "Design", "Marketing", "Sales", "Operations"]
ROLES = {
    "Engineering": ["Backend Engineer", "Frontend Engineer", "DevOps", "Staff Engineer"],
    "Product": ["Product Manager", "Senior PM", "Director of Product"],
    "Design": ["UX Designer", "Product Designer", "Design Lead"],
    "Marketing": ["Marketing Manager", "Content Strategist", "Growth Lead"],
    "Sales": ["Account Executive", "Sales Manager", "SDR"],
    "Operations": ["Operations Manager", "Data Analyst", "Program Manager"],
}
EMOTIONS = ["joy", "frustration", "anxiety", "neutral", "sadness"]
ALERT_TYPES = ["burnout", "conflict", "attrition", "isolation"]
SEVERITIES = ["low", "medium", "high", "critical"]


def seed_employees(session):
    print("Seeding employees...")
    employees = []

    # Create 24 employees across 4 teams
    for dept_idx, dept in enumerate(DEPARTMENTS[:4]):  # 4 departments
        for i in range(6):  # 6 people per department
            role = random.choice(ROLES[dept])
            tenure = random.randint(3, 72)
            emp_id = session.execute(
                text("""
                    INSERT INTO employees (name, department, role, tenure_months)
                    VALUES (:name, :dept, :role, :tenure)
                    RETURNING id
                """),
                {
                    "name": f"Employee_{dept[:3].lower()}_{i+1:02d}",
                    "dept": dept,
                    "role": role,
                    "tenure": tenure,
                },
            ).scalar_one()
            employees.append({"id": emp_id, "dept": dept, "team_id": dept_idx + 1})

    # Assign managers (first person in each dept is the manager)
    dept_managers = {}
    for emp in employees:
        dept = emp["dept"]
        if dept not in dept_managers:
            dept_managers[dept] = emp["id"]

    for emp in employees:
        manager_id = dept_managers.get(emp["dept"])
        if manager_id and manager_id != emp["id"]:
            session.execute(
                text("UPDATE employees SET manager_id = :mid WHERE id = :eid"),
                {"mid": manager_id, "eid": emp["id"]},
            )

    session.commit()
    print(f"  Created {len(employees)} employees")
    return employees


def seed_sentiment_and_features(session, employees):
    print("Seeding sentiment scores and behavioral features (12 weeks)...")
    today = date.today()
    # Start from the Monday 12 weeks ago
    start_monday = today - timedelta(weeks=12, days=today.weekday())

    count = 0
    for emp in employees:
        emp_id = emp["id"]
        # Give at-risk employees worse baseline scores
        is_at_risk = emp_id % 5 == 0  # every 5th employee is "at risk"

        base_sentiment = random.uniform(-0.3, 0.2) if is_at_risk else random.uniform(0.1, 0.6)
        base_after_hrs = random.randint(8, 25) if is_at_risk else random.randint(0, 6)

        for week_offset in range(12):
            week = start_monday + timedelta(weeks=week_offset)

            # Trend: at-risk employees get worse over time
            trend = -0.03 * week_offset if is_at_risk else random.uniform(-0.02, 0.02)
            sentiment = max(-1.0, min(1.0, base_sentiment + trend + random.gauss(0, 0.05)))
            emotion = random.choice(EMOTIONS)
            if is_at_risk:
                emotion = random.choice(["frustration", "anxiety", "sadness", "neutral"])

            # Upsert sentiment
            session.execute(
                text("""
                    INSERT INTO sentiment_scores (employee_id, week, sentiment, emotion, confidence)
                    VALUES (:eid, :week, :sent, :emo, :conf)
                    ON CONFLICT (employee_id, week) DO UPDATE SET
                        sentiment = EXCLUDED.sentiment, emotion = EXCLUDED.emotion
                """),
                {"eid": emp_id, "week": week, "sent": round(sentiment, 4),
                 "emo": emotion, "conf": round(random.uniform(0.6, 0.95), 3)},
            )

            after_hrs = max(0, base_after_hrs + week_offset * (1 if is_at_risk else 0) + random.randint(-2, 2))
            msg_count = random.randint(5, 50)
            participation = round(random.uniform(0.3, 0.6) if is_at_risk else random.uniform(0.7, 1.0), 3)
            sent_velocity = round(trend + random.gauss(0, 0.02), 4)

            # Upsert behavioral features
            session.execute(
                text("""
                    INSERT INTO behavioral_features
                        (employee_id, week, avg_response_hours, after_hours_count,
                         message_count, avg_message_length, participation_rate,
                         manager_comm_ratio, sentiment_velocity)
                    VALUES
                        (:eid, :week, :arh, :ahc, :mc, :aml, :pr, :mcr, :sv)
                    ON CONFLICT (employee_id, week) DO UPDATE SET
                        after_hours_count = EXCLUDED.after_hours_count,
                        participation_rate = EXCLUDED.participation_rate,
                        sentiment_velocity = EXCLUDED.sentiment_velocity
                """),
                {
                    "eid": emp_id, "week": week,
                    "arh": round(random.uniform(15, 36) if is_at_risk else random.uniform(0.5, 8), 2),
                    "ahc": after_hrs,
                    "mc": msg_count,
                    "aml": round(random.uniform(5, 20), 1),
                    "pr": participation,
                    "mcr": round(random.uniform(0.05, 0.2), 3),
                    "sv": sent_velocity,
                },
            )
            count += 1

    session.commit()
    print(f"  Created {count} weekly records")


def seed_health_scores(session, employees):
    print("Seeding health scores...")
    today = date.today()
    current_week = today - timedelta(days=today.weekday())

    for emp in employees:
        emp_id = emp["id"]
        team_id = emp["team_id"]
        is_at_risk = emp_id % 5 == 0
        is_medium = emp_id % 3 == 0

        if is_at_risk:
            burnout = round(random.uniform(0.72, 0.95), 3)
            overall = random.randint(18, 38)
            attrition_base = round(random.uniform(0.55, 0.85), 3)
            conflict = round(random.uniform(0.45, 0.75), 3)
            engagement = round(random.uniform(0.15, 0.35), 3)
            flags = ["high_burnout_risk", "declining_sentiment", "excessive_after_hours"]
        elif is_medium:
            burnout = round(random.uniform(0.35, 0.65), 3)
            overall = random.randint(42, 66)
            attrition_base = round(random.uniform(0.25, 0.50), 3)
            conflict = round(random.uniform(0.20, 0.45), 3)
            engagement = round(random.uniform(0.45, 0.65), 3)
            flags = ["moderate_after_hours"]
        else:
            burnout = round(random.uniform(0.05, 0.28), 3)
            overall = random.randint(72, 95)
            attrition_base = round(random.uniform(0.05, 0.22), 3)
            conflict = round(random.uniform(0.05, 0.20), 3)
            engagement = round(random.uniform(0.70, 0.95), 3)
            flags = []

        shap_values = {
            "satisfaction_level": round(random.uniform(-0.5, 0.1) if is_at_risk else random.uniform(0.1, 0.5), 4),
            "average_montly_hours": round(random.uniform(0.1, 0.6) if is_at_risk else random.uniform(-0.3, 0.1), 4),
            "after_hours_count": round(random.uniform(0.05, 0.4) if is_at_risk else random.uniform(-0.1, 0.05), 4),
            "participation_rate": round(random.uniform(-0.4, -0.1) if is_at_risk else random.uniform(0.1, 0.4), 4),
            "sentiment_velocity": round(random.uniform(-0.3, -0.05) if is_at_risk else random.uniform(0.0, 0.2), 4),
        }

        session.execute(
            text("""
                INSERT INTO health_scores
                    (employee_id, team_id, week, burnout_risk,
                     attrition_risk_30d, attrition_risk_60d, attrition_risk_90d,
                     conflict_risk, engagement_score, overall_score,
                     shap_values, flags)
                VALUES
                    (:eid, :tid, :week, :br,
                     :a30, :a60, :a90,
                     :cr, :eng, :os,
                     cast(:shap as jsonb), cast(:flags as jsonb))
                ON CONFLICT (employee_id, week) DO UPDATE SET
                    burnout_risk = EXCLUDED.burnout_risk,
                    overall_score = EXCLUDED.overall_score
            """),
            {
                "eid": emp_id, "tid": team_id, "week": current_week,
                "br": burnout,
                "a30": round(attrition_base * 0.35, 3),
                "a60": round(attrition_base * 0.55, 3),
                "a90": round(attrition_base * 0.70, 3),
                "cr": conflict,
                "eng": engagement,
                "os": overall,
                "shap": json.dumps(shap_values),
                "flags": json.dumps(flags),
            },
        )

    session.commit()
    print(f"  Created health scores for {len(employees)} employees")


def seed_alerts(session, employees):
    print("Seeding alerts...")
    alert_count = 0
    recommendations_map = {
        "burnout": ["Schedule 1-on-1 this week", "Review workload and redistribute tasks", "Discuss work-life balance strategies"],
        "conflict": ["Arrange mediation session", "Facilitate team alignment meeting", "Check in with both parties separately"],
        "attrition": ["Have a career development conversation", "Review compensation", "Recognize recent contributions publicly"],
        "isolation": ["Pair with a buddy or mentor", "Invite to cross-team projects", "Check in about social engagement"],
    }

    for emp in employees:
        emp_id = emp["id"]
        is_at_risk = emp_id % 5 == 0
        is_medium = emp_id % 7 == 0

        if is_at_risk:
            for alert_type in ["burnout", "attrition"]:
                severity = "critical" if alert_type == "burnout" else "high"
                session.execute(
                    text("""
                        INSERT INTO alerts (employee_id, alert_type, severity, description, recommendations)
                        VALUES (:eid, :at, :sev, :desc, cast(:recs as jsonb))
                    """),
                    {
                        "eid": emp_id,
                        "at": alert_type,
                        "sev": severity,
                        "desc": f"{'Critical burnout' if alert_type == 'burnout' else 'High attrition'} risk detected. Immediate manager attention recommended.",
                        "recs": json.dumps(recommendations_map[alert_type]),
                    },
                )
                alert_count += 1

        elif is_medium:
            session.execute(
                text("""
                    INSERT INTO alerts (employee_id, alert_type, severity, description, recommendations)
                    VALUES (:eid, :at, :sev, :desc, cast(:recs as jsonb))
                """),
                {
                    "eid": emp_id,
                    "at": random.choice(["conflict", "isolation"]),
                    "sev": "medium",
                    "desc": "Moderate risk signals detected in communication patterns. Worth monitoring.",
                    "recs": json.dumps(recommendations_map["conflict"]),
                },
            )
            alert_count += 1

    session.commit()
    print(f"  Created {alert_count} alerts")


def seed_comm_graph(session, employees):
    print("Seeding communication graph...")
    today = date.today()
    current_week = today - timedelta(days=today.weekday())
    count = 0

    # Create edges between employees in same department
    dept_groups = {}
    for emp in employees:
        dept_groups.setdefault(emp["dept"], []).append(emp["id"])

    for dept, emp_ids in dept_groups.items():
        for i, src in enumerate(emp_ids):
            for dst in emp_ids[i+1:]:
                rel_health = round(random.uniform(0.4, 0.95), 3)
                session.execute(
                    text("""
                        INSERT INTO comm_graph
                            (employee_a, employee_b, week, message_count,
                             avg_sentiment, avg_response_hours, relationship_health)
                        VALUES (:a, :b, :week, :mc, :as, :arh, :rh)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "a": src, "b": dst, "week": current_week,
                        "mc": random.randint(5, 50),
                        "as": round(random.uniform(-0.3, 0.7), 3),
                        "arh": round(random.uniform(0.5, 12), 2),
                        "rh": rel_health,
                    },
                )
                count += 1

    session.commit()
    print(f"  Created {count} graph edges")


def seed_users(session, employees):
    print("Seeding users...")

    users = [
        {"email": "hr@cogniteam.ai", "password": "hr123", "role": "hr", "emp_id": None},
        {"email": "manager@cogniteam.ai", "password": "manager123", "role": "manager", "emp_id": employees[0]["id"]},
        {"email": "manager2@cogniteam.ai", "password": "manager123", "role": "manager", "emp_id": employees[6]["id"]},
        {"email": "employee@cogniteam.ai", "password": "employee123", "role": "employee", "emp_id": employees[1]["id"]},
    ]

    for user in users:
        hashed = pwd_context.hash(user["password"])
        session.execute(
            text("""
                INSERT INTO users (email, hashed_password, role, employee_id)
                VALUES (:email, :hpw, :role, :eid)
                ON CONFLICT (email) DO UPDATE SET hashed_password = EXCLUDED.hashed_password
            """),
            {"email": user["email"], "hpw": hashed, "role": user["role"], "eid": user["emp_id"]},
        )

    session.commit()
    print(f"  Created {len(users)} user accounts")
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║         DEMO LOGIN CREDENTIALS           ║")
    print("  ╠══════════════════════════════════════════╣")
    for user in users:
        print(f"  ║  {user['role'].upper():<8} │ {user['email']:<24} ║")
        print(f"  ║           │ password: {user['password']:<19} ║")
        print("  ╠══════════════════════════════════════════╣")
    print("  ╚══════════════════════════════════════════╝")


def main():
    print("\n🚀 Seeding CogniTeam demo database...\n")
    with Session(engine) as session:
        # Check if already seeded
        count = session.execute(text("SELECT COUNT(*) FROM employees")).scalar()
        if count > 0:
            print(f"  Database already has {count} employees. Skipping seed.")
            print("  (Drop tables and re-run schema to reseed)\n")
            return

        employees = seed_employees(session)
        seed_sentiment_and_features(session, employees)
        seed_health_scores(session, employees)
        seed_alerts(session, employees)
        seed_comm_graph(session, employees)
        seed_users(session, employees)

    print("\n✅ Demo database seeded successfully!\n")


if __name__ == "__main__":
    main()
