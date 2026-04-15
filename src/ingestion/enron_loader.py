"""
Enron Email Dataset Loader
Reads the Enron CSV, extracts metadata only (no content stored),
maps employee names to IDs, and writes to the database.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")
DATA_RAW = Path(os.getenv("DATA_DIR", "data")) / "raw"
DATA_PROCESSED = Path(os.getenv("DATA_DIR", "data")) / "processed"

WORK_START_HOUR = 8
WORK_END_HOUR = 19  # 7 pm

_email_re = re.compile(r"[\w.+-]+@[\w-]+\.\w+")
_sig_patterns = re.compile(
    r"(_{3,}|={3,}|-{3,}|Best regards|Kind regards|Sincerely|Cheers|Thanks,|Thank you,|Regards,)",
    re.IGNORECASE,
)


def _hash_name(name: str) -> str:
    """Deterministic anonymous ID for an employee name."""
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()[:12]


def _is_after_hours(dt: datetime) -> bool:
    hour = dt.hour
    return hour < WORK_START_HOUR or hour >= WORK_END_HOUR


def _strip_signature(body: str) -> str:
    """Remove boilerplate signatures from email body."""
    match = _sig_patterns.search(body)
    return body[: match.start()].strip() if match else body.strip()


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _extract_address(raw: str) -> Optional[str]:
    """Extract first email address from a raw header field."""
    if not isinstance(raw, str):
        return None
    match = _email_re.search(raw)
    return match.group(0).lower() if match else None


def _parse_timestamp(raw: str) -> Optional[datetime]:
    """Parse Enron email timestamps (varied formats) to UTC."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    # Enron format: "Mon, 14 May 2001 16:39:43 -0700 (PDT)"
    raw = re.sub(r"\s*\(.*?\)", "", raw).strip()
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M %z",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────

def _get_or_create_employee(session: Session, email: str, name_map: dict[str, int]) -> Optional[int]:
    if email in name_map:
        return name_map[email]
    # Derive display name from email prefix and anonymize
    local = email.split("@")[0].replace(".", " ").replace("-", " ").title()
    anon_name = f"Employee_{_hash_name(local)}"
    result = session.execute(
        text("INSERT INTO employees (name) VALUES (:name) RETURNING id"),
        {"name": anon_name},
    )
    emp_id: int = result.scalar_one()
    session.commit()
    name_map[email] = emp_id
    return emp_id


# ─────────────────────────────────────────────────────────────
# Thread response-time calculation
# ─────────────────────────────────────────────────────────────

def _compute_response_times(df: pd.DataFrame) -> pd.Series:
    """
    For each email, find the previous email in the same thread
    sent by a different person and compute hours elapsed.
    Operates on the full df sorted by (message_id, date).
    Returns a Series of float (NaN when no prior message exists).
    """
    response_hours = pd.Series([None] * len(df), dtype=float)
    # Group by thread using 'X-Thread-Info' or subject
    if "subject" not in df.columns:
        return response_hours

    df = df.copy()
    df["_thread_key"] = df["subject"].str.replace(r"^(Re:\s*)+", "", regex=True).str.strip().str.lower()
    grouped = df.groupby("_thread_key")

    for _, thread in grouped:
        thread_sorted = thread.sort_values("date_parsed")
        prev_sender = None
        prev_time = None
        for idx, row in thread_sorted.iterrows():
            if prev_sender is not None and row["sender"] != prev_sender and prev_time is not None:
                delta = (row["date_parsed"] - prev_time).total_seconds() / 3600.0
                if 0 < delta < 168:  # ignore >1-week gaps
                    response_hours.at[idx] = round(delta, 2)
            prev_sender = row["sender"]
            prev_time = row["date_parsed"]

    return response_hours


# ─────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────

def load_enron_emails(csv_path: Optional[Path] = None, limit: Optional[int] = None) -> None:
    """
    Full pipeline: read CSV → parse → insert metadata only into DB.

    Args:
        csv_path: Path to enron_emails.csv. Defaults to data/raw/enron_emails.csv.
        limit:    Optional row limit for testing.
    """
    csv_path = csv_path or DATA_RAW / "enron_emails.csv"
    if not csv_path.exists():
        logger.error(f"Enron CSV not found at {csv_path}. Download from Kaggle first.")
        return

    logger.info(f"Loading Enron emails from {csv_path} …")
    engine = create_engine(DATABASE_URL)

    # Read in chunks to handle 500k rows
    chunk_size = 10_000
    name_map: dict[str, int] = {}
    total_processed = 0
    total_inserted = 0

    reader = pd.read_csv(csv_path, chunksize=chunk_size, nrows=limit)

    for chunk_idx, chunk in enumerate(reader):
        chunk.columns = [c.lower().strip() for c in chunk.columns]

        # Enron CSV columns: file, message
        if "message" not in chunk.columns:
            logger.warning("Expected 'message' column not found. Check CSV format.")
            break

        records = []
        for _, row in chunk.iterrows():
            raw_msg: str = str(row.get("message", ""))
            # Parse headers from raw message text
            header: dict[str, str] = {}
            body_lines: list[str] = []
            in_body = False
            for line in raw_msg.splitlines():
                if in_body:
                    body_lines.append(line)
                elif line.strip() == "":
                    in_body = True
                elif ":" in line:
                    key, _, val = line.partition(":")
                    header[key.strip().lower()] = val.strip()

            sender_raw = header.get("from", "")
            receiver_raw = header.get("to", "")
            date_raw = header.get("date", "")
            subject = header.get("subject", "")

            sender = _extract_address(sender_raw)
            receiver = _extract_address(receiver_raw)
            dt = _parse_timestamp(date_raw)

            if not sender or not receiver or not dt:
                continue

            body = _strip_signature("\n".join(body_lines))
            wc = _word_count(body)
            after_hours = _is_after_hours(dt)

            records.append({
                "sender": sender,
                "receiver": receiver,
                "date_parsed": dt,
                "subject": subject,
                "word_count": wc,
                "is_after_hours": after_hours,
            })

        if not records:
            continue

        chunk_df = pd.DataFrame(records)
        response_hours_series = _compute_response_times(chunk_df)

        with Session(engine) as session:
            for i, record in enumerate(records):
                try:
                    sender_id = _get_or_create_employee(session, record["sender"], name_map)
                    receiver_id = _get_or_create_employee(session, record["receiver"], name_map)
                    if sender_id is None or receiver_id is None:
                        continue

                    resp_time = response_hours_series.iloc[i]

                    session.execute(
                        text("""
                            INSERT INTO message_metadata
                                (sender_id, receiver_id, channel, timestamp,
                                 word_count, is_after_hours, response_time_hours)
                            VALUES
                                (:sender_id, :receiver_id, 'email', :ts,
                                 :wc, :aft, :rt)
                            ON CONFLICT DO NOTHING
                        """),
                        {
                            "sender_id": sender_id,
                            "receiver_id": receiver_id,
                            "ts": record["date_parsed"],
                            "wc": record["word_count"],
                            "aft": record["is_after_hours"],
                            "rt": float(resp_time) if pd.notna(resp_time) else None,
                        },
                    )
                    total_inserted += 1
                except Exception as exc:
                    logger.debug(f"Skipped record: {exc}")
                    session.rollback()
                    continue

            session.commit()

        total_processed += len(chunk)
        if total_processed % 10_000 == 0 or chunk_idx == 0:
            logger.info(
                f"Processed {total_processed:,} rows | "
                f"Inserted {total_inserted:,} metadata records | "
                f"Employees mapped: {len(name_map):,}"
            )

    # Export employee map for reference
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"email": k, "employee_id": v} for k, v in name_map.items()]
    ).to_csv(DATA_PROCESSED / "employee_map.csv", index=False)

    logger.info(
        f"Enron load complete. "
        f"Total processed: {total_processed:,} | "
        f"Metadata inserted: {total_inserted:,} | "
        f"Unique employees: {len(name_map):,}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load Enron email dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process")
    args = parser.parse_args()
    load_enron_emails(limit=args.limit)
