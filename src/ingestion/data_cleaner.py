"""
Data Cleaner
Cleans raw email and text data: deduplication, UTC normalization,
signature stripping, and consistent name anonymization.
Outputs cleaned files to data/processed/.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

DATA_RAW = Path(os.getenv("DATA_DIR", "data")) / "raw"
DATA_PROCESSED = Path(os.getenv("DATA_DIR", "data")) / "processed"

_SIG_PATTERN = re.compile(
    r"(_{3,}|={3,}|-{3,}|Best regards|Kind regards|Sincerely|"
    r"Cheers|Thanks,|Thank you,|Regards,|From:.*?Sent:)",
    re.IGNORECASE | re.DOTALL,
)
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.\w+")
_PHONE_PATTERN = re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


# ─────────────────────────────────────────────────────────────
# Anonymization helpers
# ─────────────────────────────────────────────────────────────

def anonymize_name(name: str) -> str:
    """
    Consistently maps a real name to a stable anonymous handle.
    Same input always produces the same output across runs.
    """
    digest = hashlib.sha256(name.strip().lower().encode()).hexdigest()[:8]
    return f"Emp_{digest}"


def strip_pii(text: str) -> str:
    """Remove emails, phone numbers, and URLs from text."""
    text = _EMAIL_PATTERN.sub("[EMAIL]", text)
    text = _PHONE_PATTERN.sub("[PHONE]", text)
    text = _URL_PATTERN.sub("[URL]", text)
    return text


def strip_signature(body: str) -> str:
    """Remove email signature boilerplate."""
    if not isinstance(body, str):
        return ""
    match = _SIG_PATTERN.search(body)
    return body[: match.start()].strip() if match else body.strip()


# ─────────────────────────────────────────────────────────────
# Core cleaning steps
# ─────────────────────────────────────────────────────────────

def _remove_empty_bodies(df: pd.DataFrame, body_col: str = "body") -> pd.DataFrame:
    before = len(df)
    if body_col in df.columns:
        df = df[df[body_col].notna() & (df[body_col].str.strip() != "")]
    after = len(df)
    logger.debug(f"Removed {before - after:,} rows with empty body.")
    return df


def _normalize_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    if ts_col not in df.columns:
        return df
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    dropped = df[ts_col].isna().sum()
    if dropped:
        logger.debug(f"Dropped {dropped:,} rows with unparseable timestamps.")
    df = df[df[ts_col].notna()]
    return df


def _remove_duplicates(df: pd.DataFrame, subset: Optional[list[str]] = None) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    after = len(df)
    logger.debug(f"Removed {before - after:,} duplicate rows.")
    return df


def _apply_signature_stripping(df: pd.DataFrame, body_col: str = "body") -> pd.DataFrame:
    if body_col in df.columns:
        df[body_col] = df[body_col].apply(strip_signature)
    return df


def _anonymize_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: anonymize_name(str(x)) if pd.notna(x) else x)
    return df


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def clean_emails(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full cleaning pipeline for the Enron email data.
    Writes cleaned output to data/processed/emails_clean.csv.

    Returns:
        Cleaned DataFrame.
    """
    input_path = input_path or DATA_RAW / "enron_emails.csv"
    output_path = output_path or DATA_PROCESSED / "emails_clean.csv"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.warning(f"Input not found: {input_path}")
        return pd.DataFrame()

    logger.info(f"Cleaning email data from {input_path} …")
    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows.")

    df = _remove_empty_bodies(df, body_col="message")
    df = _remove_duplicates(df)
    df = _normalize_timestamps(df, ts_col="date")
    df = _apply_signature_stripping(df, body_col="message")

    # Anonymize sender/receiver columns if present
    for col in ("from", "sender", "to", "receiver"):
        df = _anonymize_column(df, col)

    # Strip PII from message body
    if "message" in df.columns:
        df["message"] = df["message"].apply(strip_pii)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} cleaned rows to {output_path}")
    return df


def clean_reddit_posts(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Clean Reddit posts: remove duplicates, strip PII, normalize text.
    """
    input_path = input_path or DATA_RAW / "reddit_posts.csv"
    output_path = output_path or DATA_PROCESSED / "reddit_clean.csv"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.warning(f"Reddit CSV not found at {input_path}")
        return pd.DataFrame()

    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Cleaning {len(df):,} Reddit posts …")
    df = _remove_duplicates(df, subset=["post_id"])
    df = _remove_empty_bodies(df, body_col="text")
    df["text"] = df["text"].apply(strip_pii)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} cleaned Reddit posts to {output_path}")
    return df


if __name__ == "__main__":
    clean_emails()
    clean_reddit_posts()
