#!/usr/bin/env python3
"""
CogniTeam Dataset Downloader
==============================
Downloads all required datasets for training CogniTeam's ML models.

Usage:
    python scripts/download_data.py          # download everything available
    python scripts/download_data.py --skip-reddit
    python scripts/download_data.py --only hr        # hr | goemotions | reddit
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

ENV_FILE = REPO_ROOT / ".env"

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_env() -> dict:
    """Read .env file into a dict without requiring python-dotenv."""
    env: dict = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _row_count(path: Path) -> int:
    """Count data rows in a CSV (total lines minus header)."""
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return 0


def _check(label: str, path: Path, expected_rows: int | None = None) -> None:
    """Print a status line for the final report."""
    rel = str(path.relative_to(REPO_ROOT))
    if path.exists():
        rows = _row_count(path)
        row_str = f"{rows:,} rows" if rows >= 0 else "exists"
        print(f"  {rel:<42} ✅  {row_str}")
    else:
        print(f"  {rel:<42} ❌  MISSING")


def _separator(char: str = "═", width: int = 56) -> str:
    return char * width


# ── Step 1: HR Analytics (Kaggle) ────────────────────────────────────────────

def download_hr_analytics() -> bool:
    """
    Download HR Analytics dataset via Kaggle CLI.
    Returns True if the file is present after attempting download.
    """
    dest = DATA_RAW / "hr_analytics.csv"

    if dest.exists():
        rows = _row_count(dest)
        print(f"  [HR Analytics] Already present ({rows:,} rows) — skipping.")
        return True

    print("  [HR Analytics] Attempting Kaggle download…")

    # Check if kaggle package is importable; install if not
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("  [HR Analytics] Installing kaggle package…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "kaggle", "--quiet"]
        )

    # Check for kaggle.json credentials
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_cfg.exists():
        print()
        print(_separator())
        print("  KAGGLE CREDENTIALS NOT FOUND")
        print(_separator())
        print("  To download HR Analytics automatically:")
        print("  1. Go to  https://www.kaggle.com/account")
        print("  2. Scroll to 'API' and click 'Create New Token'")
        print("  3. This downloads  kaggle.json")
        print(f"  4. Move it to:    {kaggle_cfg}")
        print("  5. Run:           chmod 600 ~/.kaggle/kaggle.json")
        print("  6. Re-run this script")
        print()
        print("  Alternatively, download manually:")
        print("  https://www.kaggle.com/datasets/giripujar/hr-analytics")
        print(f"  Rename to hr_analytics.csv → place at {dest}")
        print(_separator())
        return False

    try:
        subprocess.check_call(
            [
                sys.executable, "-m", "kaggle",
                "datasets", "download",
                "-d", "giripujar/hr-analytics",
                "-p", str(DATA_RAW),
                "--unzip",
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(f"  [HR Analytics] Kaggle download failed: {exc}")
        return False

    # The Kaggle archive often extracts to HR_comma_sep.csv — rename it
    for candidate in DATA_RAW.glob("HR_comma_sep*.csv"):
        candidate.rename(dest)
        break

    if dest.exists():
        print(f"  [HR Analytics] Downloaded successfully ({_row_count(dest):,} rows).")
        return True

    print("  [HR Analytics] File not found after download — check Kaggle output above.")
    return False


# ── Step 2: GoEmotions (HuggingFace) ─────────────────────────────────────────

def download_go_emotions() -> bool:
    """
    Download GoEmotions 'simplified' split from HuggingFace and save as CSVs.
    Returns True if both files are present.
    """
    train_dest = DATA_RAW / "go_emotions_train.csv"
    val_dest   = DATA_RAW / "go_emotions_val.csv"

    if train_dest.exists() and val_dest.exists():
        print(
            f"  [GoEmotions] Already present "
            f"(train={_row_count(train_dest):,}, val={_row_count(val_dest):,}) — skipping."
        )
        return True

    print("  [GoEmotions] Downloading from HuggingFace (datasets library)…")

    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        print("  [GoEmotions] Installing 'datasets' package…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "datasets", "--quiet"]
        )
        from datasets import load_dataset  # noqa: PLC0415

    try:
        ds = load_dataset("go_emotions", "simplified")
    except Exception as exc:
        print(f"  [GoEmotions] load_dataset failed: {exc}")
        return False

    try:
        print("  [GoEmotions] Saving train split…")
        ds["train"].to_csv(str(train_dest), index=False)

        print("  [GoEmotions] Saving validation split…")
        ds["validation"].to_csv(str(val_dest), index=False)
    except Exception as exc:
        print(f"  [GoEmotions] Save failed: {exc}")
        return False

    print(
        f"  [GoEmotions] Done — "
        f"train={_row_count(train_dest):,} rows, "
        f"val={_row_count(val_dest):,} rows."
    )
    return True


# ── Step 3: Enron (manual download instructions) ─────────────────────────────

def check_enron() -> bool:
    """
    Enron dataset is ~1.7 GB; automatic download requires a paid Kaggle account
    for files this large.  We print clear instructions instead.
    """
    dest = DATA_RAW / "enron_emails.csv"

    if dest.exists():
        rows = _row_count(dest)
        print(f"  [Enron] Already present ({rows:,} rows).")
        return True

    print()
    print(_separator())
    print("  MANUAL DOWNLOAD REQUIRED FOR ENRON DATA")
    print(_separator())
    print("  1. Go to: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset")
    print("  2. Click Download  (free Kaggle account required)")
    print("  3. Unzip the downloaded archive")
    print("  4. Rename the CSV file to:  enron_emails.csv")
    print(f"  5. Place it at:  {dest}")
    print("  File size: approximately 1.7 GB, ~517 000 rows")
    print()
    print("  Once the file is in place, run:")
    print("      python -m src.ingestion.enron_loader")
    print("  to load message metadata into PostgreSQL.")
    print(_separator())
    return False


# ── Step 4: Reddit posts (PRAW) ───────────────────────────────────────────────

def download_reddit(posts_per_sub: int = 500) -> bool:
    """
    Pull posts from workplace subreddits via PRAW.
    Only runs when REDDIT_CLIENT_ID is configured in .env.
    """
    dest = DATA_RAW / "reddit_posts.csv"

    env = _load_env()
    client_id     = env.get("REDDIT_CLIENT_ID", "").strip()
    client_secret = env.get("REDDIT_CLIENT_SECRET", "").strip()
    user_agent    = env.get("REDDIT_USER_AGENT", "CogniTeam/1.0")

    if not client_id or not client_secret:
        print(
            "  [Reddit] REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set in .env — skipping."
        )
        print("  To enable Reddit data:")
        print("    1. Go to https://www.reddit.com/prefs/apps")
        print("    2. Create a 'script' app")
        print("    3. Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env")
        return False

    if dest.exists():
        rows = _row_count(dest)
        print(f"  [Reddit] Already present ({rows:,} rows) — skipping.")
        return True

    print(f"  [Reddit] Scraping {posts_per_sub} posts × 3 subreddits…")

    try:
        import praw  # noqa: PLC0415
    except ImportError:
        print("  [Reddit] Installing praw…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "praw", "--quiet"]
        )
        import praw  # noqa: PLC0415

    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pandas", "--quiet"]
        )
        import pandas as pd  # noqa: PLC0415

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )

    subreddits = ["WorkReform", "antiwork", "cscareerquestions"]
    rows: list[dict] = []

    for sub_name in subreddits:
        print(f"    Scraping r/{sub_name}…", end=" ", flush=True)
        fetched = 0
        try:
            sub = reddit.subreddit(sub_name)
            for post in sub.hot(limit=posts_per_sub):
                rows.append(
                    {
                        "subreddit":   sub_name,
                        "title":       post.title,
                        "body":        post.selftext or "",
                        "score":       post.score,
                        "created_utc": post.created_utc,
                    }
                )
                fetched += 1
            print(f"{fetched} posts")
        except praw.exceptions.PRAWException as exc:
            print(f"ERROR — {exc}")

    if not rows:
        print("  [Reddit] No posts fetched — check credentials.")
        return False

    df = pd.DataFrame(rows, columns=["subreddit", "title", "body", "score", "created_utc"])
    df.to_csv(str(dest), index=False)
    print(f"  [Reddit] Saved {len(df):,} posts to {dest.relative_to(REPO_ROOT)}")
    return True


# ── Step 5: Final status report ───────────────────────────────────────────────

def print_status_report() -> None:
    print()
    print(_separator("─"))
    print("  DATASET STATUS REPORT")
    print(_separator("─"))

    hr_path         = DATA_RAW / "hr_analytics.csv"
    ge_train_path   = DATA_RAW / "go_emotions_train.csv"
    ge_val_path     = DATA_RAW / "go_emotions_val.csv"
    enron_path      = DATA_RAW / "enron_emails.csv"
    reddit_path     = DATA_RAW / "reddit_posts.csv"

    _check("HR Analytics",        hr_path)
    _check("GoEmotions train",    ge_train_path)
    _check("GoEmotions val",      ge_val_path)
    _check("Enron emails",        enron_path)

    env = _load_env()
    reddit_label = "data/raw/reddit_posts.csv"
    if not env.get("REDDIT_CLIENT_ID", "").strip():
        print(f"  {reddit_label:<42} ⚠️   skipped (no Reddit credentials)")
    else:
        _check("Reddit posts", reddit_path)

    print(_separator("─"))

    # Next-step hints based on what's available
    hints: list[str] = []
    if hr_path.exists():
        hints.append("python -m src.ml.burnout_predictor    # train burnout model")
        hints.append("python -m src.ml.attrition_model      # train attrition model")
    if ge_train_path.exists():
        hints.append(
            "python -c \"from src.nlp.emotion_classifier import fine_tune_emotion_classifier; "
            "fine_tune_emotion_classifier()\"  # fine-tune BERT"
        )
    if enron_path.exists():
        hints.append("python -m src.ingestion.enron_loader  # load Enron into DB")

    if hints:
        print()
        print("  NEXT STEPS — run these to train models:")
        for h in hints:
            print(f"    {h}")
        print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download CogniTeam datasets")
    parser.add_argument(
        "--only",
        choices=["hr", "goemotions", "reddit", "enron"],
        help="Download only one specific dataset",
    )
    parser.add_argument(
        "--skip-reddit",
        action="store_true",
        help="Skip Reddit scraping even if credentials exist",
    )
    parser.add_argument(
        "--posts-per-sub",
        type=int,
        default=500,
        help="Number of Reddit posts to fetch per subreddit (default: 500)",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       CogniTeam — Dataset Downloader v1.0           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Output directory: {DATA_RAW}")
    print()

    only = args.only

    # ── Step 1 ──
    if only in (None, "hr"):
        print("━━ Step 1/4  HR Analytics (Kaggle) ━━")
        download_hr_analytics()
        print()

    # ── Step 2 ──
    if only in (None, "goemotions"):
        print("━━ Step 2/4  GoEmotions (HuggingFace) ━━")
        download_go_emotions()
        print()

    # ── Step 3 ──
    if only in (None, "enron"):
        print("━━ Step 3/4  Enron Email Dataset ━━")
        check_enron()
        print()

    # ── Step 4 ──
    if only in (None, "reddit") and not args.skip_reddit:
        print("━━ Step 4/4  Reddit Workplace Posts (PRAW) ━━")
        download_reddit(posts_per_sub=args.posts_per_sub)
        print()

    # ── Step 5 ──
    print_status_report()


if __name__ == "__main__":
    main()
