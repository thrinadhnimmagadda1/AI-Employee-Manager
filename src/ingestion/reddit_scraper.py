"""
Reddit Workplace Posts Scraper
Pulls posts from workplace-related subreddits using PRAW
and saves to data/raw/reddit_posts.csv.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import praw
from loguru import logger

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "CogniTeam/1.0")

DATA_RAW = Path(os.getenv("DATA_DIR", "data")) / "raw"
SUBREDDITS = ["WorkReform", "antiwork", "cscareerquestions"]
POSTS_PER_SUBREDDIT = 1000


def _build_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


def _scrape_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    limit: int = POSTS_PER_SUBREDDIT,
) -> list[dict]:
    """
    Scrape up to `limit` posts from a subreddit.
    Handles rate limiting with exponential backoff.
    """
    records: list[dict] = []
    subreddit = reddit.subreddit(subreddit_name)
    logger.info(f"Scraping r/{subreddit_name} (up to {limit} posts) …")

    backoff = 2
    attempts = 0

    try:
        for post in subreddit.hot(limit=limit):
            try:
                text_body = (post.selftext or "").strip()
                title = (post.title or "").strip()
                combined = f"{title}. {text_body}".strip()
                if not combined:
                    continue

                records.append(
                    {
                        "subreddit": subreddit_name,
                        "post_id": post.id,
                        "title": title,
                        "body": text_body,
                        "text": combined,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": post.created_utc,
                        "url": post.url,
                    }
                )
                backoff = 2  # reset on success
                attempts = 0

            except praw.exceptions.APIException as api_exc:
                logger.warning(f"API exception on post: {api_exc}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                attempts += 1
                if attempts > 5:
                    logger.error("Too many API errors, stopping this subreddit.")
                    break

    except praw.exceptions.PRAWException as exc:
        logger.error(f"PRAW error for r/{subreddit_name}: {exc}")

    logger.info(f"r/{subreddit_name}: collected {len(records)} posts.")
    return records


def scrape_reddit_posts(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Scrape all configured subreddits and save to CSV.

    Returns:
        DataFrame of all scraped posts.
    """
    output_path = output_path or DATA_RAW / "reddit_posts.csv"
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.warning(
            "Reddit API credentials not set. "
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env. "
            "Returning empty DataFrame."
        )
        return pd.DataFrame()

    reddit = _build_reddit_client()
    all_records: list[dict] = []

    for sub in SUBREDDITS:
        records = _scrape_subreddit(reddit, sub)
        all_records.extend(records)
        time.sleep(1)  # polite delay between subreddits

    if not all_records:
        logger.warning("No posts collected from any subreddit.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records).drop_duplicates(subset=["post_id"])
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} Reddit posts to {output_path}")
    return df


if __name__ == "__main__":
    scrape_reddit_posts()
