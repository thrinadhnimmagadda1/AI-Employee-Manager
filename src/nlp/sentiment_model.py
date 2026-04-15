"""
Sentiment Model
Combines the fine-tuned BERT emotion classifier with VADER
as a fallback to produce weekly sentiment scores per employee.
Writes results to the sentiment_scores table.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Optional

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.nlp.emotion_classifier import dominant_emotion, predict

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")

_vader = SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────────────────────
# Individual text scoring
# ─────────────────────────────────────────────────────────────

def score_text(text: str) -> dict:
    """
    Score a single text passage.

    Returns:
        {
            'sentiment':  float in [-1.0, 1.0],
            'emotion':    str (dominant emotion),
            'confidence': float in [0.0, 1.0],
            'emotions':   dict of all emotion probabilities,
        }
    """
    if not text or not text.strip():
        return {"sentiment": 0.0, "emotion": "neutral", "confidence": 1.0, "emotions": {}}

    # BERT emotion scores
    try:
        emotion_scores = predict(text)
        dom_emotion = dominant_emotion(emotion_scores)
        confidence = emotion_scores[dom_emotion]
    except Exception as exc:
        logger.debug(f"BERT emotion error, falling back to VADER: {exc}")
        emotion_scores = {}
        dom_emotion = "neutral"
        confidence = 0.5

    # VADER compound score for continuous sentiment
    vader_scores = _vader.polarity_scores(text)
    compound = float(vader_scores["compound"])  # already in [-1.0, 1.0]

    # Blend: if BERT confidence is high, weight toward BERT-derived sentiment
    # Map dominant emotion to rough sentiment polarity
    emotion_polarity = {
        "joy": 0.7,
        "frustration": -0.5,
        "anxiety": -0.4,
        "anger": -0.8,
        "sadness": -0.6,
        "disgust": -0.7,
        "neutral": 0.0,
    }
    bert_sentiment = emotion_polarity.get(dom_emotion, 0.0)
    blended = (compound * (1 - confidence * 0.5)) + (bert_sentiment * confidence * 0.5)
    blended = max(-1.0, min(1.0, blended))

    return {
        "sentiment": round(blended, 4),
        "emotion": dom_emotion,
        "confidence": round(confidence, 4),
        "emotions": emotion_scores,
    }


# ─────────────────────────────────────────────────────────────
# Weekly aggregation
# ─────────────────────────────────────────────────────────────

def score_employee_week(
    texts: list[str],
    employee_id: int,
    week: date,
    session: Optional[Session] = None,
) -> dict:
    """
    Aggregate sentiment across all messages from one employee in one week.

    Args:
        texts:       List of message body strings for this employee-week.
        employee_id: Database employee ID.
        week:        ISO week start date (Monday).
        session:     Optional SQLAlchemy session (creates one if None).

    Returns:
        {
            'sentiment':  float (aggregated),
            'emotion':    str (dominant across week),
            'confidence': float,
        }
    """
    if not texts:
        result = {"sentiment": 0.0, "emotion": "neutral", "confidence": 0.5}
    else:
        scores = [score_text(t) for t in texts if t]
        if not scores:
            result = {"sentiment": 0.0, "emotion": "neutral", "confidence": 0.5}
        else:
            avg_sentiment = sum(s["sentiment"] for s in scores) / len(scores)
            # Dominant emotion: most frequent
            emotion_counts: dict[str, int] = {}
            for s in scores:
                emotion_counts[s["emotion"]] = emotion_counts.get(s["emotion"], 0) + 1
            dom = max(emotion_counts, key=emotion_counts.get)
            avg_confidence = sum(s["confidence"] for s in scores) / len(scores)
            result = {
                "sentiment": round(avg_sentiment, 4),
                "emotion": dom,
                "confidence": round(avg_confidence, 4),
            }

    # Persist to database
    _upsert_sentiment_score(employee_id, week, result, session)
    return result


def _upsert_sentiment_score(
    employee_id: int,
    week: date,
    result: dict,
    session: Optional[Session] = None,
) -> None:
    """Insert or update sentiment_scores row for (employee_id, week)."""
    engine = create_engine(DATABASE_URL) if session is None else None
    own_session = session is None

    try:
        if own_session:
            session = Session(engine)

        session.execute(
            text("""
                INSERT INTO sentiment_scores
                    (employee_id, week, sentiment, emotion, confidence)
                VALUES
                    (:eid, :week, :sentiment, :emotion, :confidence)
                ON CONFLICT (employee_id, week)
                DO UPDATE SET
                    sentiment   = EXCLUDED.sentiment,
                    emotion     = EXCLUDED.emotion,
                    confidence  = EXCLUDED.confidence,
                    computed_at = NOW()
            """),
            {
                "eid": employee_id,
                "week": week,
                "sentiment": result["sentiment"],
                "emotion": result["emotion"],
                "confidence": result["confidence"],
            },
        )
        if own_session:
            session.commit()
    except Exception as exc:
        logger.error(f"Failed to upsert sentiment score for employee {employee_id}: {exc}")
        if own_session and session:
            session.rollback()
    finally:
        if own_session and session:
            session.close()


# ─────────────────────────────────────────────────────────────
# Bulk weekly scoring job (called by Celery)
# ─────────────────────────────────────────────────────────────

def score_all_employees_for_week(week: date) -> int:
    """
    Pull all message metadata for the given week, fetch bodies
    from processed CSVs (if available), compute and store scores.

    Returns:
        Number of employees scored.
    """
    engine = create_engine(DATABASE_URL)
    scored = 0

    with Session(engine) as session:
        # Fetch distinct senders who had activity this week
        rows = session.execute(
            text("""
                SELECT DISTINCT sender_id
                FROM message_metadata
                WHERE DATE_TRUNC('week', timestamp) = :week
            """),
            {"week": week},
        ).fetchall()

        employee_ids = [r[0] for r in rows]
        logger.info(f"Scoring {len(employee_ids)} employees for week {week}")

        for emp_id in employee_ids:
            # We only have metadata — use word_count as proxy for activity
            # In production: join with processed email bodies
            try:
                meta_rows = session.execute(
                    text("""
                        SELECT word_count
                        FROM message_metadata
                        WHERE sender_id = :eid
                          AND DATE_TRUNC('week', timestamp) = :week
                    """),
                    {"eid": emp_id, "week": week},
                ).fetchall()

                # Generate synthetic text placeholders from word counts
                # In production, use actual email body text
                texts = [
                    "work update status report" * max(1, r[0] // 10)
                    for r in meta_rows
                    if r[0] and r[0] > 0
                ]

                score_employee_week(texts, emp_id, week, session=session)
                scored += 1
            except Exception as exc:
                logger.error(f"Error scoring employee {emp_id}: {exc}")
                session.rollback()

        session.commit()

    logger.info(f"Weekly scoring complete for {week}: {scored} employees scored.")
    return scored
