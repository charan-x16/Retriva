"""SQLite logging helpers for Retriva evaluation results."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("EVAL_DB_PATH", "backend/evaluation/eval_log.db"))


def init_db() -> None:
    """Create the query evaluation log table if it does not exist."""

    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                contexts TEXT NOT NULL,
                grade_score REAL,
                faithfulness REAL,
                answer_relevancy REAL,
                context_precision REAL,
                context_recall REAL
            )
            """
        )
        conn.commit()


def log_query(question, answer, contexts, grade_score) -> int:
    """Insert one query/answer row and return its new row id."""

    timestamp = datetime.now(timezone.utc).isoformat()
    contexts_json = json.dumps(contexts, ensure_ascii=False)

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO query_log (
                timestamp,
                question,
                answer,
                contexts,
                grade_score
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, question, answer, contexts_json, grade_score),
        )
        conn.commit()
        return int(cursor.lastrowid)


def update_ragas_scores(row_id, scores: dict) -> None:
    """Update Ragas metric scores for an existing query log row."""

    with _connect() as conn:
        conn.execute(
            """
            UPDATE query_log
            SET faithfulness = ?,
                answer_relevancy = ?,
                context_precision = ?,
                context_recall = ?
            WHERE id = ?
            """,
            (
                scores.get("faithfulness"),
                scores.get("answer_relevancy"),
                scores.get("context_precision"),
                scores.get("context_recall"),
                row_id,
            ),
        )
        conn.commit()


def get_all_logs() -> list[dict]:
    """Return all evaluation log rows as dictionaries."""

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id,
                   timestamp,
                   question,
                   answer,
                   contexts,
                   grade_score,
                   faithfulness,
                   answer_relevancy,
                   context_precision,
                   context_recall
            FROM query_log
            ORDER BY id DESC
            """
        ).fetchall()

    logs = []
    for row in rows:
        item = dict(row)
        try:
            item["contexts"] = json.loads(item["contexts"])
        except json.JSONDecodeError:
            item["contexts"] = []
        logs.append(item)
    return logs


def _connect():
    """Open a SQLite connection configured for dict-like rows."""

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=MEMORY")
    return conn
