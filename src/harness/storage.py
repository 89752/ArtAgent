"""User-scoped artifacts and durable tool receipts. Full outputs stay off prompts."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection():
    path = Path(os.getenv("ARTAGENT_HARNESS_DB_PATH", str(Path(os.getenv("INDEX_DIR", "./data/index")) / "harness.db")))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, task_id TEXT NOT NULL,
            step_id TEXT NOT NULL, revision INTEGER NOT NULL, kind TEXT NOT NULL,
            content TEXT NOT NULL, content_hash TEXT NOT NULL, metadata TEXT NOT NULL,
            created_at TEXT NOT NULL, UNIQUE(user_id,task_id,step_id,revision));
        CREATE INDEX IF NOT EXISTS artifact_owner ON artifacts(user_id,task_id);
        CREATE TABLE IF NOT EXISTS tool_receipts (
            key TEXT PRIMARY KEY, user_id TEXT NOT NULL, task_id TEXT NOT NULL,
            status TEXT NOT NULL, result TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL);
    """)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def put_artifact(user_id: str, task_id: str, step_id: str, content: str, *, kind="report", metadata=None) -> dict:
    digest = hashlib.sha256(content.encode()).hexdigest()
    with connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute("SELECT * FROM artifacts WHERE user_id=? AND task_id=? AND step_id=? ORDER BY revision DESC LIMIT 1",
                                (user_id, task_id, step_id)).fetchone()
        if (previous and previous["content_hash"] == digest and previous["kind"] == kind
                and json.loads(previous["metadata"]) == (metadata or {})):
            return decode(previous)
        revision = previous["revision"] + 1 if previous else 1
        aid = "a_" + uuid.uuid4().hex
        conn.execute("INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?)", (
            aid, user_id, task_id, step_id, revision, kind, content, digest,
            json.dumps(metadata or {}, ensure_ascii=False), now()))
        return decode(conn.execute("SELECT * FROM artifacts WHERE id=?", (aid,)).fetchone())


def decode(row):
    if row is None:
        return None
    item = dict(row)
    item["metadata"] = json.loads(item["metadata"])
    return item


def get_artifact(aid, user_id):
    with connection() as conn:
        return decode(conn.execute("SELECT * FROM artifacts WHERE id=? AND user_id=?", (aid, user_id)).fetchone())


def list_artifacts(task_id, user_id):
    with connection() as conn:
        return [decode(row) for row in conn.execute(
            "SELECT * FROM artifacts WHERE task_id=? AND user_id=? ORDER BY created_at,revision", (task_id, user_id))]


def start_tool(key, user_id, task_id):
    """An existing in-flight receipt is UNKNOWN, never permission to replay a write."""
    with connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM tool_receipts WHERE key=? AND user_id=?", (key, user_id)).fetchone()
        if row:
            return dict(row)
        conn.execute("INSERT INTO tool_receipts VALUES (?,?,?,'running','',?)", (key, user_id, task_id, now()))
    return None


def finish_tool(key, result, status="done"):
    with connection() as conn:
        conn.execute("UPDATE tool_receipts SET status=?,result=?,updated_at=? WHERE key=?", (status, result, now(), key))


def delete_user(user_id):
    with connection() as conn:
        for table in ("artifacts", "tool_receipts"):
            conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
