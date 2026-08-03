"""收藏清单存储（Phase 5）：collections(user_id, name, items_json, updated_at)。

与 agent_memory 同库文件（data/memory/agent_memory.db），独立表。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "memory"
_DB_PATH = _DB_DIR / "agent_memory.db"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS collections (
                user_id    TEXT NOT NULL,
                name       TEXT NOT NULL,
                items_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, name)
            )
            """
        )
        _conn.commit()
    return _conn


def save_collection(user_id: str, name: str, items: list[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(items, ensure_ascii=False)
    with _lock:
        _get_conn().execute(
            """
            INSERT INTO collections (user_id, name, items_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, name) DO UPDATE SET
                items_json = excluded.items_json, updated_at = excluded.updated_at
            """,
            (user_id, name.strip(), payload, now),
        )
        _get_conn().commit()


def list_collections(user_id: str) -> list[dict]:
    try:
        rows = _get_conn().execute(
            "SELECT name, items_json, updated_at FROM collections "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for name, payload, updated_at in rows:
        try:
            items = json.loads(payload)
        except json.JSONDecodeError:
            items = []
        out.append({"name": name, "items": items, "updated_at": updated_at})
    return out
