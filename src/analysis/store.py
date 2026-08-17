"""用户图片与分析报告的持久化（独立 SQLite，不污染 documents_store）。"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger("analysis.store")

DB_PATH = Path(os.getenv("INDEX_DIR", "./data/index")) / "user_images.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS user_images (
    image_id      TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL DEFAULT 'web_user',
    session_id    TEXT,
    original_name TEXT,
    file_path     TEXT,
    file_size     INTEGER,
    mime_type     TEXT,
    width         INTEGER,
    height        INTEGER,
    created_at    TEXT,
    status        TEXT,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS painting_analysis_results (
    image_id     TEXT PRIMARY KEY REFERENCES user_images(image_id),
    framework    TEXT,
    result_path  TEXT,
    created_at   TEXT,
    updated_at   TEXT,
    metadata     TEXT
);
"""


def _ensure_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表（幂等），服务启动时调用。"""
    with _connect() as conn:
        conn.executescript(_CREATE_SQL)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(user_images)").fetchall()}
        if "user_id" not in cols:
            conn.execute(
                "ALTER TABLE user_images ADD COLUMN user_id TEXT NOT NULL DEFAULT 'web_user'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_images_user "
            "ON user_images(user_id, session_id)"
        )
        conn.commit()


def _row(row) -> dict | None:
    return dict(row) if row is not None else None


def add_image(
    image_id: str,
    session_id: str,
    original_name: str,
    file_path: str,
    file_size: int,
    mime_type: str,
    width: int,
    height: int,
    status: str = "uploaded",
    user_id: str = "web_user",
) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO user_images
              (image_id, user_id, session_id, original_name, file_path, file_size,
               mime_type, width, height, created_at, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id, user_id, session_id, original_name, file_path, file_size,
                mime_type, width, height, now, status, "",
            ),
        )
        conn.commit()


def get_image(image_id: str, user_id: str | None = None) -> dict | None:
    """按 image_id 读取；user_id 提供时校验归属（None=内部读取不校验）。"""
    with _connect() as conn:
        if user_id:
            cur = conn.execute(
                "SELECT * FROM user_images WHERE image_id = ? AND user_id = ?",
                (image_id, user_id),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM user_images WHERE image_id = ?", (image_id,)
            )
        return _row(cur.fetchone())


def list_images_by_session(
    session_id: str,
    user_id: str | None = None,
) -> list[dict]:
    with _connect() as conn:
        if user_id:
            cur = conn.execute(
                """
                SELECT * FROM user_images
                WHERE session_id = ? AND user_id = ? AND status != 'deleted'
                ORDER BY created_at DESC
                """,
                (session_id, user_id),
            )
        else:
            cur = conn.execute(
                """
                SELECT * FROM user_images
                WHERE session_id = ? AND status != 'deleted'
                ORDER BY created_at DESC
                """,
                (session_id,),
            )
        return [_row(r) for r in cur.fetchall()]


def update_image_status(image_id: str, status: str, error: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE user_images SET status = ?, error = ? WHERE image_id = ?",
            (status, error, image_id),
        )
        conn.commit()


def delete_image(image_id: str, user_id: str | None = None) -> bool:
    """删除数据库记录（含关联分析结果）；文件清理由调用方负责。"""
    if get_image(image_id, user_id) is None:
        return False
    with _connect() as conn:
        conn.execute(
            "DELETE FROM painting_analysis_results WHERE image_id = ?", (image_id,)
        )
        if user_id:
            cur = conn.execute(
                "DELETE FROM user_images WHERE image_id = ? AND user_id = ?",
                (image_id, user_id),
            )
        else:
            cur = conn.execute("DELETE FROM user_images WHERE image_id = ?", (image_id,))
        conn.commit()
    return cur.rowcount > 0


def save_analysis(
    image_id: str,
    framework: str,
    result_path: str,
    metadata: dict | None = None,
) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    import json

    meta = json.dumps(metadata or {}, ensure_ascii=False)
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO painting_analysis_results
              (image_id, framework, result_path, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (image_id, framework, result_path, now, now, meta),
        )
        conn.commit()


def get_analysis(image_id: str) -> dict | None:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT * FROM painting_analysis_results WHERE image_id = ?",
            (image_id,),
        )
        row = _row(cur.fetchone())
    if row and row.get("metadata"):
        try:
            import json

            row["metadata"] = json.loads(row["metadata"])
        except Exception:  # noqa: BLE001
            row["metadata"] = {}
    return row


def list_analysis_by_session(
    session_id: str,
    user_id: str | None = None,
) -> list[dict]:
    with _connect() as conn:
        sql = """
            SELECT r.image_id, r.framework, r.result_path,
                   r.created_at, r.updated_at, r.metadata
            FROM painting_analysis_results r
            JOIN user_images u ON u.image_id = r.image_id
            WHERE u.session_id = ?
        """
        params: list = [session_id]
        if user_id:
            sql += " AND u.user_id = ?"
            params.append(user_id)
        sql += " ORDER BY r.updated_at DESC"
        cur = conn.execute(sql, params)
        rows = [_row(r) for r in cur.fetchall()]
    for row in rows:
        if row.get("metadata"):
            try:
                import json

                row["metadata"] = json.loads(row["metadata"])
            except Exception:  # noqa: BLE001
                row["metadata"] = {}
    return rows


def cleanup_expired(ttl_days: int = 30) -> list[str]:
    """删除超过 TTL 的图片记录（文件由调用方清理）。返回被删 image_id。"""
    import datetime

    cutoff = (
        datetime.datetime.now() - datetime.timedelta(days=ttl_days)
    ).strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        cur = conn.execute(
            "SELECT image_id FROM user_images WHERE created_at < ?", (cutoff,)
        )
        ids = [r["image_id"] for r in cur.fetchall()]
        for iid in ids:
            conn.execute(
                "DELETE FROM painting_analysis_results WHERE image_id = ?", (iid,)
            )
            conn.execute("DELETE FROM user_images WHERE image_id = ?", (iid,))
        conn.commit()
    return ids
