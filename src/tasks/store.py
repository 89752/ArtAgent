"""通用任务表。

状态机：pending → processing → done | failed；重启时 processing → interrupted，
失败/中断任务可 reset 后重试。文档解析与表格入库迁入本模型（旧 API 形状不变）。
落库 data/index/tasks.db（INDEX_DIR 可覆盖）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data import db

_DB_PATH = Path(os.getenv("INDEX_DIR", "./data/index")) / "tasks.db"
_lock = threading.Lock()
_db_ready = False

VALID_STATUS = {"pending", "processing", "done", "failed", "interrupted"}


def _get_conn() -> sqlite3.Connection:
    global _db_ready
    conn = db.get_conn(_DB_PATH, row_factory=sqlite3.Row)
    if not _db_ready:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id     TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                payload     TEXT NOT NULL DEFAULT '{}',
                progress    REAL NOT NULL DEFAULT 0,
                error       TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                started_at  TEXT,
                finished_at TEXT
            )
            """
        )
        conn.commit()
        _db_ready = True
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_task(
    type: str,
    payload: Optional[dict] = None,
    task_id: Optional[str] = None,
) -> str:
    """创建任务（默认 pending），返回 task_id。"""
    tid = task_id or f"t_{uuid.uuid4().hex[:12]}"
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT OR IGNORE INTO tasks
                (task_id, type, status, payload, created_at)
            VALUES (?, ?, 'pending', ?, ?)
            """,
            (tid, str(type)[:40], json.dumps(payload or {}, ensure_ascii=False), _now()),
        )
        conn.commit()
    return tid


def get_task(task_id: str) -> Optional[dict]:
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
    if not row:
        return None
    out = dict(row)
    try:
        out["payload"] = json.loads(out.get("payload") or "{}")
    except json.JSONDecodeError:
        out["payload"] = {}
    return out


def list_tasks(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    limit = min(max(1, int(limit)), 500)
    with _lock:
        conn = _get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d["payload"] = json.loads(d.get("payload") or "{}")
        except json.JSONDecodeError:
            d["payload"] = {}
        out.append(d)
    return out


def update_task(task_id: str, **fields) -> None:
    """更新任务字段；status 必须是合法值；自动维护 started/finished 时间。"""
    allowed = {"status", "payload", "progress", "error"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if "status" in updates and updates["status"] not in VALID_STATUS:
        raise ValueError(f"非法任务状态：{updates['status']}")
    if not updates:
        return
    if "payload" in updates and isinstance(updates["payload"], dict):
        updates["payload"] = json.dumps(updates["payload"], ensure_ascii=False)
    now = _now()
    if updates.get("status") == "processing":
        updates["started_at"] = now
    if updates.get("status") in ("done", "failed", "interrupted"):
        updates["finished_at"] = now
    sets = ", ".join(f"{k} = :{k}" for k in updates)
    updates["task_id"] = task_id
    with _lock:
        _get_conn().execute(
            f"UPDATE tasks SET {sets} WHERE task_id = :task_id", updates
        )
        _get_conn().commit()


def reset_task(task_id: str, status: str = "pending") -> bool:
    """失败/中断任务重置为 pending（重试入口）。"""
    if status not in ("pending",):
        raise ValueError("重置目标状态只能是 pending")
    with _lock:
        cur = _get_conn().execute(
            """
            UPDATE tasks SET status = 'pending', error = '', progress = 0,
                             started_at = NULL, finished_at = NULL
            WHERE task_id = ? AND status IN ('failed', 'interrupted')
            """,
            (task_id,),
        )
        _get_conn().commit()
    return cur.rowcount > 0


def mark_interrupted_on_startup() -> int:
    """服务启动时把 processing 任务标记为 interrupted（进程崩溃恢复）。"""
    with _lock:
        cur = _get_conn().execute(
            """
            UPDATE tasks SET status = 'interrupted', error = '服务重启，任务中断',
                             finished_at = ?
            WHERE status = 'processing'
            """,
            (_now(),),
        )
        _get_conn().commit()
    return cur.rowcount


def _reset_for_tests(path: Path | None = None) -> None:
    """测试专用：重置到指定数据库文件。"""
    global _db_ready, _DB_PATH
    db.close_all()
    _db_ready = False
    _DB_PATH = path or Path("./data/index/_test_tasks.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
