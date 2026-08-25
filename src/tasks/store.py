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

VALID_STATUS = {"pending", "processing", "paused", "done", "failed", "interrupted"}


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
        _ensure_column(conn, "tasks", "plan_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "tasks", "steps_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "tasks", "step_index", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "tasks", "artifacts_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "tasks", "cancel_requested", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "tasks", "pause_requested", "INTEGER NOT NULL DEFAULT 0")
        conn.commit()
        _db_ready = True
    return conn


def _ensure_column(conn, table: str, column: str, definition: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


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


def create_agent_job(objective: str, user_id: str, plan: Optional[list[str]] = None) -> str:
    """Create a durable, user-scoped multi-step Agent job."""
    steps = [{"title": str(step)[:300], "status": "pending"} for step in (plan or [])]
    tid = create_task("agent_job", {"objective": str(objective)[:4000], "user_id": user_id})
    with _lock:
        _get_conn().execute(
            "UPDATE tasks SET plan_json = ?, steps_json = ? WHERE task_id = ?",
            (json.dumps(plan or [], ensure_ascii=False), json.dumps(steps, ensure_ascii=False), tid),
        )
        _get_conn().commit()
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
    for key, default in (("plan_json", []), ("steps_json", []), ("artifacts_json", [])):
        try:
            out[key.removesuffix("_json")] = json.loads(out.pop(key) or "[]")
        except (json.JSONDecodeError, TypeError):
            out[key.removesuffix("_json")] = default
    out["cancel_requested"] = bool(out.get("cancel_requested"))
    out["pause_requested"] = bool(out.get("pause_requested"))
    return out


def advance_agent_job(task_id: str, *, artifact: Optional[dict] = None, error: str = "") -> bool:
    """Atomically complete the current step and checkpoint durable job state."""
    job = get_task(task_id)
    if not job or job.get("type") != "agent_job" or job.get("status") not in {"pending", "processing"}:
        return False
    if job.get("cancel_requested"):
        update_task(task_id, status="interrupted", error="用户取消")
        return False
    steps = list(job.get("steps") or [])
    index = int(job.get("step_index") or 0)
    if index < len(steps):
        steps[index]["status"] = "failed" if error else "done"
        if error:
            steps[index]["error"] = error[:300]
    artifacts = list(job.get("artifacts") or [])
    if artifact:
        artifacts.append(artifact)
    # A failed step must remain the current step.  Retrying an AgentJob then
    # resumes precisely where it stopped instead of silently skipping work.
    next_index = index if error else index + 1
    status = "failed" if error else ("done" if next_index >= len(steps) else "processing")
    with _lock:
        _get_conn().execute(
            """UPDATE tasks SET steps_json = ?, artifacts_json = ?, step_index = ?, status = ?,
               error = ?, finished_at = CASE WHEN ? IN ('done','failed') THEN ? ELSE finished_at END
               WHERE task_id = ?""",
            (json.dumps(steps, ensure_ascii=False), json.dumps(artifacts, ensure_ascii=False),
             next_index, status, error[:300], status, _now(), task_id),
        )
        _get_conn().commit()
    return True


def cancel_agent_job(task_id: str) -> bool:
    with _lock:
        cur = _get_conn().execute(
            "UPDATE tasks SET cancel_requested = 1 WHERE task_id = ? AND type = 'agent_job' AND status IN ('pending','processing')",
            (task_id,),
        )
        _get_conn().commit()
    return cur.rowcount > 0


def pause_agent_job(task_id: str) -> bool:
    """Request a safe pause between steps; pending jobs pause immediately."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            """UPDATE tasks SET pause_requested = 1,
               status = CASE WHEN status = 'pending' THEN 'paused' ELSE status END
               WHERE task_id = ? AND type = 'agent_job' AND status IN ('pending','processing')""",
            (task_id,),
        )
        conn.commit()
    return cur.rowcount > 0


def resume_agent_job(task_id: str) -> bool:
    """Resume a paused job at its current durable step index."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            """UPDATE tasks SET status = 'pending', pause_requested = 0, error = '',
               started_at = NULL, finished_at = NULL
               WHERE task_id = ? AND type = 'agent_job' AND status = 'paused'""",
            (task_id,),
        )
        conn.commit()
    return cur.rowcount > 0


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
        conn = _get_conn()
        row = conn.execute(
            "SELECT type, steps_json, step_index FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row and row["type"] == "agent_job":
            try:
                steps = json.loads(row["steps_json"] or "[]")
            except json.JSONDecodeError:
                steps = []
            index = int(row["step_index"] or 0)
            if index < len(steps) and steps[index].get("status") == "failed":
                steps[index].pop("error", None)
                steps[index]["status"] = "pending"
            cur = conn.execute(
                """
                UPDATE tasks SET status = 'pending', error = '', progress = 0,
                    cancel_requested = 0, pause_requested = 0, steps_json = ?, started_at = NULL, finished_at = NULL
                WHERE task_id = ? AND status IN ('failed', 'interrupted')
                """,
                (json.dumps(steps, ensure_ascii=False), task_id),
            )
        else:
            cur = conn.execute(
                """
                UPDATE tasks SET status = 'pending', error = '', progress = 0,
                                 started_at = NULL, finished_at = NULL
                WHERE task_id = ? AND status IN ('failed', 'interrupted')
                """,
                (task_id,),
            )
        conn.commit()
    return cur.rowcount > 0


def mark_interrupted_on_startup() -> int:
    """服务启动时标记未被恢复调度的持久任务为 interrupted。

    进程可能在 BackgroundTask 尚未切到 ``processing`` 时重启；这类
    ``pending`` 导入任务已经不在内存队列中，若保留 pending 会永远无法由
    UI 重试。统一改为 interrupted 后，文件任务可由用户重试，AgentJob 则由
    ``recover_interrupted_agent_jobs`` 自动恢复。
    """
    with _lock:
        cur = _get_conn().execute(
            """
            UPDATE tasks SET status = 'interrupted', error = '服务重启，任务中断',
                             finished_at = ?
            WHERE status IN ('pending', 'processing')
            """,
            (_now(),),
        )
        _get_conn().commit()
    return cur.rowcount


def recover_interrupted_agent_jobs() -> list[dict]:
    """Return restart-interrupted AgentJobs to the durable queue for auto-resume."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT task_id, payload FROM tasks WHERE type = 'agent_job' AND status = 'interrupted' "
            "AND cancel_requested = 0 AND pause_requested = 0"
        ).fetchall()
        task_ids = [str(row["task_id"]) for row in rows]
        if task_ids:
            conn.executemany(
                "UPDATE tasks SET status = 'pending', error = '服务恢复，继续执行' WHERE task_id = ?",
                [(task_id,) for task_id in task_ids],
            )
            conn.commit()
    recovered = []
    for row in rows:
        try:
            payload = json.loads(row["payload"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        recovered.append({"task_id": str(row["task_id"]), "user_id": str(payload.get("user_id") or "")})
    return recovered


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
