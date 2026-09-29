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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data import db

_DB_PATH = Path(os.getenv("INDEX_DIR", "./data/index")) / "tasks.db"
_lock = threading.RLock()
_db_ready = False

VALID_STATUS = {"pending", "processing", "paused", "done", "failed", "interrupted", "waiting_input", "verification_failed", "budget_exhausted", "unknown_execution_state"}


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
        _ensure_column(conn, "tasks", "lease_owner", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "tasks", "lease_until", "REAL NOT NULL DEFAULT 0")
        _ensure_column(conn, "tasks", "attempt_id", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "tasks", "usage_json", "TEXT NOT NULL DEFAULT '{}'")
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


def create_agent_job(objective: str, user_id: str, plan: Optional[list[str]] = None, *, spec: Optional[dict] = None) -> str:
    """Create a durable, user-scoped multi-step Agent job."""
    steps = [{"title": str(step)[:300], "status": "pending"} for step in (plan or [])]
    tid = f"t_{uuid.uuid4().hex[:12]}"
    payload = {"objective": str(objective)[:4000], "user_id": user_id, "spec": spec or {}}
    with _lock:
        _get_conn().execute(
            "INSERT INTO tasks (task_id,type,status,payload,created_at,plan_json,steps_json) VALUES (?,'agent_job','pending',?,?,?,?)",
            (tid, json.dumps(payload, ensure_ascii=False), _now(), json.dumps(plan or [], ensure_ascii=False), json.dumps(steps, ensure_ascii=False)),
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
    for key, default in (("plan_json", []), ("steps_json", []), ("artifacts_json", []), ("usage_json", {})):
        try:
            out[key.removesuffix("_json")] = json.loads(out.pop(key) or "[]")
        except (json.JSONDecodeError, TypeError):
            out[key.removesuffix("_json")] = default
    out["cancel_requested"] = bool(out.get("cancel_requested"))
    out["pause_requested"] = bool(out.get("pause_requested"))
    return out


def _advance_agent_job(task_id: str, *, artifact: Optional[dict] = None, error: str = "") -> bool:
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


def list_tasks(status: Optional[str] = None, limit: int = 100, *, user_id: str | None = None) -> list[dict]:
    limit = min(max(1, int(limit)), 500)
    with _lock:
        conn = _get_conn()
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if user_id is not None:
            clauses.append("json_extract(payload, '$.user_id') = ?")
            params.append(user_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = conn.execute("SELECT task_id FROM tasks" + where + " ORDER BY created_at DESC LIMIT ?", (*params, limit)).fetchall()
    out = []
    for row in rows:
        out.append(get_task(row["task_id"]))
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
                WHERE task_id = ? AND status IN ('failed', 'interrupted', 'verification_failed', 'waiting_input')
                """,
                (json.dumps(steps, ensure_ascii=False), task_id),
            )
        else:
            cur = conn.execute(
                """
                UPDATE tasks SET status = 'pending', error = '', progress = 0,
                                 started_at = NULL, finished_at = NULL, usage_json='{}',
                                 lease_owner='',lease_until=0,cancel_requested=0,pause_requested=0
                WHERE task_id = ? AND status IN ('failed', 'interrupted', 'budget_exhausted')
                """,
                (task_id,),
            )
        conn.commit()
    return cur.rowcount > 0


def mark_interrupted_on_startup() -> int:
    """仅中断租约已过期的运行任务，保留有效执行者。

    待解析任务由持久队列重新领取；已开始但失联的解析任务需显式重试，
    避免自动重复解析副作用。研究任务另由恢复调度处理。
    """
    with _lock:
        cur = _get_conn().execute(
            """
            UPDATE tasks SET status = 'interrupted', error = '服务重启，任务中断',
                             finished_at = ?
            WHERE (status='processing' OR (status='pending' AND type NOT IN ('ingest_pdf','ingest_table'))) AND lease_until < ?
            """,
            (_now(), time.time()),
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


def runnable_agent_jobs(limit: int = 16) -> list[dict]:
    """Poll durable work, including workers whose leases expired after startup."""
    with _lock:
        conn = _get_conn()
        conn.execute("""UPDATE tasks SET status=CASE WHEN cancel_requested=1 THEN 'interrupted' ELSE 'paused' END,
            lease_owner='',lease_until=0 WHERE type='agent_job' AND status IN ('pending','processing')
            AND lease_until < ? AND (cancel_requested=1 OR pause_requested=1)""", (time.time(),))
        conn.commit()
        rows = _get_conn().execute(
            "SELECT task_id,payload FROM tasks WHERE type='agent_job' "
            "AND status IN ('pending','processing') AND lease_until < ? "
            "AND cancel_requested=0 AND pause_requested=0 ORDER BY created_at LIMIT ?",
            (time.time(), limit),
        ).fetchall()
    return [{"task_id": row["task_id"], "user_id": json.loads(row["payload"]).get("user_id", "")} for row in rows]


def claim_job(task_id: str, user_id: str, owner: str, ttl: int = 90) -> str | None:
    """Compare-and-set claim; only the holder may commit a step."""
    return claim_task(task_id, user_id, owner, ttl, task_types=("agent_job",))


def claim_task(task_id: str, user_id: str, owner: str, ttl: int = 90, *, task_types=("agent_job", "ingest_pdf", "ingest_table")) -> str | None:
    attempt = uuid.uuid4().hex
    with _lock:
        conn = _get_conn()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT type FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row and row["type"] in {"ingest_pdf","ingest_table"}:
            active = conn.execute("SELECT COUNT(*) FROM tasks WHERE type IN ('ingest_pdf','ingest_table') AND status='processing' AND lease_until>?", (time.time(),)).fetchone()[0]
            if active >= max(1,int(os.getenv("TASK_PARSE_CONCURRENCY","2"))):
                conn.rollback()
                return None
        placeholders = ",".join("?" for _ in task_types)
        cur = conn.execute(f"""UPDATE tasks SET status='processing',lease_owner=?,lease_until=?,attempt_id=?
            WHERE task_id=? AND type IN ({placeholders}) AND json_extract(payload,'$.user_id')=?
            AND status IN ('pending','processing') AND lease_until < ? AND cancel_requested=0 AND pause_requested=0""",
            (owner, time.time()+ttl, attempt, task_id, *task_types, user_id, time.time()))
        conn.commit()
        return attempt if cur.rowcount else None


def runnable_ingestion_tasks(limit=2):
    """Pending uploads survive a restart. Expired parsing needs explicit retry."""
    with _lock:
        conn = _get_conn()
        conn.execute("UPDATE tasks SET status='interrupted',error='用户取消' WHERE type IN ('ingest_pdf','ingest_table') AND status='pending' AND cancel_requested=1")
        conn.execute("""UPDATE tasks SET status='interrupted',error='解析进程失联，请重试',lease_owner='',lease_until=0
            WHERE type IN ('ingest_pdf','ingest_table') AND status='processing' AND lease_until<?""", (time.time(),))
        rows = conn.execute("SELECT task_id,payload FROM tasks WHERE type IN ('ingest_pdf','ingest_table') AND status='pending' AND cancel_requested=0 ORDER BY created_at LIMIT ?", (max(0,limit),)).fetchall()
        conn.commit()
    return [{"task_id":r["task_id"], "payload":json.loads(r["payload"])} for r in rows]


def cancel_ingestion(doc_id, user_id):
    with _lock:
        conn = _get_conn()
        conn.execute("""UPDATE tasks SET cancel_requested=1,
            status=CASE WHEN status='pending' THEN 'interrupted' ELSE status END
            WHERE type IN ('ingest_pdf','ingest_table') AND json_extract(payload,'$.user_id')=?
            AND (task_id=? OR json_extract(payload,'$.doc_id')=?) AND status IN ('pending','processing')""", (user_id,doc_id,doc_id))
        conn.commit()


def heartbeat(task_id: str, attempt: str, usage: dict | None = None) -> bool:
    with _lock:
        conn = _get_conn()
        cur = conn.execute("""UPDATE tasks SET lease_until=?,usage_json=COALESCE(?,usage_json)
            WHERE task_id=? AND attempt_id=? AND status='processing' AND lease_until>?""",
            (time.time()+90, json.dumps(usage) if usage is not None else None, task_id, attempt, time.time()))
        conn.commit()
        return cur.rowcount > 0


def finish_attempt(task_id: str, attempt: str, status: str, error: str = "") -> bool:
    if status not in VALID_STATUS:
        raise ValueError(status)
    with _lock:
        conn = _get_conn()
        cur = conn.execute("""UPDATE tasks SET status=?,error=?,lease_until=0,lease_owner='',finished_at=?,progress=CASE WHEN ?='done' THEN 100 ELSE progress END
            WHERE task_id=? AND attempt_id=? AND lease_until>?""",
            (status, error[:1000], _now(), status, task_id, attempt, time.time()))
        conn.commit()
        return cur.rowcount > 0


def advance_agent_job(task_id: str, *, artifact: Optional[dict] = None, error: str = "", attempt: str = "") -> bool:
    with _lock:
        conn = _get_conn()
        # A SQLite write reservation also fences concurrent API processes.
        conn.execute("BEGIN IMMEDIATE")
        try:
            job = get_task(task_id)
            if not job:
                conn.rollback()
                return False
            if job.get("lease_owner") and (not attempt or job["attempt_id"] != attempt or job["lease_until"] <= time.time()):
                conn.rollback()
                return False
            if not error and not job.get("cancel_requested") and not (artifact or {}).get("content", "").strip():
                conn.rollback()
                return False
            advanced = _advance_agent_job(task_id, artifact=artifact, error=error)
            conn.commit()
            return advanced
        except BaseException:
            conn.rollback()
            raise


def revise_job(task_id: str, user_id: str, text: str) -> bool:
    with _lock:
        conn = _get_conn()
        conn.execute("BEGIN IMMEDIATE")
        job = get_task(task_id)
        if not job or job["payload"].get("user_id") != user_id or job["status"] not in {"done", "verification_failed", "budget_exhausted"}:
            conn.rollback()
            return False
        payload = job["payload"]
        payload["additional_input"] = text[:8000]
        payload["revision"] = int(payload.get("revision",0))+1
        conn.execute("""UPDATE tasks SET payload=?,status='pending',step_index=?,usage_json='{}',
            error='',lease_until=0,lease_owner='',cancel_requested=0,pause_requested=0 WHERE task_id=?""",
            (json.dumps(payload,ensure_ascii=False),max(0,len(job["plan"])-1),task_id))
        conn.commit()
        return True


def supply_input(task_id: str, user_id: str, text: str) -> bool:
    with _lock:
        job = get_task(task_id)
        if not job or job["payload"].get("user_id") != user_id or job["status"] not in {"waiting_input", "verification_failed", "failed"}:
            return False
        payload = job["payload"]
        payload["additional_input"] = str(text)[:8000]
        update_task(task_id, payload=payload)
        return reset_task(task_id)


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
