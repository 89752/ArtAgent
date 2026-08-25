"""Privacy-conscious Agent run/trajectory observability store."""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data import db

_DB_PATH = Path(os.getenv("INDEX_DIR", "./data/index")) / "observability.db"
_lock = threading.Lock()
_db_ready = False

_SENSITIVE_KEYS = {"authorization", "cookie", "password", "token", "api_key", "key", "secret"}


def redact(value):
    """Remove credentials before data crosses the trace persistence boundary."""
    if isinstance(value, dict):
        return {
            str(k): "[REDACTED]" if str(k).lower() in _SENSITIVE_KEYS else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return value[:500]
    return value


def usage_from_message(message, role: str = "") -> dict:
    """Normalize provider usage metadata from LangChain AI messages."""
    raw = getattr(message, "usage_metadata", None) or {}
    response = getattr(message, "response_metadata", None) or {}
    usage = response.get("usage") or response.get("token_usage") or {}
    raw = raw if isinstance(raw, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    inp = raw.get("input_tokens", usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    out = raw.get("output_tokens", usage.get("completion_tokens", usage.get("output_tokens", 0)))
    total = raw.get("total_tokens", usage.get("total_tokens", 0))
    try:
        inp, out, total = int(inp or 0), int(out or 0), int(total or 0)
    except (TypeError, ValueError):
        inp = out = total = 0
    return {
        "model": str(response.get("model_name") or response.get("model") or "")[:128],
        "role": str(role or "main")[:32],
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total or inp + out,
        "token_source": "provider" if (inp or out or total) else "unavailable",
        "finish_reason": str(response.get("finish_reason") or "")[:64],
    }


def _get_conn() -> sqlite3.Connection:
    global _db_ready
    conn = db.get_conn(_DB_PATH, row_factory=sqlite3.Row)
    if not _db_ready:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id         TEXT NOT NULL DEFAULT '',
                user_id            TEXT NOT NULL DEFAULT '',
                session_id         TEXT NOT NULL DEFAULT '',
                intent             TEXT NOT NULL DEFAULT '',
                steps_json         TEXT NOT NULL DEFAULT '[]',
                tools_json         TEXT NOT NULL DEFAULT '[]',
                context_chars      INTEGER NOT NULL DEFAULT 0,
                tool_rounds        INTEGER NOT NULL DEFAULT 0,
                latency_ms         REAL NOT NULL DEFAULT 0,
                est_cost           REAL NOT NULL DEFAULT 0,
                input_tokens       INTEGER NOT NULL DEFAULT 0,
                output_tokens      INTEGER NOT NULL DEFAULT 0,
                token_source       TEXT NOT NULL DEFAULT 'estimated',
                final_answer_len   INTEGER NOT NULL DEFAULT 0,
                reflection_triggered INTEGER NOT NULL DEFAULT 0,
                web_fallback       INTEGER NOT NULL DEFAULT 0,
                cancelled          INTEGER NOT NULL DEFAULT 0,
                error              TEXT NOT NULL DEFAULT '',
                created_at         TEXT NOT NULL
            )
            """
        )
        _ensure_column(conn, "agent_runs", "user_id", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "agent_runs", "input_tokens", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "agent_runs", "output_tokens", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "agent_runs", "token_source", "TEXT NOT NULL DEFAULT 'estimated'")
        conn.execute("""CREATE TABLE IF NOT EXISTS node_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
            node_name TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
            latency_ms REAL NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'ok',
            state_keys_json TEXT NOT NULL DEFAULT '[]')""")
        conn.execute("""CREATE TABLE IF NOT EXISTS model_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
            model TEXT NOT NULL DEFAULT '', role TEXT NOT NULL DEFAULT 'main', input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0,
            token_source TEXT NOT NULL DEFAULT 'unavailable', latency_ms REAL NOT NULL DEFAULT 0,
            finish_reason TEXT NOT NULL DEFAULT '')""")
        _ensure_column(conn, "model_calls", "role", "TEXT NOT NULL DEFAULT 'main'")
        conn.execute("""CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
            tool_name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ok', latency_ms REAL NOT NULL DEFAULT 0,
            result_size INTEGER NOT NULL DEFAULT 0, error_type TEXT NOT NULL DEFAULT '',
            args_json TEXT NOT NULL DEFAULT '{}')""")
        conn.commit()
        _db_ready = True
    return conn


def _ensure_column(conn, table: str, column: str, definition: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input_rate() -> float:
    return float(os.getenv("COST_PER_1K_INPUT_TOKENS", "0") or 0)


def _output_rate() -> float:
    return float(os.getenv("COST_PER_1K_OUTPUT_TOKENS", "0") or 0)


def estimate_cost(context_chars: int, answer_len: int, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """按 4 字符 ≈ 1 token 估算成本（单价为 0 时返回 0）。"""
    inp = input_tokens or context_chars / 4
    out = output_tokens or answer_len / 4
    return (inp / 1000) * _input_rate() + (out / 1000) * _output_rate()


def record_run(
    *,
    request_id: str = "",
    user_id: str = "",
    session_id: str = "",
    intent: str = "",
    steps: Optional[list] = None,
    tools: Optional[list] = None,
    context_chars: int = 0,
    tool_rounds: int = 0,
    latency_ms: float = 0.0,
    final_answer_len: int = 0,
    reflection_triggered: bool = False,
    web_fallback: bool = False,
    cancelled: bool = False,
    error: str = "",
    node_events: Optional[list[dict]] = None,
    model_calls: Optional[list[dict]] = None,
    tool_calls: Optional[list[dict]] = None,
) -> int:
    """写入一条轨迹；返回自增 id。"""
    model_calls = model_calls or []
    tool_calls = tool_calls or []
    input_tokens = sum(int(c.get("input_tokens") or 0) for c in model_calls)
    output_tokens = sum(int(c.get("output_tokens") or 0) for c in model_calls)
    provider_usage = any(c.get("token_source") == "provider" for c in model_calls)
    source = "provider" if provider_usage else "estimated"
    est = estimate_cost(int(context_chars or 0), int(final_answer_len or 0), input_tokens, output_tokens)
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            """
            INSERT INTO agent_runs (
                request_id, user_id, session_id, intent, steps_json, tools_json,
                context_chars, tool_rounds, latency_ms, est_cost,
                input_tokens, output_tokens, token_source,
                final_answer_len, reflection_triggered, web_fallback,
                cancelled, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(request_id or "")[:64],
                str(user_id or "")[:128],
                str(session_id or "")[:128],
                str(intent or "")[:32],
                json.dumps(steps or [], ensure_ascii=False),
                json.dumps(tools or [], ensure_ascii=False),
                int(context_chars or 0),
                int(tool_rounds or 0),
                round(float(latency_ms or 0), 1),
                round(est, 6),
                input_tokens,
                output_tokens,
                source,
                int(final_answer_len or 0),
                1 if reflection_triggered else 0,
                1 if web_fallback else 0,
                1 if cancelled else 0,
                str(error or "")[:300],
                _now(),
            ),
        )
        run_id = int(cur.lastrowid)
        for event in node_events or []:
            conn.execute("""INSERT INTO node_events
                (run_id,node_name,started_at,ended_at,latency_ms,status,state_keys_json)
                VALUES (?,?,?,?,?,?,?)""", (
                run_id, str(event.get("node_name") or "")[:80],
                str(event.get("started_at") or _now()), str(event.get("ended_at") or _now()),
                float(event.get("latency_ms") or 0), str(event.get("status") or "ok")[:32],
                json.dumps(event.get("state_keys") or [], ensure_ascii=False),
            ))
        for call in model_calls:
            conn.execute("""INSERT INTO model_calls
                (run_id,model,role,input_tokens,output_tokens,total_tokens,token_source,latency_ms,finish_reason)
                VALUES (?,?,?,?,?,?,?,?,?)""", (
                run_id, str(call.get("model") or "")[:128], str(call.get("role") or "main")[:32], int(call.get("input_tokens") or 0),
                int(call.get("output_tokens") or 0), int(call.get("total_tokens") or 0),
                str(call.get("token_source") or "unavailable")[:32], float(call.get("latency_ms") or 0),
                str(call.get("finish_reason") or "")[:64],
            ))
        for call in tool_calls:
            conn.execute("""INSERT INTO tool_calls
                (run_id,tool_name,status,latency_ms,result_size,error_type,args_json)
                VALUES (?,?,?,?,?,?,?)""", (
                run_id, str(call.get("tool_name") or "")[:80], str(call.get("status") or "ok")[:48],
                float(call.get("latency_ms") or 0), int(call.get("result_size") or 0),
                str(call.get("error_type") or "")[:80],
                json.dumps(redact(call.get("args") or {}), ensure_ascii=False),
            ))
        _cleanup_retention(conn)
        conn.commit()
    return run_id


def _cleanup_retention(conn) -> None:
    days = max(1, int(os.getenv("TRACE_RETENTION_DAYS", "30") or 30))
    conn.execute("DELETE FROM agent_runs WHERE created_at < datetime('now', ?)", (f"-{days} days",))


def list_runs(limit: int = 50, user_id: str = "") -> list[dict]:
    limit = min(max(1, int(limit)), 500)
    with _lock:
        sql = "SELECT * FROM agent_runs"
        params: list = []
        if user_id:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        rows = _get_conn().execute(sql + " ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d["steps"] = json.loads(d.pop("steps_json") or "[]")
        except json.JSONDecodeError:
            d["steps"] = []
        try:
            d["tools"] = json.loads(d.pop("tools_json") or "[]")
        except json.JSONDecodeError:
            d["tools"] = []
        out.append(d)
    return out


def get_run_detail(run_id: int, user_id: str) -> Optional[dict]:
    """Return a complete trace only to its owning user."""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM agent_runs WHERE id = ? AND user_id = ?", (int(run_id), user_id)
        ).fetchone()
        if not row:
            return None
        out = dict(row)
        out["node_events"] = [
            {**dict(r), "state_keys": json.loads(r["state_keys_json"] or "[]")}
            for r in conn.execute("SELECT * FROM node_events WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        ]
        out["model_calls"] = [dict(r) for r in conn.execute(
            "SELECT * FROM model_calls WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()]
        out["tool_calls"] = [
            {**dict(r), "args": json.loads(r["args_json"] or "{}")}
            for r in conn.execute("SELECT * FROM tool_calls WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        ]
    return out


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def metrics(limit: int = 500, user_id: str = "") -> dict:
    """最近 N 条轨迹汇总：请求量/延迟/成本/工具分布/反思与兜底率/错误率。"""
    runs = list_runs(limit=limit, user_id=user_id)
    if not runs:
        return {"count": 0}
    latencies = [r["latency_ms"] for r in runs]
    tool_counter: dict[str, int] = {}
    role_counter: dict[str, int] = {}
    for r in runs:
        for t in r.get("tools") or []:
            tool_counter[str(t)] = tool_counter.get(str(t), 0) + 1
        # The endpoint is already owner-scoped when user_id is provided.  Keep
        # aggregate metrics cheap and privacy-safe by querying only this run.
        detail = get_run_detail(int(r["id"]), user_id) if user_id else None
        for call in (detail or {}).get("model_calls", []):
            role = str(call.get("role") or "main")
            role_counter[role] = role_counter.get(role, 0) + 1
    return {
        "count": len(runs),
        "window": f"last {len(runs)} runs",
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 1),
            "p50": round(_pct(latencies, 0.5), 1),
            "p95": round(_pct(latencies, 0.95), 1),
        },
        "est_cost_total": round(sum(r["est_cost"] for r in runs), 6),
        "tokens": {
            "input_total": sum(r["input_tokens"] for r in runs),
            "output_total": sum(r["output_tokens"] for r in runs),
            "average_total": round(statistics.mean(r["input_tokens"] + r["output_tokens"] for r in runs), 1),
            "provider_usage_rate": round(sum(r["token_source"] == "provider" for r in runs) / len(runs), 4),
        },
        "tool_calls": tool_counter,
        "model_roles": role_counter,
        "reflection_rate": round(
            sum(r["reflection_triggered"] for r in runs) / len(runs), 4
        ),
        "web_fallback_rate": round(
            sum(r["web_fallback"] for r in runs) / len(runs), 4
        ),
        "error_rate": round(sum(1 for r in runs if r["error"]) / len(runs), 4),
        "cancelled": sum(r["cancelled"] for r in runs),
    }


def _reset_for_tests(path: Path | None = None) -> None:
    """测试专用：重置到指定数据库文件。"""
    global _db_ready, _DB_PATH
    db.close_all()
    _db_ready = False
    _DB_PATH = path or Path("./data/index/_test_observability.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
