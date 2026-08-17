"""记忆生命周期维护（对齐 DeerMem 的 staleness review）。

每条事实可带 expected_valid_days；后台按频率（默认 24h/用户）审查到期条目，
LLM 决定 REMOVE（软删除）或 EXTEND（延长有效期），失败时保守延长。

存储：agent_memory.db 的 memory_meta 表（记录上次维护时间）。
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.memory.memory_items import list_memories
from src.utils.json_utils import parse_json
from src.utils.logging_config import get_logger

logger = get_logger("memory.lifecycle")

_DB_DIR = Path(os.getenv(
    "ARTAGENT_MEMORY_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "memory"),
))
_DB_PATH = _DB_DIR / "agent_memory.db"

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_meta (
                user_id             TEXT PRIMARY KEY,
                last_maintenance_at TEXT NOT NULL
            )
            """
        )
        _conn.commit()
    return _conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_since(iso: str) -> float:
    try:
        dt = datetime.fromisoformat(iso)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return 0.0


def maintenance_interval_hours() -> float:
    """两次维护的最小间隔（MEMORY_MAINTENANCE_INTERVAL_HOURS，默认 24）。"""
    try:
        return max(1.0, float(os.getenv("MEMORY_MAINTENANCE_INTERVAL_HOURS", "24")))
    except (TypeError, ValueError):
        return 24.0


def _last_maintenance(user_id: str) -> Optional[str]:
    row = _get_conn().execute(
        "SELECT last_maintenance_at FROM memory_meta WHERE user_id = ?", (user_id,)
    ).fetchone()
    return row[0] if row else None


def _set_last_maintenance(user_id: str) -> None:
    with _lock:
        _get_conn().execute(
            """
            INSERT INTO memory_meta (user_id, last_maintenance_at)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET last_maintenance_at = excluded.last_maintenance_at
            """,
            (user_id, _now()),
        )
        _get_conn().commit()


def delete_user_meta(user_id: str) -> int:
    """删除某用户的维护元数据（级联删除用）。"""
    if not user_id:
        return 0
    with _lock:
        cur = _get_conn().execute(
            "DELETE FROM memory_meta WHERE user_id = ?", (user_id,)
        )
        _get_conn().commit()
        return cur.rowcount


def staleness_due(user_id: str, limit: int = 20) -> list[dict]:
    """返回已过有效期的活跃条目（expected_valid_days 非空且已超期）。"""
    items = [
        i for i in list_memories(user_id, scope="user")
        if i.get("kind") != "profile" and i.get("expected_valid_days")
    ]
    due = []
    for item in items:
        try:
            days = int(item.get("expected_valid_days") or 0)
        except (TypeError, ValueError):
            continue
        age = _days_since(str(item.get("updated_at") or ""))
        if days > 0 and age >= days:
            due.append(item)
        if len(due) >= limit:
            break
    return due


STALENESS_PROMPT = """你是记忆生命周期审查模块。以下记忆条目已经超过预期有效期，
请决定是删除还是延长。

条目：
{items}

只输出 JSON（不要解释、不要 markdown）：
{{"decisions": [
  {{"id": "mem_xxx", "action": "remove", "reason": "已过时"}},
  {{"id": "mem_yyy", "action": "extend", "extend_by_days": 90, "reason": "仍然成立"}}
]}}

规则：
- remove：内容已不成立/过时/与用户当前状态矛盾；
- extend：内容仍然成立但需要续期，extend_by_days 给新的有效期；
- 每条都要有 id 和 action；不确定时选 extend（保守）。"""


CONSOLIDATION_PROMPT = """你是记忆合并模块。以下多条记忆属于同一实体（entity），
内容重复或高度相似，请合并成一条精炼、完整、第三人称中文陈述。

条目：
{items}

只输出 JSON（不要解释、不要 markdown）：
{{"consolidated": "合并后的完整记忆内容"}}

规则：
- 保留所有仍然成立的细节，去掉重复；
- 不编造，不引入条目里没有的信息；
- 如果内容互相矛盾，保留更可信/更新的表述。"""


def _default_llm() -> Callable[[str], str]:
    from src.utils.llm import get_deterministic_llm

    def _invoke(p: str) -> str:
        return get_deterministic_llm().invoke(p).content

    return _invoke


def consolidate_similar(
    user_id: str,
    llm: Optional[Callable[[str], str]] = None,
) -> dict:
    """把同 entity 的 3 条以上相似记忆合并为一条（LLM 合成）。

    合并结果以 source='consolidated' 写入，原条目全部 supersede 到新条目；
    LLM 失败或畸形输出跳过该组（保守不合并）。
    """
    from src.memory.memory_items import add_memory

    items = [
        i for i in list_memories(user_id, scope="user")
        if i.get("kind") in ("preference", "fact") and i.get("entity")
    ]
    groups: dict[str, list[dict]] = {}
    for item in items:
        key = str(item.get("entity") or "").strip().lower()
        groups.setdefault(key, []).append(item)
    candidates = [g for g in groups.values() if len(g) >= 2]
    if not candidates:
        return {"skipped": "no_groups"}

    merged = 0
    conn = _get_conn()
    for group in candidates[:5]:
        lines = "\n".join(
            f"- {i['id']}: {str(i.get('content') or '')[:200]}"
            f" (importance={i.get('importance')})"
            for i in group
        )
        try:
            if llm is None:
                llm = _default_llm()
            data = parse_json(llm(CONSOLIDATION_PROMPT.format(items=lines)))
        except Exception:  # noqa: BLE001 —— 合并失败跳过该组
            data = None
        content = str((data or {}).get("consolidated") or "").strip()
        if not content:
            continue
        try:
            importance = max(float(i.get("importance") or 0.5) for i in group)
        except (TypeError, ValueError):
            importance = 0.7
        valid_days = [
            int(i.get("expected_valid_days") or 0)
            for i in group
            if i.get("expected_valid_days")
        ]
        new_item = add_memory(
            user_id=user_id,
            content=content,
            kind=group[0]["kind"],
            entity=group[0]["entity"],
            scope="user",
            source="consolidated",
            importance=importance,
            expected_valid_days=max(valid_days) if valid_days else None,
        )
        new_id = new_item["id"]
        with _lock:
            for old in group:
                if old["id"] != new_id:
                    conn.execute(
                        "UPDATE memory_items SET superseded_by = ? WHERE id = ?",
                        (new_id, old["id"]),
                    )
            conn.commit()
        merged += 1
    return {"merged": merged, "groups": len(candidates)}


def review_staleness(
    user_id: str,
    llm: Optional[Callable[[str], str]] = None,
) -> dict:
    """审查到期条目：LLM 决定 remove/extend；失败或畸形输出保守全部延长。"""
    due = staleness_due(user_id)
    if not due:
        return {"skipped": "no_due"}
    lines = "\n".join(
        f"- id={i['id']} content={str(i.get('content') or '')[:120]} "
        f"valid_days={i.get('expected_valid_days')}"
        for i in due
    )
    prompt = STALENESS_PROMPT.format(items=lines)
    try:
        if llm is None:
            llm = _default_llm()
        data = parse_json(llm(prompt))
    except Exception as e:  # noqa: BLE001 —— 失败保守延长
        logger.warning("[lifecycle] staleness review failed: %s", e)
        data = None

    decisions: dict[str, dict] = {}
    if isinstance(data, dict):
        for d in data.get("decisions") or []:
            if isinstance(d, dict) and d.get("id"):
                decisions[str(d["id"])] = d

    removed = 0
    extended = 0
    conn = _get_conn()
    now = _now()
    with _lock:
        for item in due:
            decision = decisions.get(item["id"]) or {}
            action = str(decision.get("action") or "extend").strip().lower()
            if action == "remove":
                conn.execute(
                    "UPDATE memory_items SET deleted_at = ?, last_reviewed_at = ? WHERE id = ?",
                    (now, now, item["id"]),
                )
                removed += 1
                continue
            try:
                extend_by = max(1, min(3650, int(decision.get("extend_by_days") or 30)))
            except (TypeError, ValueError):
                extend_by = 30
            old_valid = int(item.get("expected_valid_days") or 30)
            conn.execute(
                """
                UPDATE memory_items
                SET expected_valid_days = ?, last_reviewed_at = ?
                WHERE id = ?
                """,
                (min(3650, old_valid + extend_by), now, item["id"]),
            )
            extended += 1
        conn.commit()
    return {"removed": removed, "extended": extended, "reviewed": len(due)}


def maybe_maintenance(
    user_id: str,
    llm: Optional[Callable[[str], str]] = None,
) -> dict:
    """按频率触发维护：未到间隔直接跳过；否则跑失效审查并记录时间。"""
    last = _last_maintenance(user_id)
    if last and _days_since(last) * 24 < maintenance_interval_hours():
        return {"skipped": "fresh"}
    result = review_staleness(user_id, llm=llm)
    result["consolidation"] = consolidate_similar(user_id, llm=llm)
    _set_last_maintenance(user_id)
    return result


def _reset_for_tests(path: Optional[Path] = None) -> None:
    """测试专用：重置到指定数据库文件。"""
    global _conn, _DB_PATH
    _conn = None
    _DB_PATH = path or (Path(__file__).resolve().parent.parent.parent
                        / "data" / "index" / "_test_lifecycle.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
