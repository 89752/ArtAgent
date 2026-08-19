"""结构化用户文档（对齐 DeerMem 的 user document）。

存储：agent_memory.db 的 user_docs 表，每用户一份 JSON 文档。

结构：
  personalContext  语言 / 沟通偏好 / 稳定兴趣（1-2 句）
  topOfMind        当前正在关注 / 进行中的事项（2-4 句，可随时间替换）
  recent           最近 1-3 个月的活动
  earlier          3-12 个月前的背景
  longTerm         长期稳定的背景

更新：由后台 worker 每次对话后调用 update_user_doc（LLM 增量合并，
shouldUpdate 门控），不再用"条目堆积→重压缩"的画像刷新。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.data import db
from src.utils.json_utils import parse_json
from src.utils.logging_config import get_logger

logger = get_logger("memory.user_doc")

_DB_DIR = Path(os.getenv(
    "ARTAGENT_MEMORY_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "memory"),
))
_DB_PATH = _DB_DIR / "agent_memory.db"

_lock = threading.RLock()
_db_ready = False

_SECTIONS = ("personalContext", "topOfMind", "recent", "earlier", "longTerm")


def _get_conn() -> sqlite3.Connection:
    global _db_ready
    conn = db.get_conn(_DB_PATH, row_factory=sqlite3.Row)
    if not _db_ready:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_docs (
                user_id    TEXT PRIMARY KEY,
                doc_json   TEXT NOT NULL,
                revision   INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(user_docs)").fetchall()}
        if "revision" not in cols:
            conn.execute(
                "ALTER TABLE user_docs ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
            )
        conn.commit()
        _db_ready = True
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_doc() -> dict:
    """返回空文档骨架（所有 section 均带 summary/updatedAt）。"""
    ts = _now()
    return {
        "version": 1,
        **{
            section: {"summary": "", "updatedAt": ts}
            for section in _SECTIONS
        },
    }


def load_doc_with_revision(user_id: str) -> tuple[dict, int]:
    """读取用户文档及其 revision（乐观并发控制用）。"""
    if not user_id:
        return empty_doc(), 0
    try:
        row = _get_conn().execute(
            "SELECT doc_json, revision FROM user_docs WHERE user_id = ?", (user_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return empty_doc(), 0
    if not row:
        return empty_doc(), 0
    try:
        raw = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return empty_doc(), int(row[1] or 0)
    base = empty_doc()
    for section in _SECTIONS:
        src = raw.get(section)
        if isinstance(src, dict):
            base[section]["summary"] = str(src.get("summary") or "")[:2000]
            base[section]["updatedAt"] = str(src.get("updatedAt") or base[section]["updatedAt"])
    return base, int(row[1] or 0)


def load_doc(user_id: str) -> dict:
    """读取用户文档；不存在/损坏时返回空骨架（安全降级）。"""
    doc, _rev = load_doc_with_revision(user_id)
    return doc


def save_doc(
    user_id: str,
    doc: dict,
    expected_revision: Optional[int] = None,
) -> bool:
    """整份写回用户文档（乐观并发：expected_revision 不匹配时拒绝覆盖）。

    返回 True 表示写入成功；False 表示版本冲突（调用方应重读后重试）。
    expected_revision=None 时保持旧语义（无条件 upsert）。
    """
    if not user_id:
        return False
    with _lock:
        conn = _get_conn()
        payload = json.dumps(doc, ensure_ascii=False)
        if expected_revision is None:
            conn.execute(
                """
                INSERT INTO user_docs (user_id, doc_json, revision, updated_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    doc_json = excluded.doc_json,
                    revision = user_docs.revision + 1,
                    updated_at = excluded.updated_at
                """,
                (user_id, payload, _now()),
            )
        else:
            exists = conn.execute(
                "SELECT 1 FROM user_docs WHERE user_id = ?", (user_id,)
            ).fetchone()
            if not exists and expected_revision == 0:
                conn.execute(
                    """
                    INSERT INTO user_docs (user_id, doc_json, revision, updated_at)
                    VALUES (?, ?, 1, ?)
                    """,
                    (user_id, payload, _now()),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE user_docs
                    SET doc_json = ?, revision = revision + 1, updated_at = ?
                    WHERE user_id = ? AND revision = ?
                    """,
                    (payload, _now(), user_id, expected_revision),
                )
                if cur.rowcount == 0:
                    conn.commit()
                    return False
        _get_conn().commit()
    return True


def delete_user_doc(user_id: str) -> int:
    """删除某用户的结构化文档；返回删除条数（级联删除用）。"""
    if not user_id:
        return 0
    with _lock:
        cur = _get_conn().execute(
            "DELETE FROM user_docs WHERE user_id = ?", (user_id,)
        )
        _get_conn().commit()
        return cur.rowcount


def _default_llm() -> Callable[[str], str]:
    from src.utils.llm import get_deterministic_llm

    def _invoke(p: str) -> str:
        return get_deterministic_llm().invoke(p).content

    return _invoke


MEMORY_UPDATE_PROMPT = """你是用户记忆文档更新模块。根据【当前文档】与【新增对话】，
增量更新用户的长期记忆文档。

只输出 JSON（不要解释、不要 markdown）：
{{
  "personalContext": {{"shouldUpdate": true, "summary": "用户主要使用中文交流，偏好简洁回答，喜欢莫奈等印象派画家。"}},
  "topOfMind": {{"shouldUpdate": false, "summary": ""}},
  "recent": {{"shouldUpdate": true, "summary": "用户最近在研究西方艺术史，重点对比莫奈、梵高、毕加索的构图。"}},
  "earlier": {{"shouldUpdate": false, "summary": ""}},
  "longTerm": {{"shouldUpdate": false, "summary": ""}}
}}

规则：
- 只有新增对话里出现"值得长期保留的用户信息"时，对应 section 才 shouldUpdate=true；
- personalContext：语言、沟通偏好、称呼、稳定兴趣（1-2 句）；
- topOfMind：当前正在关注/进行中的事项（2-4 句），过时事项可被替换；
- recent：最近 1-3 个月的活动；earlier：3-12 个月前；longTerm：长期稳定背景；
- 拿不准就 shouldUpdate=false；不编造；保留原有内容中仍然成立的部分；
- summary 用第三人称中文陈述句。

【当前文档】
{current_doc}

【新增对话】
{conversation}"""


def update_user_doc(
    user_id: str,
    messages,
    llm: Optional[Callable[[str], str]] = None,
) -> dict:
    """增量更新结构化用户文档（LLM 合并，shouldUpdate 门控）。

    返回 {"updated": [section...], "doc": doc}；
    无更新返回 {"skipped": "no_update"}；失败返回 {"error": ...}（不改旧文档）。
    """
    from src.memory.extract import recent_conversation_text

    conversation = recent_conversation_text(messages)
    if not conversation or conversation == "(无)":
        return {"skipped": "no_conversation"}
    current, revision = load_doc_with_revision(user_id)
    prompt = MEMORY_UPDATE_PROMPT.format(
        current_doc=json.dumps(current, ensure_ascii=False),
        conversation=conversation,
    )
    try:
        if llm is None:
            llm = _default_llm()
        data = parse_json(llm(prompt))
    except Exception as e:  # noqa: BLE001 —— 文档更新失败不影响主流程
        logger.warning("[user_doc] update failed: %s", e)
        return {"error": str(e)[:200]}
    if not isinstance(data, dict):
        return {"error": "bad_shape"}

    updated: list[str] = []
    for section in _SECTIONS:
        patch = data.get(section)
        if not isinstance(patch, dict):
            continue
        if patch.get("shouldUpdate") is not True:
            continue
        summary = str(patch.get("summary") or "").strip()
        if not summary:
            continue
        current[section]["summary"] = summary[:2000]
        current[section]["updatedAt"] = _now()
        updated.append(section)

    if not updated:
        return {"skipped": "no_update", "doc": current}
    if not save_doc(user_id, current, expected_revision=revision):
        logger.warning("[user_doc] 写入冲突，放弃本次更新 user=%s", user_id)
        return {"conflict": True, "doc": current}
    logger.info("[user_doc] updated sections=%s user=%s", updated, user_id)
    return {"updated": updated, "doc": current}


def _reset_for_tests(path: Optional[Path] = None) -> None:
    """测试专用：重置到指定数据库文件。"""
    global _db_ready, _DB_PATH
    db.close_all()
    _db_ready = False
    _DB_PATH = path or (Path(__file__).resolve().parent.parent.parent
                        / "data" / "index" / "_test_user_doc.db")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _DB_PATH.unlink(missing_ok=True)
    except OSError:
        pass
