"""共享节点：记忆读写、信息澄清、反思、长期记忆写入。

温和版：意图路由管线已删除，这里只保留 ReAct 需要的收尾节点。
"""

from langchain_core.messages import AIMessage

from src.agent.prompts import REFLECTION_PROMPT
from src.agent.state import AgentState
from src.utils.json_utils import parse_json  # noqa: F401 —— 保持导出供既有调用方使用
from src.utils.llm import get_deterministic_llm
from src.utils.logging_config import get_logger, log_event

logger = get_logger("nodes")


def collect_artworks(docs_by_group: dict[str, list[dict]], limit: int = 8) -> list[dict]:
    """把分组检索结果扁平化、去重，供 UI 卡片展示。

    只收画作（SemArt 结果无 source 键）；用户 PDF 片段带 source 键，
    不进配图卡片（没有 SemArt 本地图可配），但仍保留在 retrieved_docs
    里作为 LLM 证据。
    """
    seen = set()
    flat = []
    for group in docs_by_group.values():
        for d in group:
            if d.get("source"):  # 用户文档片段 → 跳过配图卡片
                continue
            title = d.get("title", "")
            if not title or title in seen:
                continue
            seen.add(title)
            flat.append(
                {
                    "title": title,
                    "author": d.get("author", ""),
                    "date": d.get("date", ""),
                    "image_file": d.get("image_file", ""),
                }
            )
            if len(flat) >= limit:
                return flat
    return flat


# ── 信息缺口澄清 ───────────────────────────────────────────────
_STYLE_SIGNALS = (
    "喜欢", "偏爱", "风格", "色彩", "笔触", "画家", "作品",
    "类似", "像", "主题", "氛围", "光影", "色调", "构图",
    # 常见审美/风格词（避免"浓烈奔放"这类被误判为信息不足）
    "浓烈", "奔放", "宁静", "优雅", "华丽", "简约", "古典", "现代",
    "抽象", "写实", "印象", "巴洛克", "洛可可", "浪漫", "深沉", "明亮",
    "柔和", "风景", "静物", "肖像", "宗教", "神话",
)


def _info_gap(question: str, intent: str) -> tuple[bool, str]:
    """判定是否存在"信息不足"（仅明确缺口才追问，一般歧义放行）。"""
    q = (question or "").strip()
    if len(q) < 6:
        return True, "能再具体说说想了解什么吗？例如某位画家、某幅画或某种艺术风格。"
    if intent == "recommendation" and not any(s in q for s in _STYLE_SIGNALS):
        return True, "你更偏好哪种风格？或者有没有喜欢的画家/作品作为参考？"
    return False, ""


def ask_user(state: AgentState) -> dict:
    """信息缺口澄清节点：不足则追问并短路，否则放行继续主流程。"""
    if state.rewrite_ambiguous:
        gap, message = True, (
            "你的意思我不太确定，能说得更具体一点吗？"
            "例如想了解哪位画家、哪幅画或哪种风格。"
        )
    else:
        # 用改写前的原始问题判断信息缺口：改写可能压缩掉疑问词（如"莫奈晚年"），
        # 长度启发式不应作用在内部压缩句上（mt-002 回归）
        raw_question = state.original_user_query or state.user_query or ""
        gap, message = _info_gap(raw_question, state.intent)
    if not gap:
        return {
            "ask_user": "continue",
            "pending_clarification": "",
            "current_step": "ask_user",
        }
    log_event(logger, "ask_user", query=state.user_query, question=message)
    return {
        "ask_user": "ask",
        "pending_clarification": message,
        "final_answer": message,
        "messages": [AIMessage(content=message)],
        "current_step": "ask_user→澄清",
    }


# ── 长期记忆读取 ──────────────────────────────────────────────────
def load_memory(state: AgentState) -> dict:
    """检索注入相关记忆 + 读取会话滚动摘要与用户画像。"""
    from src.agent.context import build_memory_block
    from src.memory.memory_items import list_memories, search_memories
    from src.memory.memory_items import get_memory_user_id
    from src.memory.profile import load_profile_item
    from src.memory.summary import load_summary, load_summary_item

    uid = state.user_id or get_memory_user_id()
    pref_items = [
        i for i in list_memories(uid, scope="user")
        if i.get("kind") == "preference"
    ]
    pref_contents = [str(i.get("content") or "") for i in pref_items]
    prefs = {
        "preferences": pref_contents,
        "artists": pref_contents,
        "styles": [],
    }
    summary = load_summary(state.conversation_id)
    items = search_memories(
        uid,
        state.user_query or "",
        scope="user",
        top_k=5,
    )
    if len(items) < 3:
        fallback = search_memories(
            uid, "", scope="user", top_k=3, min_score=0.0,
        )
        have = {i["id"] for i in items}
        items = items + [i for i in fallback if i["id"] not in have]
    # 注入保底：语言/沟通偏好与用户纠正永远排在前面，不被检索结果挤掉
    guaranteed = [
        i for i in list_memories(uid, scope="user")
        if (
            i.get("kind") == "correction"
            or (
                i.get("kind") == "preference"
                and (
                    str(i.get("entity") or "") in ("语言", "回复风格", "称呼")
                    or "交流" in str(i.get("content") or "")
                    or "回复" in str(i.get("content") or "")
                )
            )
        )
    ]
    guaranteed.sort(key=lambda i: -float(i.get("importance") or 0))
    have = {i["id"] for i in guaranteed}
    items = guaranteed[:4] + [i for i in items if i["id"] not in have]
    memory_block = build_memory_block(items, budget=600)
    from src.memory.user_doc import load_doc

    profile_item = load_profile_item(uid)
    doc = load_doc(uid)
    doc_parts = []
    pc = str((doc.get("personalContext") or {}).get("summary") or "").strip()
    tm = str((doc.get("topOfMind") or {}).get("summary") or "").strip()
    rc = str((doc.get("recent") or {}).get("summary") or "").strip()
    if pc:
        doc_parts.append(f"【用户画像】{pc[:200]}")
    if tm:
        doc_parts.append(f"【当前关注】{tm[:150]}")
    if rc:
        doc_parts.append(f"【近期】{rc[:150]}")
    doc_block = "\n\n".join(doc_parts)
    if not doc_block:
        if profile_item and (profile_item.get("content") or "").strip():
            doc_block = f"【用户画像】{str(profile_item['content']).strip()[:200]}"
    if doc_block:
        memory_block = (
            doc_block + ("\n\n" + memory_block if memory_block else "")
        )
    summary_item = (
        load_summary_item(state.conversation_id, uid)
        if state.conversation_id else None
    )
    if summary_item and (summary_item.get("summary") or "").strip():
        ep_text = (
            f"【上次对话回顾 · {summary_item.get('turn_count', 0)} 轮】"
            f"{str(summary_item['summary']).strip()[:150]}"
        )
        memory_block = (memory_block + "\n\n" + ep_text).strip()
    log_event(
        logger, "load_memory",
        user=uid,
        artists=prefs.get("artists", []),
        styles=prefs.get("styles", []),
        memory_hits=len(items),
        memory_block_chars=len(memory_block),
        summary_len=len(summary),
        episode=bool(summary_item),
        profile=bool(profile_item),
    )
    return {
        "user_preferences": prefs,
        "conversation_summary": summary,
        "memory_items": items,
        "memory_block": memory_block,
        "current_step": "load_memory",
    }


# ── 反思 ────────────────────────────────────────────────────────
def reflection(state: AgentState) -> dict:
    """检查最终回答是否充分；不充分则标记 RETRY（最多重试一轮）。"""
    answer = state.final_answer
    if not answer and state.messages:
        last = state.messages[-1]
        answer = getattr(last, "content", "") or ""

    if state.retry_count >= 1:
        log_event(logger, "reflection", verdict="PASS", note="retry_exhausted")
        return {"reflection_notes": "PASS", "final_answer": answer, "current_step": "reflection"}

    prompt = REFLECTION_PROMPT.format(user_query=state.user_query, final_answer=answer)
    verdict = get_deterministic_llm().invoke(prompt).content.strip().upper()
    notes = "RETRY" if "RETRY" in verdict else "PASS"
    updates: dict = {
        "reflection_notes": notes,
        "final_answer": answer,
        "current_step": "reflection",
    }
    if notes == "RETRY":
        updates["retry_count"] = state.retry_count + 1
    log_event(logger, "reflection", verdict=notes, answer_len=len(answer or ""))
    return updates


# ── 长期记忆写入 ──────────────────────────────────────────────────
def save_memory(state: AgentState) -> dict:
    """触发会话滚动摘要 + 用户画像刷新 + 自动抽取（开关控制）。"""
    from src.memory.summary import maybe_summarize
    from src.memory.memory_items import get_memory_user_id
    from src.memory.profile import maybe_refresh_profile

    uid = state.user_id or get_memory_user_id()
    summary = maybe_summarize(
        state.messages, state.conversation_id, uid,
        volume_chars=state.context_chars,
    )
    profile_result: dict = {}
    try:
        profile_result = maybe_refresh_profile(uid)
    except Exception as e:  # noqa: BLE001 —— 画像聚合失败不阻塞主流程
        profile_result = {"error": str(e)}
    extracted_turns = state.memory_extracted_turns
    extract_result: dict = {}
    try:
        from src.memory.extract import schedule_extract

        extract_result = schedule_extract(state.messages, uid)
    except Exception as e:  # noqa: BLE001 —— 抽取失败不阻塞主流程
        extract_result = {"error": str(e)}
    log_event(
        logger, "save_memory",
        user=state.user_id,
        turns=sum(1 for m in state.messages if getattr(m, "type", "") == "human"),
        tool_rounds=state.tool_rounds,
        context_chars=state.context_chars,
        summary_len=len(summary),
        auto_extract=bool(extract_result),
        auto_profile=bool(profile_result),
    )
    return {
        "current_step": "save_memory",
        "conversation_summary": summary,
        "memory_extracted_turns": extracted_turns,
        "memory_extract_result": extract_result,
        "memory_profile_result": profile_result,
    }
