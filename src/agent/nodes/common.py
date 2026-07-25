"""
共享节点：意图路由、记忆读写、反思、web 兜底，以及工具函数。
"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from langchain_core.messages import HumanMessage

from src.agent.state import AgentState
from src.agent.prompts import (
    CONTEXTUALIZE_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    REFLECTION_PROMPT,
    WEB_FALLBACK_SYNTHESIZE_PROMPT,
)
from src.utils.llm import get_llm, get_deterministic_llm
from src.utils.logging_config import get_logger, log_event
from src.memory.store import load_preferences, upsert_preference

logger = get_logger("nodes")


# ── 工具函数 ────────────────────────────────────────────────────
def parse_json(text: str) -> Any:
    """从 LLM 输出中鲁棒地解析 JSON（去 markdown 代码块、截取首个 {} 或 []）。"""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n")
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # 兜底：截取第一个完整的对象或数组
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_c)
        end = cleaned.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                continue
    return None


def collect_artworks(docs_by_group: dict[str, list[dict]], limit: int = 8) -> list[dict]:
    """把分组检索结果扁平化、去重，供 UI 卡片展示。"""
    seen = set()
    flat = []
    for group in docs_by_group.values():
        for d in group:
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


# ── 多轮指代消解 ────────────────────────────────────────────────
def contextualize(state: AgentState) -> dict:
    """
    用对话历史把带指代的追问改写成独立问题，写回 user_query。
    下游所有节点都读 user_query，所以在此一处消解即可全局生效。
    首轮（无历史）直接跳过，不额外调用 LLM。
    """
    # 收集本轮之前的人类/助手消息（当前 HumanMessage 是最后一条）
    prior = [
        m for m in state.messages[:-1]
        if isinstance(m, (HumanMessage, AIMessage)) and getattr(m, "content", "")
    ]
    if not prior:
        return {"current_step": "contextualize"}

    def _role(m) -> str:
        return "用户" if isinstance(m, HumanMessage) else "助手"

    history = "\n".join(
        f"{_role(m)}：{str(m.content)[:280]}" for m in prior[-6:]
    )
    prompt = CONTEXTUALIZE_PROMPT.format(history=history, query=state.user_query)
    rewritten = get_deterministic_llm().invoke(prompt).content.strip().strip('"').strip()

    if not rewritten or rewritten == state.user_query:
        return {"current_step": "contextualize"}

    log_event(logger, "contextualize", original=state.user_query, rewritten=rewritten)
    return {"user_query": rewritten, "current_step": "contextualize"}


# ── 意图路由 ────────────────────────────────────────────────────
def classify_intent(state: AgentState) -> dict:
    """判断意图，路由到对应子管线或 general 分支。"""
    prompt = INTENT_CLASSIFIER_PROMPT.format(user_query=state.user_query)
    raw = get_deterministic_llm().invoke(prompt).content.strip().lower()
    for intent in ("comparison", "timeline", "recommendation", "general"):
        if intent in raw:
            log_event(logger, "classify", query=state.user_query, intent=intent)
            return {"intent": intent, "current_step": f"classify→{intent}"}
    log_event(logger, "classify", query=state.user_query, intent="general", note="fallback")
    return {"intent": "general", "current_step": "classify→general"}


# ── 长期记忆读取（S5） ──────────────────────────────────────────
def load_memory(state: AgentState) -> dict:
    """从持久化存储读取用户偏好，注入 state。"""
    prefs = load_preferences(state.user_id)
    log_event(
        logger, "load_memory",
        user=state.user_id,
        artists=prefs.get("artists", []),
        styles=prefs.get("styles", []),
    )
    return {"user_preferences": prefs, "current_step": "load_memory"}


# ── 反思 ────────────────────────────────────────────────────────
def reflection(state: AgentState) -> dict:
    """检查最终回答是否充分；不充分则标记 RETRY 触发 web 兜底。"""
    answer = state.final_answer
    if not answer and state.messages:
        last = state.messages[-1]
        answer = getattr(last, "content", "") or ""

    # 已经兜底过一次就不再重试，避免死循环
    if state.retry_count >= 1:
        log_event(logger, "reflection", verdict="PASS", note="retry_exhausted")
        return {"reflection_notes": "PASS", "final_answer": answer, "current_step": "reflection"}

    prompt = REFLECTION_PROMPT.format(user_query=state.user_query, final_answer=answer)
    verdict = get_deterministic_llm().invoke(prompt).content.strip().upper()
    notes = "RETRY" if "RETRY" in verdict else "PASS"
    log_event(logger, "reflection", verdict=notes, answer_len=len(answer or ""))
    return {"reflection_notes": notes, "final_answer": answer, "current_step": "reflection"}


# ── web 兜底（S4） ──────────────────────────────────────────────
def web_fallback(state: AgentState) -> dict:
    """本地信息不足时联网搜索并重新综合回答。"""
    from src.tools.web_search import _search_impl

    results = _search_impl(state.user_query)
    log_event(logger, "web_fallback", query=state.user_query, results=results)
    web_text = "\n".join(
        f"- {r['title']}: {r['snippet']} ({r.get('url', '')})" for r in results
    )
    prompt = WEB_FALLBACK_SYNTHESIZE_PROMPT.format(
        user_query=state.user_query,
        prev_answer=state.final_answer or "(无)",
        web_results=web_text,
    )
    answer = get_llm(0.4).invoke(prompt).content
    return {
        "web_results": results,
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "retry_count": state.retry_count + 1,
        "current_step": "web_fallback",
    }


# ── 长期记忆写入（S5） ──────────────────────────────────────────
def save_memory(state: AgentState) -> dict:
    """
    推荐场景后，把用户明确表达喜欢的画家/风格写入持久化存储。
    只在 recommendation 意图下写入（此时用户确实表达了偏好）。
    """
    if state.intent == "recommendation":
        for artist in state.subjects:
            upsert_preference(state.user_id, "artist", artist, weight=1.0)
        if state.extracted_features:
            # 存一个精简的风格关键词（取前若干词）
            style_kw = state.extracted_features.strip()[:80]
            if style_kw:
                upsert_preference(state.user_id, "style", style_kw, weight=1.0)
        log_event(logger, "save_memory", user=state.user_id, saved_artists=state.subjects)
    return {"current_step": "save_memory"}
