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
from src.data.access import format_evidence_block
from src.memory.store import load_preferences

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


# ── 查询改写 + 拆分（P0-③，替代 contextualize） ────────────────
def rewrite_split(state: AgentState) -> dict:
    """查询改写 + 多问题拆分。

    改写后的完整问题写回 user_query（下游全部读它，全局生效）；
    rewritten_question / sub_questions 存入 state，供意图打分与后续
    多意图并行检索使用。LLM 失败自动回落原问题，不阻塞主流程。
    """
    from src.agent.rewrite import rewrite_and_split

    result = rewrite_and_split(state.user_query, state.messages)
    log_event(
        logger,
        "rewrite_split",
        original=state.user_query,
        rewritten=result.rewritten_question,
        sub_questions=result.sub_questions,
    )
    return {
        "user_query": result.rewritten_question,
        "rewritten_question": result.rewritten_question,
        "sub_questions": result.sub_questions,
        "rewritten_key_entities": result.key_entities,
        "rewrite_ambiguous": result.ambiguous,
        "current_step": "rewrite_split",
    }


# ── 意图路由 ────────────────────────────────────────────────────
def classify_intent(state: AgentState) -> dict:
    """意图树打分分类（P0-②）：对所有叶子打分，主意图仍路由到现有分支。

    分数写入 state.intent_scores（供 UI 展示与后续多意图并行使用）；
    LLM 失败 / 畸形输出自动回落 general，行为与旧版一致。
    """
    from src.agent.intent_tree import classify_intents

    scores, intent = classify_intents(state.user_query)
    log_event(
        logger,
        "classify",
        query=state.user_query,
        intent=intent,
        top_scores="; ".join(f"{s.leaf.id}={s.score:.2f}" for s in scores[:3]),
    )
    return {
        "intent": intent,
        "intent_scores": [s.to_dict() for s in scores],
        "current_step": f"classify→{intent}",
    }


# ── RAG 开关（收尾项）：判断是否需要检索 ───────────────────────
import re as _re

# 确定性问候模式（全匹配，允许尾部语气词/标点）：命中即无需检索，
# 不依赖 LLM 打分（打分有随机性，2026-08-02 实测同题两次结果不同）。
_GREETING_RE = _re.compile(
    r"^(你好|您好|嗨|哈喽|hello|hi|hey|thanks|thank you|谢谢|多谢|"
    r"再见|拜拜|bye|你是谁|你能做什么|能帮我什么)[!！。.？?～~呀啊呢吧 ]*$",
    _re.IGNORECASE,
)


def _rag_gate(question: str, intent_scores) -> bool:
    """是否需要检索：寒暄类（system）意图高分 → 不需要。

    双保险：① 确定性问候词匹配（主要）；② LLM 意图打分 system_greeting
    ≥ 0.7 且为最高分（兜底）。intent_scores 为已按分数降序的 dict 列表。
    """
    if not question:
        return True
    if _GREETING_RE.match(question.strip()):
        return False
    if not intent_scores:
        return True
    best = intent_scores[0]
    if not isinstance(best, dict):
        return True
    try:
        score = float(best.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    if best.get("kind") == "system" and score >= 0.7:
        return False
    return True


def rag_gate(state: AgentState) -> dict:
    """RAG 开关节点：寒暄高分 → 关闭检索（走直接回答），否则放行。"""
    needed = _rag_gate(state.user_query, state.intent_scores)
    log_event(logger, "rag_gate", query=state.user_query, rag_needed=needed)
    return {"rag_needed": needed, "current_step": "rag_gate"}


def direct_answer(state: AgentState) -> dict:
    """无需检索时的直接回答节点（LLM 不带任何工具）。"""
    from src.utils.llm import get_deterministic_llm

    prompt = (
        "你是艺术领域助手。请简短、自然地回答用户的问题，不要编造事实。\n\n"
        f"用户：{state.user_query}\n\n回答："
    )
    answer = get_deterministic_llm().invoke(prompt).content.strip()
    log_event(logger, "direct_answer", query=state.user_query)
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "current_step": "direct_answer",
    }


# ── 信息缺口澄清（P1-1.5） ─────────────────────────────────────
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
        gap, message = _info_gap(state.user_query or "", state.intent)
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


# ── 多意图并行检索（P0-A / Phase 2） ───────────────────────────
def multi_retrieve(state: AgentState) -> dict:
    """复合问题并行预取证据：sub_questions > 1 时按子问题并行 semantic_search。

    结果按子问题分组存入 state.multi_evidence，由 ContextBuilder 以
    【子任务N】分组注入上下文；agent 主循环仍可自行调用工具补细节。
    单一子问题时零开销放行（不启动线程池、不检索）。
    """
    subs = [s for s in (state.sub_questions or []) if s.strip()]
    if len(subs) <= 1:
        return {"multi_evidence": {}, "current_step": "multi_retrieve"}

    from concurrent.futures import ThreadPoolExecutor

    from src.tools.retrieval import semantic_search

    def _search(sub: str) -> list[dict]:
        try:
            return semantic_search.invoke({"query": sub, "top_k": 5})
        except Exception as e:  # noqa: BLE001 — 单子任务失败不影响整体
            log_event(logger, "multi_retrieve", sub=sub, error=str(e))
            return []

    grouped: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(subs))) as pool:
        futures = {pool.submit(_search, sub): sub for sub in subs}
        for future in futures:
            grouped[futures[future]] = future.result()

    log_event(
        logger, "multi_retrieve",
        sub_questions=len(subs),
        hits={sub[:24]: len(items) for sub, items in grouped.items()},
    )
    return {"multi_evidence": grouped, "current_step": "multi_retrieve"}


# ── 长期记忆读取（S5） ──────────────────────────────────────────
def load_memory(state: AgentState) -> dict:
    """从持久化存储读取用户偏好与会话摘要，注入 state。"""
    from src.memory.summary import load_summary

    prefs = load_preferences(state.user_id)
    summary = load_summary(state.conversation_id)
    log_event(
        logger, "load_memory",
        user=state.user_id,
        artists=prefs.get("artists", []),
        styles=prefs.get("styles", []),
        summary_len=len(summary),
    )
    return {
        "user_preferences": prefs,
        "conversation_summary": summary,
        "current_step": "load_memory",
    }


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
    # 证据格式化统一走数据访问层
    web_text = format_evidence_block(results, "- {title}: {snippet} ({url})")
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
    Phase 4/5：触发会话滚动摘要（达到轮数后增量压缩并落库）。

    说明：旧版"推荐场景写偏好"逻辑已删除——扁平化后 state.subjects 不再由
    管线填充，偏好记忆改由 Agent 用 remember 工具显式记录（Phase 4）。
    """
    from src.memory.summary import maybe_summarize

    summary = maybe_summarize(
        state.messages, state.conversation_id, state.user_id,
        volume_chars=state.context_chars,
    )
    log_event(
        logger, "save_memory",
        user=state.user_id,
        turns=sum(1 for m in state.messages if getattr(m, "type", "") == "human"),
        tool_rounds=state.tool_rounds,
        context_chars=state.context_chars,
        summary_len=len(summary),
    )
    return {"current_step": "save_memory", "conversation_summary": summary}
