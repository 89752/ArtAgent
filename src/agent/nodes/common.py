"""共享节点：意图路由、记忆读写、反思、web 兜底，以及工具函数。"""

import re

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
from src.utils.json_utils import parse_json
from src.data.access import format_evidence_block
from src.memory.store import load_preferences

logger = get_logger("nodes")


# ── 工具函数 ────────────────────────────────────────────────────
# parse_json 统一在 src/utils/json_utils.py，这里保留导出供既有调用方使用。


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


# ── 查询改写 + 拆分（替代 contextualize） ──────────────────────
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
        "original_user_query": state.user_query,
        "user_query": result.rewritten_question,
        "rewritten_question": result.rewritten_question,
        "sub_questions": result.sub_questions,
        "rewritten_key_entities": result.key_entities,
        "rewrite_ambiguous": result.ambiguous,
        "current_step": "rewrite_split",
    }


# ── 意图路由 ────────────────────────────────────────────────────

# 确定性预筛：高置信场景规则短路（不花 LLM），未命中才走 LLM 打分。
# 注意：只覆盖"几乎不会误判"的模式——知识定义/寒暄/算术、时效信息、
# 强比较动词、演变/推荐动词；"区别/不同"类留给 LLM（领域比较 vs 常识区别）。
_KNOWLEDGE_PREFIX_RE = re.compile(
    r"^(什么是|啥是|什么叫|解释一下|简单解释)[^，。！？]{0,40}"
    r"[!！。.？?～~呀啊呢吧 ]*$|"
    r"^[0-9]+\s*[+\-*×÷/]\s*[0-9]+[^，。！？]{0,20}$",
    re.IGNORECASE,
)
_WEB_TRIGGER_RE = re.compile(
    r"(天气|气温|降雨|新闻|股价|汇率|现在几点|几点钟|最新行情|今日头条)",
    re.IGNORECASE,
)
_COMPARE_PREFIX_RE = re.compile(
    r"^(对比|比较|比一比|哪个更|谁更|孰更)[^，。！？]*$",
    re.IGNORECASE,
)
_TIMELINE_TRIGGER_RE = re.compile(
    r"(风格演变|发展历程|不同时期|阶段变化|时间线|早期.*晚期|晚期.*早期)",
    re.IGNORECASE,
)
_RECOMMEND_TRIGGER_RE = re.compile(
    r"(推荐|类似.*画家|喜欢.*(风格|画家).*(推荐|还有谁|还会喜欢))",
    re.IGNORECASE,
)


def _prefilter_route(question: str) -> tuple[str, str] | None:
    """确定性预筛：命中返回 (route, reason)，未命中返回 None。"""
    q = (question or "").strip()
    if not q:
        return "rag", "空问题默认检索"
    if _GREETING_RE.match(q) or _KNOWLEDGE_PREFIX_RE.match(q):
        return "direct", "prefilter:寒暄/常识定义/算术"
    if _WEB_TRIGGER_RE.search(q):
        return "web", "prefilter:时效/实时信息"
    if _COMPARE_PREFIX_RE.match(q):
        return "comparison", "prefilter:强比较动词"
    if _TIMELINE_TRIGGER_RE.search(q):
        return "timeline", "prefilter:演变/时间线"
    if _RECOMMEND_TRIGGER_RE.search(q):
        return "recommendation", "prefilter:推荐动词"
    return None


def classify_intent(state: AgentState) -> dict:
    """意图打分 + 路由决策：对所有叶子打分，输出 route。

    分数写入 state.intent_scores（供 UI 展示与后续多意图并行使用）；
    LLM 失败 / 畸形输出自动回落 general，行为与旧版一致。
    """
    from src.agent.intent_tree import classify_intents

    pref = _prefilter_route(state.user_query)
    if pref is not None:
        route, reason = pref
        scores, intent = [], "general"
        # 预筛命中时仍给一个轻量 LLM 打分供 UI/建议使用（失败不阻塞）
        try:
            scores, intent, _, _ = classify_intents(state.user_query)
        except Exception:  # noqa: BLE001
            pass
    else:
        scores, intent, route, reason = classify_intents(state.user_query)
    log_event(
        logger,
        "classify",
        query=state.user_query,
        intent=intent,
        route=route,
        route_reason=reason,
        top_scores="; ".join(f"{s.leaf.id}={s.score:.2f}" for s in scores[:3]),
    )
    return {
        "intent": intent,
        "route": route,
        "route_reason": reason,
        "intent_scores": [s.to_dict() for s in scores],
        "current_step": f"classify→{route}",
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


# ── 多意图并行检索 ─────────────────────────────────────────────
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


# ── 长期记忆读取 ──────────────────────────────────────────────────
def load_memory(state: AgentState) -> dict:
    """检索注入相关记忆 + 读取会话滚动摘要与用户画像。

    - 记忆检索：按当前问题对 memory_items 打分（相关度+新鲜度+重要性），
      只注入相关条目（解决"读到无关旧记忆"），带来源/时间；
    - 身份：统一走 get_memory_user_id()（与 remember/recall 工具一致），
      避免 web 端 state.user_id 与工具层 MEMORY_USER_ID 不一致（2026-08-04）；
    - 兼容：user_preferences 仍返回（旧 profile/API 消费），但画像块已
      由 memory_block 承载更完整的记忆语义。
    """
    from src.agent.context import build_memory_block
    from src.memory.memory_items import search_memories
    from src.memory.memory_items import get_memory_user_id
    from src.memory.profile import load_profile_item
    from src.memory.summary import load_summary, load_summary_item

    uid = get_memory_user_id()
    prefs = load_preferences(uid)
    summary = load_summary(state.conversation_id)
    items = search_memories(
        uid,
        state.user_query or "",
        scope="user",
        top_k=5,
    )
    # 跨轮引用兜底：问题与记忆词面不重叠时，检索可能命中不足；
    # 补 top 最近/最重要条目，保证"我喜欢什么/我住在哪"也能引用
    if len(items) < 3:
        fallback = search_memories(
            uid, "", scope="user", top_k=3, min_score=0.0,
        )
        have = {i["id"] for i in items}
        items = items + [i for i in fallback if i["id"] not in have]
    # 同一会话上次的滚动摘要，预算内注入
    memory_block = build_memory_block(items, budget=600)
    # 跨线程用户画像（"记住你"）优先注入，预算内
    profile_item = load_profile_item(uid)
    if profile_item and (profile_item.get("content") or "").strip():
        memory_block = (
            f"【用户画像】{str(profile_item['content']).strip()[:200]}"
            + ("\n\n" + memory_block if memory_block else "")
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


# ── web 兜底 ───────────────────────────────────────────────────
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


# ── 工具升级兜底（reflection RETRY 不再只走联网） ─────────────────
def tool_upgrade(state: AgentState) -> dict:
    """反思 RETRY 后的工具升级：按 route 意向先补本地证据，再补联网。

    - route=web → 直接联网兜底（与旧 web_fallback 等价）；
    - 其余 → 先 semantic_search 补本地证据，无结果再升级联网。
    """
    from src.agent.prompts import LOCAL_EVIDENCE_SYNTHESIZE_PROMPT
    from src.tools.retrieval import semantic_search
    from src.utils.llm import get_llm

    if state.route == "web":
        return web_fallback(state)

    try:
        results = semantic_search.invoke(
            {"query": state.user_query, "top_k": 5}
        )
    except Exception:  # noqa: BLE001
        results = []
    evidence = format_evidence_block(results)
    if not evidence.strip():
        return web_fallback(state)

    prompt = LOCAL_EVIDENCE_SYNTHESIZE_PROMPT.format(
        user_query=state.user_query,
        prev_answer=state.final_answer or "(无)",
        evidence=evidence,
    )
    answer = get_llm(0.4).invoke(prompt).content
    log_event(
        logger, "tool_upgrade",
        query=state.user_query, route=state.route, hits=len(results),
    )
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "retry_count": state.retry_count + 1,
        "current_step": "tool_upgrade",
    }


# ── 长期记忆写入 ──────────────────────────────────────────────────
def save_memory(state: AgentState) -> dict:
    """触发会话滚动摘要 + 用户画像刷新 + 自动抽取（开关控制）。

    说明：旧版"推荐场景写偏好"逻辑已删除——扁平化后 state.subjects 不再由
    管线填充，偏好记忆改由 Agent 用 remember 工具显式记录。
    滚动摘要只落 conversation_summary 一张表（原 episodes 双写已移除）。
    """
    from src.memory.summary import maybe_summarize
    from src.memory.memory_items import get_memory_user_id
    from src.memory.profile import maybe_refresh_profile

    uid = get_memory_user_id()
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
        from src.memory.extract import maybe_extract

        extracted_turns, extract_result = maybe_extract(
            state.messages,
            extracted_turns=state.memory_extracted_turns,
        )
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
