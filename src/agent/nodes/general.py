"""
general 分支：ReAct 工具循环。

用于开放式/单一事实查询，让 LLM 自主决定调用哪个工具、调用几次。
保留原有的 agent ⇄ tools 循环能力。
"""

import json
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.prompts import SYSTEM_PROMPT
from src.tools.retrieval import semantic_search, exact_lookup
from src.tools.knowledge import query_painter_knowledge
from src.tools.image_lookup import image_lookup
from src.tools.page_reader import read_page_image
from src.tools.web_search import web_search
from src.tools.guard import ToolDecision, guard_tool_message, validate_args
from src.skills.loader import register_skills
from src.tools.memory import remember, recall, forget
from src.tools.collections import save_collection, list_collections, list_preferences
from src.tools.capabilities import (
    compare_subjects,
    timeline_by_periods,
    recommend_with_exclusions,
)
from src.retrieval.relevance import llm_relevance_filter
from src.utils.llm import get_deterministic_llm
from src.utils.logging_config import get_logger, log_event

logger = get_logger("general")

# Phase 5：ReAct 工具轮次上限（实测出现过 29 次调用不收敛的循环）
MAX_TOOL_ROUNDS = 5

# general 分支可用的全部工具（Stage 1 精简：7 → 5。
# compare_artwork_styles 删除——外层 Agent 拿到两幅画的元数据后可自行组织对比；
# analyze_image 并入 image_lookup 的 analyze 参数。
# Stage 3 补：read_page_image——Qwen-VL 读取用户上传 PDF 整页图）
GENERAL_TOOLS = [
    semantic_search,
    exact_lookup,
    query_painter_knowledge,
    image_lookup,
    read_page_image,
    web_search,
    # Stage 7（能力工具化）：原子子管线逻辑下沉，agent 统一编排
    compare_subjects,
    timeline_by_periods,
    recommend_with_exclusions,
    # Phase 4：agent 主动记忆
    remember,
    recall,
    forget,
    # Phase 5：收藏与偏好
    save_collection,
    list_collections,
    list_preferences,
] + register_skills()

TOOL_BY_NAME: dict[str, object] = {t.name: t for t in GENERAL_TOOLS}


def _tool_schema(tool: object) -> dict:
    """取工具的 JSON Schema（兼容 pydantic v1/v2 的 args_schema，兜底 tool.args）。"""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        for method in ("model_json_schema", "schema"):
            fn = getattr(args_schema, method, None)
            if fn is None:
                continue
            try:
                return fn()
            except Exception:
                continue
    args = getattr(tool, "args", None)
    if isinstance(args, dict) and args:
        return {
            "properties": args,
            "required": [
                k for k, v in args.items()
                if isinstance(v, dict) and "default" not in v
            ],
        }
    return {}


def _repeat_guard_message(tc: dict, name: str) -> ToolMessage:
    """防重复调用守卫：相同工具+相同参数已被执行过时，提示模型换路。"""
    return ToolMessage(
        content=json.dumps(
            {
                "status": "REPEAT",
                "message": (
                    f"工具 {name} 已用相同参数调用过，结果不会变化。请改用其他方式；"
                    "如果目标是读取用户上传的整页图文档，请调用 "
                    "read_page_image(doc_name=..., page=...) 按文档定位页面。"
                ),
            },
            ensure_ascii=False,
        ),
        name=name,
        tool_call_id=tc.get("id"),
        id=f"guard:{tc.get('id')}",
    )


def _guarded_tool_calls(state: AgentState, tool_node) -> tuple[list, list]:
    """P0-1 tool invocation guard: 3-state validation before execution.

    - SUCCESS: args pass JSON-schema validation -> executed via tool_node
    - NEED_CLARIFICATION / FAILED: tool is NOT executed; a guard ToolMessage
      is injected so the model can fix args or ask the user (the graph edge
      general_tools -> general_agent feeds it back).
    """
    last = state.messages[-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return []

    valid_calls: list[dict] = []
    guard_by_id: dict[str, ToolMessage] = {}
    executed_sigs: set[str] = set(state.executed_tool_signatures or [])
    new_sigs: list[str] = []
    for tc in calls:
        name = tc.get("name")
        tool = TOOL_BY_NAME.get(name)
        if tool is None:
            guard_by_id[tc.get("id")] = guard_tool_message(
                tc.get("id"),
                str(name),
                ToolDecision(status="FAILED", errors=[f"unknown tool: {name}"]),
            )
            continue
        sig = f"{name}:{json.dumps(tc.get('args') or {}, sort_keys=True, ensure_ascii=False)}"
        if sig in executed_sigs:
            guard_by_id[tc.get("id")] = _repeat_guard_message(tc, name)
            continue
        decision = validate_args(_tool_schema(tool), tc.get("args") or {})
        if decision.status == "SUCCESS":
            valid_calls.append(tc)
            new_sigs.append(sig)
        else:
            guard_by_id[tc.get("id")] = guard_tool_message(tc.get("id"), name, decision)
            log_event(
                logger, "tool_guard", tool=name, status=decision.status,
                missing=decision.missing, errors=decision.errors,
            )

    executed: list[ToolMessage] = []
    if valid_calls:
        valid_ai = AIMessage(
            content=getattr(last, "content", "") or "",
            tool_calls=valid_calls,
            id=f"{last.id}:valid" if last.id else "valid-calls",
        )
        copy_fn = getattr(state, "model_copy", None) or state.copy
        valid_state = copy_fn(update={"messages": list(state.messages)[:-1] + [valid_ai]})
        out = tool_node.invoke(valid_state)
        executed = out.get("messages") if isinstance(out, dict) else []
    executed_by_id = {
        getattr(m, "tool_call_id", None): m
        for m in executed
        if getattr(m, "tool_call_id", None)
    }

    merged: list = []
    for tc in calls:
        cid = tc.get("id")
        if cid in guard_by_id:
            merged.append(guard_by_id[cid])
        elif cid in executed_by_id:
            merged.append(executed_by_id[cid])
        else:
            log_event(logger, "tool_guard", note="missing_tool_output", call_id=cid)
    return merged, new_sigs


def _get_llm_with_tools():
    return get_deterministic_llm().bind_tools(GENERAL_TOOLS)


def _intent_suggestion_message(state: AgentState):
    """把意图打分的工具建议注入为一条 system 消息（软指引，不强制）。"""
    from src.agent.intent_tree import intent_tool_suggestions

    hints = intent_tool_suggestions(state.intent_scores)
    if not hints:
        return None
    body = "意图分析建议优先考虑的工具（仅供参考，按需选择）：\n" + "\n".join(
        f"- {h}" for h in hints
    )
    return SystemMessage(content=body)


def general_agent(state: AgentState) -> dict:
    """核心 LLM 节点：ContextBuilder 组装结构化上下文，决定直接回答或调用工具。"""
    from src.agent.context import (
        ContextBlocks,
        ContextBudget,
        apply_budget,
        build_profile_block,
        build_session_block,
        build_summary_block,
        condense_tool_messages,
        estimate_context_chars,
        extract_evidence_from_messages,
        format_evidence_block,
        format_multi_evidence,
        format_skills_index,
        trim_history,
    )
    from src.skills.loader import load_skills

    history = trim_history(state.messages)
    system = SYSTEM_PROMPT
    skills_index = format_skills_index(load_skills())
    if skills_index:
        system += "\n\n" + skills_index
    blocks = ContextBlocks(
        system=system,
        profile=build_profile_block(state.user_preferences),
        summary=build_summary_block(state.conversation_summary),
        session=build_session_block(
            {
                "shown_artworks": state.shown_artworks,
                "recommended_artists": state.recommended_artists,
                "pending_clarification": state.pending_clarification,
                "uploaded_docs": state.uploaded_docs,
            }
        ),
        evidence=format_evidence_block(extract_evidence_from_messages(state.messages)),
        subtasks=format_multi_evidence(state.multi_evidence),
    )
    blocks = apply_budget(blocks)
    messages = blocks.to_system_messages()
    suggestion = _intent_suggestion_message(state)
    if suggestion is not None:
        messages.append(suggestion)
    messages.extend(condense_tool_messages(history))
    context_chars = estimate_context_chars(blocks)
    log_event(logger, "context_volume", chars=context_chars,
              history_turns=len([m for m in blocks.history if getattr(m, "type", "") == "human"]))

    response = _get_llm_with_tools().invoke(messages)
    tool_calls = [tc.get("name") for tc in getattr(response, "tool_calls", []) or []]
    if tool_calls:
        log_event(logger, "react", action="call_tools", tools=tool_calls)
    else:
        log_event(logger, "react", action="final_answer")
    return {
        "messages": [response],
        "current_step": "general_agent",
        "context_chars": context_chars,
    }


def general_should_continue(state: AgentState) -> Literal["tools", "done"]:
    """路由：最后一条消息有工具调用则执行工具，否则收尾。"""
    last = state.messages[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "done"


# ── 工具执行节点（Stage 4 起带检索结果相关性过滤） ───────────────
_tool_node = ToolNode(GENERAL_TOOLS)


def _filter_search_message(msg: ToolMessage, query: str) -> ToolMessage:
    """对 semantic_search 的 ToolMessage 做相关性过滤，保持 JSON 形状不变。

    web/service.py 的 _parse_artworks_from_messages 靠 json.loads 解析
    ToolMessage 配图——过滤后重新序列化仍为 list[dict]，UI 消费不受影响
    （顺带好处：无关画作不再进配图卡片）。内容非 JSON 数组时原样返回。
    """
    try:
        items = json.loads(msg.content)
    except (TypeError, json.JSONDecodeError):
        return msg
    if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
        return msg
    filtered = llm_relevance_filter(query, items, min_keep=2)
    if len(filtered) == len(items):
        return msg  # 无删减，避免无谓的重序列化
    return ToolMessage(
        content=json.dumps(filtered, ensure_ascii=False),
        name=msg.name,
        tool_call_id=msg.tool_call_id,
        id=msg.id,
    )


def _ledger_updates(merged, state: AgentState) -> dict:
    """会话台账自动登记：已展示画作 / 已推荐画家（去重保序）。"""
    shown = list(state.shown_artworks or [])
    recommended = list(state.recommended_artists or [])
    for msg in merged:
        name = getattr(msg, "name", "") or ""
        try:
            data = json.loads(str(msg.content))
        except Exception:
            continue
        if name in ("semantic_search", "exact_lookup", "image_lookup") and isinstance(
            data, list
        ):
            for d in data:
                if isinstance(d, dict) and d.get("title"):
                    shown.append(str(d["title"]))
        elif name == "recommend_with_exclusions" and isinstance(data, dict):
            for c in data.get("candidates") or []:
                if isinstance(c, dict) and c.get("author"):
                    recommended.append(str(c["author"]))

    def _uniq(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {"shown_artworks": _uniq(shown), "recommended_artists": _uniq(recommended)}


def general_tools(state: AgentState) -> dict:
    """执行工具调用；对 semantic_search 结果做相关性过滤（Stage 4）。

    过滤放在图节点层而非工具内部：semantic_search 工具保持确定性可测
    （eval Recall@5 与 test_tools 直接消费它，不含 LLM 波动），LLM 判断
    留在编排层且每次过滤日志可观测。过滤失败的降级在模块内部完成。
    执行后自动更新会话台账（shown_artworks / recommended_artists）。
    """
    # Phase 5：轮次上限——停止执行并让模型基于已有信息直接回答
    if state.tool_rounds >= MAX_TOOL_ROUNDS:
        last = state.messages[-1]
        guards = [
            ToolMessage(
                content=(
                    '{"status":"LIMIT","message":"已达到工具调用轮次上限，'
                    '请停止调用工具并基于已有信息直接给出最终回答。"}'
                ),
                name=(tc.get("name") or "system"),
                tool_call_id=tc.get("id"),
            )
            for tc in getattr(last, "tool_calls", None) or []
        ]
        log_event(logger, "tool_guard", status="LIMIT", rounds=state.tool_rounds)
        return {"messages": guards, "tool_rounds": state.tool_rounds}

    merged, new_sigs = _guarded_tool_calls(state, _tool_node)
    ledger = _ledger_updates(merged, state)
    ledger["tool_rounds"] = state.tool_rounds + 1
    ledger["executed_tool_signatures"] = (
        list(state.executed_tool_signatures or []) + new_sigs
    )
    if not merged:
        return {**ledger, "messages": merged}

    # semantic_search 的 query 从触发调用的 AIMessage tool_calls 里取
    queries: dict[str, str] = {}
    for tc in getattr(state.messages[-1], "tool_calls", None) or []:
        if tc.get("name") == "semantic_search":
            queries[tc.get("id")] = (tc.get("args") or {}).get("query") or state.user_query
    if not queries:
        return {**ledger, "messages": merged}

    return {
        **ledger,
        "messages": [
            _filter_search_message(m, queries[m.tool_call_id])
            if isinstance(m, ToolMessage) and m.name == "semantic_search" and m.tool_call_id in queries
            else m
            for m in merged
        ],
    }
