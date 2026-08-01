"""
general 分支：ReAct 工具循环。

用于开放式/单一事实查询，让 LLM 自主决定调用哪个工具、调用几次。
保留原有的 agent ⇄ tools 循环能力。
"""

import json
from typing import Literal

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.prompts import SYSTEM_PROMPT
from src.tools.retrieval import semantic_search, exact_lookup
from src.tools.knowledge import query_painter_knowledge
from src.tools.image_lookup import image_lookup
from src.tools.page_reader import read_page_image
from src.tools.web_search import web_search
from src.retrieval.relevance import llm_relevance_filter
from src.utils.llm import get_deterministic_llm
from src.utils.logging_config import get_logger, log_event

logger = get_logger("general")

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
]


def _get_llm_with_tools():
    return get_deterministic_llm().bind_tools(GENERAL_TOOLS)


def general_agent(state: AgentState) -> dict:
    """核心 LLM 节点：读消息历史，决定直接回答或调用工具。"""
    messages = state.messages
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = _get_llm_with_tools().invoke(messages)
    tool_calls = [tc.get("name") for tc in getattr(response, "tool_calls", []) or []]
    if tool_calls:
        log_event(logger, "react", action="call_tools", tools=tool_calls)
    else:
        log_event(logger, "react", action="final_answer")
    return {"messages": [response], "current_step": "general_agent"}


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


def general_tools(state: AgentState) -> dict:
    """执行工具调用；对 semantic_search 结果做相关性过滤（Stage 4）。

    过滤放在图节点层而非工具内部：semantic_search 工具保持确定性可测
    （eval Recall@5 与 test_tools 直接消费它，不含 LLM 波动），LLM 判断
    留在编排层且每次过滤日志可观测。过滤失败的降级在模块内部完成。
    """
    out = _tool_node.invoke(state)
    messages = out.get("messages") if isinstance(out, dict) else None
    if not messages:
        return out

    # semantic_search 的 query 从触发调用的 AIMessage tool_calls 里取
    queries: dict[str, str] = {}
    for tc in getattr(state.messages[-1], "tool_calls", None) or []:
        if tc.get("name") == "semantic_search":
            queries[tc.get("id")] = (tc.get("args") or {}).get("query") or state.user_query
    if not queries:
        return out

    return {
        **out,
        "messages": [
            _filter_search_message(m, queries[m.tool_call_id])
            if isinstance(m, ToolMessage) and m.name == "semantic_search" and m.tool_call_id in queries
            else m
            for m in messages
        ],
    }
