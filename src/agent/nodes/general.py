"""
general 分支：ReAct 工具循环。

用于开放式/单一事实查询，让 LLM 自主决定调用哪个工具、调用几次。
保留原有的 agent ⇄ tools 循环能力。
"""

from typing import Literal

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.agent.prompts import SYSTEM_PROMPT
from src.tools.retrieval import semantic_search, exact_lookup
from src.tools.knowledge import query_painter_knowledge
from src.tools.image_lookup import image_lookup
from src.tools.page_reader import read_page_image
from src.tools.web_search import web_search
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
