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
from src.tools.style_comparison import compare_artwork_styles
from src.tools.image_analysis import analyze_image
from src.tools.image_lookup import image_lookup
from src.tools.web_search import web_search
from src.utils.llm import get_deterministic_llm

# general 分支可用的全部工具
GENERAL_TOOLS = [
    semantic_search,
    exact_lookup,
    query_painter_knowledge,
    compare_artwork_styles,
    analyze_image,
    image_lookup,
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
    return {"messages": [response], "current_step": "general_agent"}


def general_should_continue(state: AgentState) -> Literal["tools", "done"]:
    """路由：最后一条消息有工具调用则执行工具，否则收尾。"""
    last = state.messages[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "done"
