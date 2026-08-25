"""ArtAgent 图结构（温和版：纯 ReAct + 记忆 + 澄清 + 反思）。

  START
    └─► load_memory              读取用户长期偏好
          └─► ask_user           信息缺口澄清（不足→追问短路；否则放行）
                └─► general_agent ⇄ tools（ReAct；技能和推荐检索均是工具）
                      └─► reflection（质量自查；RETRY 最多重试一轮）
                            └─► save_memory → END

管线已删除：rewrite_split / classify / rag_gate / direct_answer /
multi_retrieve / tool_upgrade。领域流程（对比/时间线/推荐）改为技能。
"""

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent import nodes as N
from src.agent.state import AgentState
from src.utils.logging_config import get_logger, traced

logger = get_logger("graph")


def _build_checkpointer():
    """Use a durable SQLite saver when installed; retain an explicit fallback.

    ``langgraph-checkpoint-sqlite`` is an optional LangGraph distribution, so
    an older developer environment can still start while clearly signalling
    that restart recovery is unavailable until dependencies are updated.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        default = Path(__file__).resolve().parents[2] / "data" / "memory" / "checkpoints.db"
        path = Path(os.getenv("ARTAGENT_CHECKPOINT_DB_PATH", str(default)))
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite 未安装，暂回退 MemorySaver；"
            "生产部署必须安装该依赖以启用重启恢复"
        )
        return MemorySaver()


def _route_after_reflection(state: AgentState) -> str:
    """反思 RETRY 且未重试过 → 回 general_agent 再来一轮；否则收尾。"""
    if state.reflection_notes == "RETRY" and state.retry_count < 1:
        return "retry"
    return "save_memory"


def build_graph():
    builder = StateGraph(AgentState)

    def add(name, fn):
        builder.add_node(name, traced(name, fn))

    add("load_memory", N.load_memory)
    add("ask_user", N.ask_user)
    add("general_agent", N.general_agent)
    add("general_tools", N.general_tools)
    add("reflection", N.reflection)
    add("save_memory", N.save_memory)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "ask_user")
    # 信息不足 → 追问并短路；其余全部交给 ReAct。意图只用于澄清和 UI
    # 展示，不能把包含“推荐”的复合请求强制截断为单一固定工作流。
    builder.add_conditional_edges(
        "ask_user",
        lambda s: s.ask_user,
        {"ask": END, "continue": "general_agent"},
    )
    # ReAct：有工具调用就执行工具，否则交给反思
    builder.add_conditional_edges(
        "general_agent",
        N.general_should_continue,
        {"tools": "general_tools", "done": "reflection"},
    )
    builder.add_edge("general_tools", "general_agent")
    # 反思：RETRY 且未重试过 → 再跑一轮 ReAct；否则收尾
    builder.add_conditional_edges(
        "reflection",
        _route_after_reflection,
        {"retry": "general_agent", "save_memory": "save_memory"},
    )
    builder.add_edge("save_memory", END)

    checkpointer = _build_checkpointer()
    return builder.compile(checkpointer=checkpointer)


# 全局单例
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
