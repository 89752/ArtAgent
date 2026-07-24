"""
ArtAgent 混合架构 LangGraph。

  START
    └─► load_memory            读取用户长期偏好（S5）
          └─► classify         意图路由
                ├─ comparison ──► comp_decompose → comp_retrieve → comp_synthesize ─┐
                ├─ timeline ────► tl_subject → tl_periods → tl_synthesize ───────────┤
                ├─ recommendation► rec_extract → rec_search → rec_filter → rec_synth ┤
                └─ general ─────► general_agent ⇄ tools ──────────────────────────────┤
                                                                                      ▼
                                                                              [reflection]
                                                                          PASS │      │ RETRY
                                                                               ▼      ▼
                                                                        save_memory  web_fallback（S4）
                                                                               │      │
                                                                               ▼      ▼
                                                                              END    save_memory → END
"""

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent import nodes as N


# ── 路由函数 ────────────────────────────────────────────────────
def _route_by_intent(
    state: AgentState,
) -> Literal["comparison", "timeline", "recommendation", "general"]:
    return state.intent if state.intent in (
        "comparison",
        "timeline",
        "recommendation",
    ) else "general"


def _route_after_reflection(state: AgentState) -> Literal["web_fallback", "save_memory"]:
    if state.reflection_notes == "RETRY" and state.retry_count < 1:
        return "web_fallback"
    return "save_memory"


# ── 构建图 ──────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)

    # 公共节点
    builder.add_node("load_memory", N.load_memory)
    builder.add_node("classify", N.classify_intent)
    builder.add_node("reflection", N.reflection)
    builder.add_node("web_fallback", N.web_fallback)
    builder.add_node("save_memory", N.save_memory)

    # comparison 子管线
    builder.add_node("comp_decompose", N.comparison_decompose)
    builder.add_node("comp_retrieve", N.comparison_retrieve)
    builder.add_node("comp_synthesize", N.comparison_synthesize)

    # timeline 子管线
    builder.add_node("tl_subject", N.timeline_extract_subject)
    builder.add_node("tl_periods", N.timeline_gather_periods)
    builder.add_node("tl_synthesize", N.timeline_synthesize)

    # recommendation 子管线
    builder.add_node("rec_extract", N.recommendation_extract_features)
    builder.add_node("rec_search", N.recommendation_feature_search)
    builder.add_node("rec_filter", N.recommendation_relevance_filter)
    builder.add_node("rec_synthesize", N.recommendation_synthesize)

    # general 分支（ReAct）
    builder.add_node("general_agent", N.general_agent)
    builder.add_node("general_tools", ToolNode(N.GENERAL_TOOLS))

    # ── 连线 ──
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "classify")

    builder.add_conditional_edges(
        "classify",
        _route_by_intent,
        {
            "comparison": "comp_decompose",
            "timeline": "tl_subject",
            "recommendation": "rec_extract",
            "general": "general_agent",
        },
    )

    # comparison
    builder.add_edge("comp_decompose", "comp_retrieve")
    builder.add_edge("comp_retrieve", "comp_synthesize")
    builder.add_edge("comp_synthesize", "reflection")

    # timeline
    builder.add_edge("tl_subject", "tl_periods")
    builder.add_edge("tl_periods", "tl_synthesize")
    builder.add_edge("tl_synthesize", "reflection")

    # recommendation
    builder.add_edge("rec_extract", "rec_search")
    builder.add_edge("rec_search", "rec_filter")
    builder.add_edge("rec_filter", "rec_synthesize")
    builder.add_edge("rec_synthesize", "reflection")

    # general ReAct 循环
    builder.add_conditional_edges(
        "general_agent",
        N.general_should_continue,
        {"tools": "general_tools", "done": "reflection"},
    )
    builder.add_edge("general_tools", "general_agent")

    # 汇聚：反思 → 兜底或收尾
    builder.add_conditional_edges(
        "reflection",
        _route_after_reflection,
        {"web_fallback": "web_fallback", "save_memory": "save_memory"},
    )
    builder.add_edge("web_fallback", "save_memory")
    builder.add_edge("save_memory", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# 全局单例
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
