"""
ArtAgent 混合架构 LangGraph。

  START
    └─► load_memory            读取用户长期偏好（S5）
          └─► contextualize    多轮指代消解（把"他/这幅"改写成具体对象）
                └─► classify   意图路由
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
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent import nodes as N
from src.utils.logging_config import get_logger, log_event, traced

logger = get_logger("graph")


# ── 路由函数 ────────────────────────────────────────────────────
def _capability_supported(intent: str, dataset_id: str) -> bool:
    """
    路由层能力开关（Stage 2）：进入 timeline / recommendation 前，检查当前
    生效数据源的 schema 是否声明了对应能力（分组轴 / 实体+描述列）。

    不支持则降级 general，而不是硬跑一个不成立的 groupby。只读 schema，
    不触发数据集加载。Stage 2 阶段 SemArt 恒为 True，暂不会触发降级；
    Stage 5 用户表格接入后真正生效。
    """
    from src.retrieval.structured_retriever import get_structured_retriever

    try:
        schema = get_structured_retriever(dataset_id).schema
    except Exception as e:
        logger.warning("[capability] 数据源 %s 不可用，降级 general：%s", dataset_id, e)
        return False
    if intent == "timeline":
        return schema.supports_timeline
    if intent == "recommendation":
        return schema.supports_recommendation
    return True


def _route_by_intent(
    state: AgentState,
) -> Literal["comparison", "timeline", "recommendation", "general"]:
    intent = state.intent
    if intent not in ("comparison", "timeline", "recommendation"):
        return "general"
    # 能力开关：数据源 schema 不支持该管线能力时降级 general
    if intent in ("timeline", "recommendation") and not _capability_supported(
        intent, state.dataset_id
    ):
        log_event(
            logger, "capability_gate",
            intent=intent, dataset_id=state.dataset_id, action="downgrade→general",
        )
        return "general"
    return intent


def _route_after_reflection(state: AgentState) -> Literal["web_fallback", "save_memory"]:
    if state.reflection_notes == "RETRY" and state.retry_count < 1:
        return "web_fallback"
    return "save_memory"


# ── 构建图 ──────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)

    # 用 traced 包装每个函数节点，统一记录节点耗时（ms）
    def add(name, fn):
        builder.add_node(name, traced(name, fn))

    # 公共节点
    add("load_memory", N.load_memory)
    add("contextualize", N.contextualize)
    add("classify", N.classify_intent)
    add("reflection", N.reflection)
    add("web_fallback", N.web_fallback)
    add("save_memory", N.save_memory)

    # comparison 子管线
    add("comp_decompose", N.comparison_decompose)
    add("comp_retrieve", N.comparison_retrieve)
    add("comp_synthesize", N.comparison_synthesize)

    # timeline 子管线
    add("tl_subject", N.timeline_extract_subject)
    add("tl_periods", N.timeline_gather_periods)
    add("tl_synthesize", N.timeline_synthesize)

    # recommendation 子管线
    add("rec_extract", N.recommendation_extract_features)
    add("rec_search", N.recommendation_feature_search)
    add("rec_filter", N.recommendation_relevance_filter)
    add("rec_synthesize", N.recommendation_synthesize)

    # general 分支（ReAct）
    add("general_agent", N.general_agent)
    # Stage 4：ToolNode 包成普通节点——执行后对 semantic_search 结果做相关性
    # 过滤（节点名不变，service.py 的"执行工具"标签无需同步）；包成普通函数
    # 后也自然获得 traced 节点耗时观测
    add("general_tools", N.general_tools)

    # ── 连线 ──
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "contextualize")
    builder.add_edge("contextualize", "classify")

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
