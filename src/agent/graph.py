"""
ArtAgent 混合架构 LangGraph。

  START
    └─► load_memory            读取用户长期偏好
          └─► rewrite_split    查询改写+拆分（含指代消解）
                └─► classify   意图打分（软指引，不再路由分支）
                      └─► rag_gate   RAG 开关（寒暄高分→直接回答，否则放行）
                            ├─(no_rag)→ direct_answer
                            └─(rag)→ ask_user   信息缺口澄清（不足→追问短路，否则放行）
                            └─► multi_retrieve   复合问题并行预取证据
                                  └─► general_agent ⇄ tools（ReAct + 工具守卫）
                              │
                              ▼
                      [reflection]
                   PASS │        │ RETRY
                        ▼        ▼
                 save_memory  web_fallback
                        │        │
                        ▼        ▼
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
    路由层能力开关：进入 timeline / recommendation 前，检查当前
    生效数据源的 schema 是否声明了对应能力（分组轴 / 实体+描述列）。

    不支持则降级 general，而不是硬跑一个不成立的 groupby。只读 schema，
    不触发数据集加载。核心库恒为 True，暂不会触发降级；
    用户表格接入后真正生效。
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


def _route_after_reflection(state: AgentState) -> Literal["tool_upgrade", "save_memory"]:
    if state.reflection_notes == "RETRY" and state.retry_count < 1:
        # RETRY 先按 route 意向升级工具（本地证据→联网），不再只走 web
        return "tool_upgrade"
    return "save_memory"


# ── 构建图 ──────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)

    # 用 traced 包装每个函数节点，统一记录节点耗时（ms）
    def add(name, fn):
        builder.add_node(name, traced(name, fn))

    # 公共节点
    add("load_memory", N.load_memory)
    add("rewrite_split", N.rewrite_split)
    add("classify", N.classify_intent)
    add("rag_gate", N.rag_gate)
    add("direct_answer", N.direct_answer)
    add("ask_user", N.ask_user)
    add("multi_retrieve", N.multi_retrieve)
    add("reflection", N.reflection)
    # 注：web_fallback 不作为图节点挂边——反思 RETRY 由 tool_upgrade 节点
    # 内部以函数形式调用 N.web_fallback（本地证据不足时联网兜底）。
    add("tool_upgrade", N.tool_upgrade)
    add("save_memory", N.save_memory)

    # 唯一主路径：ReAct（子管线逻辑已下沉为工具，见 src/tools/capabilities.py）
    add("general_agent", N.general_agent)
    # ToolNode 包成普通节点——执行后对 semantic_search 结果做相关性
    # 过滤（节点名不变，service.py 的"执行工具"标签无需同步）；包成普通函数
    # 后也自然获得 traced 节点耗时观测
    add("general_tools", N.general_tools)

    # ── 连线 ──
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "rewrite_split")
    builder.add_edge("rewrite_split", "classify")
    # 路由决策：direct → 直接回答；其余 → rag_gate（保留寒暄双保险）
    builder.add_conditional_edges(
        "classify",
        lambda s: "direct" if s.route == "direct" else "rag",
        {"direct": "direct_answer", "rag": "rag_gate"},
    )
    # RAG 开关：不需要检索 → 直接回答；需要 → 走澄清/检索主路径
    builder.add_conditional_edges(
        "rag_gate",
        lambda s: "no_rag" if not s.rag_needed else "rag",
        {"no_rag": "direct_answer", "rag": "ask_user"},
    )
    builder.add_edge("direct_answer", "reflection")
    # 信息缺口澄清：ask → 直接收尾（等用户补充）；continue → 统一 agent 主路径
    builder.add_conditional_edges(
        "ask_user",
        lambda s: s.ask_user,
        {"ask": END, "continue": "multi_retrieve"},
    )
    builder.add_edge("multi_retrieve", "general_agent")

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
        {"tool_upgrade": "tool_upgrade", "save_memory": "save_memory"},
    )
    builder.add_edge("tool_upgrade", "save_memory")
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
