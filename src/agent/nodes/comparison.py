"""
场景1：跨维度风格对比子管线。

decompose → retrieve → synthesize
  - decompose: 抽取对比对象 + 对比维度
  - retrieve:  对每个对象按"对象+维度"做语义检索，分组存储
  - synthesize: 逐维度对比，引用评论依据
"""

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.agent.prompts import COMPARISON_DECOMPOSE_PROMPT, COMPARISON_SYNTHESIZE_PROMPT
from src.agent.nodes.common import parse_json, collect_artworks
from src.utils.llm import get_llm, get_deterministic_llm
from src.utils.logging_config import get_logger, log_event

logger = get_logger("comparison")


def comparison_decompose(state: AgentState) -> dict:
    """抽取对比对象与维度，生成每个对象的子查询。"""
    prompt = COMPARISON_DECOMPOSE_PROMPT.format(user_query=state.user_query)
    raw = get_deterministic_llm().invoke(prompt).content
    parsed = parse_json(raw) or {}

    subjects = parsed.get("subjects", []) if isinstance(parsed, dict) else []
    dimensions = parsed.get("dimensions", []) if isinstance(parsed, dict) else []

    # 兜底：至少要有对象，否则退化用原始 query
    if not subjects:
        subjects = [state.user_query]
    if not dimensions:
        dimensions = ["style", "color", "technique"]

    dim_str = " ".join(dimensions)
    sub_queries = [f"{s} {dim_str} painting style characteristics" for s in subjects]

    log_event(logger, "decompose", subjects=subjects, dimensions=dimensions)
    return {
        "subjects": subjects,
        "sub_queries": sub_queries,
        "current_step": "comparison_decompose",
    }


def comparison_retrieve(state: AgentState) -> dict:
    """对每个对象分别语义检索，按对象分组存储。"""
    from src.tools.retrieval import semantic_search

    docs_by_subject: dict[str, list[dict]] = {}
    for subject, query in zip(state.subjects, state.sub_queries):
        try:
            results = semantic_search.invoke({"query": query, "top_k": 4})
        except Exception as e:
            logger.warning("[retrieve] semantic_search failed for %s: %s", subject, e)
            results = []
        docs_by_subject[subject] = results

    artworks = collect_artworks(docs_by_subject)
    log_event(
        logger, "retrieve",
        hits_per_subject={s: len(d) for s, d in docs_by_subject.items()},
        artworks=artworks,
    )
    return {
        "retrieved_docs": docs_by_subject,
        "artworks": artworks,
        "current_step": "comparison_retrieve",
    }


def comparison_synthesize(state: AgentState) -> dict:
    """逐维度组织对比，引用检索到的评论。"""
    # 构造分组证据文本
    blocks = []
    for subject, docs in state.retrieved_docs.items():
        lines = [f"【{subject}】"]
        for d in docs:
            snippet = d.get("description_snippet", "")
            lines.append(
                f"  - {d.get('title', '')} ({d.get('date', '')}): {snippet}"
            )
        blocks.append("\n".join(lines))
    grouped_evidence = "\n\n".join(blocks) if blocks else "(无检索结果)"

    prompt = COMPARISON_SYNTHESIZE_PROMPT.format(
        subjects="、".join(state.subjects),
        user_query=state.user_query,
        grouped_evidence=grouped_evidence,
    )
    answer = get_llm(0.4).invoke(prompt).content
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "current_step": "comparison_synthesize",
    }
