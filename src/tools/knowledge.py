"""画家知识查询工具（query_painter_knowledge）。

返回画家在当前核心库中的结构化统计信息。

设计原则：工具只返回结构化数据，不在内部再调一次 LLM
把结果包装成一段话——组织自然语言回答的活儿留给外层 general_agent，
由它把统计数据与自己的艺术史知识结合后统一生成。
"""

from langchain_core.tools import tool
from dotenv import load_dotenv

from src.data.access import fuzzy_match

load_dotenv()


@tool
def query_painter_knowledge(painter_name: str) -> dict:
    """
    查询画家在 SemArt 数据集中的结构化统计信息。

    适用场景：
      - 需要某画家的作品数量、所属流派、活跃时期、常用技法、代表作清单
      - 回答画家生平/风格/地位类问题时，先用本工具取数据依据，
        再结合你自己的艺术史知识组织回答

    Args:
        painter_name: 画家姓名（支持部分匹配，如 "Gogh"、"Monet"）

    Returns:
        结构化统计：found / matched_author / works_count / main_schools /
        active_timeframes / common_techniques / sample_works
    """
    # 2026-08-02：按当前生效数据源（semart / core）的角色列统计，不再绑定 SemArt
    from src.retrieval.hybrid import get_hybrid_retriever
    from src.retrieval.structured_retriever import get_structured_retriever

    dataset_id = get_hybrid_retriever().active_dataset
    retriever = get_structured_retriever(dataset_id)
    schema = retriever.schema
    df = retriever.df
    works = fuzzy_match(df, schema.entity_col, painter_name)

    if works.empty:
        return {
            "painter": painter_name,
            "found": False,
            "works_count": 0,
            "note": f"{dataset_id} 中未收录该画家的作品，请基于自身知识回答或考虑 web_search。",
        }

    return {
        "painter": painter_name,
        "found": True,
        "matched_author": works[schema.entity_col].value_counts().index[0],
        "works_count": len(works),
        "main_schools": (
            works[schema.school_col].value_counts().head(3).index.tolist()
            if schema.school_col else []
        ),
        "active_timeframes": (
            works[schema.group_axis_col].value_counts().head(3).index.tolist()
            if schema.group_axis_col else []
        ),
        "common_techniques": (
            works[schema.technique_col].value_counts().head(3).to_dict()
            if schema.technique_col else {}
        ),
        "sample_works": (
            works[schema.title_col].head(5).tolist()
            if schema.title_col else []
        ),
        "sample_work_images": (
            [
                {
                    "title": str(row[schema.title_col]),
                    "image_file": str(row.get(schema.image_col) or ""),
                }
                for _, row in works.head(5).iterrows()
            ]
            if schema.title_col and schema.image_col else []
        ),
    }
