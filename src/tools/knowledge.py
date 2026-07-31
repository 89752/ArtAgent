"""
Tool 3: Painter Knowledge Query Tool

返回画家在 SemArt 数据集中的结构化统计信息。

设计原则（Stage 1 起）：工具只返回结构化数据，不在内部再调一次 LLM
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
    from src.data.loader import get_dataset

    df = get_dataset().all
    works = fuzzy_match(df, "AUTHOR", painter_name)

    if works.empty:
        return {
            "painter": painter_name,
            "found": False,
            "works_count": 0,
            "note": "SemArt 数据集中未收录该画家的作品，请基于自身知识回答或考虑 web_search。",
        }

    return {
        "painter": painter_name,
        "found": True,
        "matched_author": works["AUTHOR"].value_counts().index[0],
        "works_count": len(works),
        "main_schools": works["SCHOOL"].value_counts().head(3).index.tolist(),
        "active_timeframes": works["TIMEFRAME"].value_counts().head(3).index.tolist(),
        "common_techniques": works["TECHNIQUE"].value_counts().head(3).to_dict(),
        "sample_works": works["TITLE"].head(5).tolist(),
    }
