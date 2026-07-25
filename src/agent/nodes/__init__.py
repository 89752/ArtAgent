"""ArtAgent 节点集合（混合架构）。"""

from src.agent.nodes.common import (
    contextualize,
    classify_intent,
    load_memory,
    reflection,
    web_fallback,
    save_memory,
    collect_artworks,
    parse_json,
)
from src.agent.nodes.comparison import (
    comparison_decompose,
    comparison_retrieve,
    comparison_synthesize,
)
from src.agent.nodes.timeline import (
    timeline_extract_subject,
    timeline_gather_periods,
    timeline_synthesize,
)
from src.agent.nodes.recommendation import (
    recommendation_extract_features,
    recommendation_feature_search,
    recommendation_relevance_filter,
    recommendation_synthesize,
)
from src.agent.nodes.general import (
    general_agent,
    general_should_continue,
    GENERAL_TOOLS,
)

__all__ = [
    "contextualize",
    "classify_intent",
    "load_memory",
    "reflection",
    "web_fallback",
    "save_memory",
    "collect_artworks",
    "parse_json",
    "comparison_decompose",
    "comparison_retrieve",
    "comparison_synthesize",
    "timeline_extract_subject",
    "timeline_gather_periods",
    "timeline_synthesize",
    "recommendation_extract_features",
    "recommendation_feature_search",
    "recommendation_relevance_filter",
    "recommendation_synthesize",
    "general_agent",
    "general_should_continue",
    "GENERAL_TOOLS",
]
