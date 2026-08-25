"""工具带统一注册表：所有可调用工具的唯一定义点。

general 节点与技能系统都从这里取工具，新增工具只需在此登记一次。
"""

from src.tools.aggregate_stats import aggregate_stats
from src.tools.collections import (
    delete_collection,
    get_collection,
    list_collections,
    list_preferences,
    rename_collection,
    save_collection,
)
from src.tools.color_analysis import color_analysis
from src.tools.compare_images import compare_images
from src.tools.delegate import delegate_task
from src.tools.image_lookup import image_lookup
from src.tools.knowledge import query_painter_knowledge
from src.tools.memory import forget, recall, remember
from src.tools.museum_search import museum_search
from src.tools.page_reader import read_page_image
from src.tools.retrieval import (
    agentic_retrieve,
    exact_lookup,
    semantic_search,
)
from src.tools.web_search import web_search
from src.tools.wiki_lookup import wiki_lookup
from src.tools.user_image import analyze_user_artwork, read_user_image

GENERAL_TOOLS = [
    semantic_search,
    agentic_retrieve,
    exact_lookup,
    query_painter_knowledge,
    image_lookup,
    read_page_image,
    web_search,
    remember,
    recall,
    forget,
    save_collection,
    list_collections,
    get_collection,
    delete_collection,
    rename_collection,
    list_preferences,
    color_analysis,
    aggregate_stats,
    compare_images,
    museum_search,
    wiki_lookup,
    read_user_image,
    analyze_user_artwork,
    delegate_task,
]

TOOL_BY_NAME: dict[str, object] = {t.name: t for t in GENERAL_TOOLS}
