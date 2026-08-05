"""
意图树 + 打分式意图分类器（借鉴 ragent `DefaultIntentClassifier`）。

设计要点：
- 意图叶子节点分三类：
  - capability：路由到现有专用分支（comparison / timeline / recommendation / general）
  - tool：建议调用的工具（image_lookup / read_page_image / web_search ...）
  - system：无需检索直接回答的场景（打招呼等）
- 分类器把所有叶子一次性发给 LLM 打分，输出
  `[{"id": "...", "score": 0.9, "reason": "..."}]`，按分数降序；
- 主意图 = 分数最高的 capability 叶子（低于阈值则回落 general）；
- 容错：LLM 调用失败 / JSON 畸形 / 未知 id 一律回落，不影响主流程。

用法：
    from src.agent.intent_tree import classify_intents
    scores, primary = classify_intents("对比莫奈和梵高的色彩")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.utils.json_utils import parse_json


@dataclass
class IntentLeaf:
    """意图树叶子节点。"""

    id: str
    path: str
    description: str
    kind: str  # "capability" | "tool" | "system"
    examples: list[str] = field(default_factory=list)
    tool_name: Optional[str] = None  # kind=tool 时挂的工具名
    threshold: float = 0.3


@dataclass
class NodeScore:
    """某个意图叶子的一次打分结果。"""

    leaf: IntentLeaf
    score: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.leaf.id,
            "path": self.leaf.path,
            "kind": self.leaf.kind,
            "score": round(self.score, 4),
            "reason": self.reason,
            "tool_name": self.leaf.tool_name,
        }


# ------------------------------------------------------------------ #
# 意图树定义（静态 JSON 即可扩展，暂不引入数据库）                          #
# ------------------------------------------------------------------ #

INTENT_LEAVES: list[IntentLeaf] = [
    IntentLeaf(
        id="comparison",
        path="capability > comparison",
        description="对比两个或多个画家/画作/艺术风格之间的差异",
        kind="capability",
        examples=["对比莫奈和梵高的色彩", "《星月夜》和《向日葵》有什么不同"],
    ),
    IntentLeaf(
        id="timeline",
        path="capability > timeline",
        description="按时间顺序梳理某位画家或某个流派的风格演变",
        kind="capability",
        examples=["梳理伦勃朗的风格演变", "透纳不同时期的画风变化"],
    ),
    IntentLeaf(
        id="recommendation",
        path="capability > recommendation",
        description="基于用户表达的喜好，推荐其他画家或作品",
        kind="capability",
        examples=["我喜欢梵高浓烈的风格，还会喜欢谁", "推荐类似卡拉瓦乔的画家"],
    ),
    IntentLeaf(
        id="general",
        path="capability > general",
        description="其余所有情况：单一事实查询、知识问答、看画分析、找某画家的画",
        kind="capability",
        examples=["梵高有哪些作品", "什么是巴洛克", "分析《星月夜》的构图"],
    ),
    IntentLeaf(
        id="tool_image_lookup",
        path="tool > image_lookup",
        description="用户要'看'某幅画，或要求对构图/色彩/笔触做视觉层面的分析",
        kind="tool",
        tool_name="image_lookup",
        examples=["帮我分析《星月夜》的构图特点", "从视觉角度介绍卡拉瓦乔的画"],
    ),
    IntentLeaf(
        id="tool_read_page_image",
        path="tool > read_page_image",
        description="需要读取用户上传 PDF 中整页图像的实际内容",
        kind="tool",
        tool_name="read_page_image",
        examples=["看看我上传画册第3页的内容"],
    ),
    IntentLeaf(
        id="tool_web_search",
        path="tool > web_search",
        description="本地库查不到所需信息，或问题明显超出西方艺术馆藏范围",
        kind="tool",
        tool_name="web_search",
        examples=["某幅画不在数据库里，查它的最新资料"],
    ),
    IntentLeaf(
        id="system_greeting",
        path="system > greeting",
        description="打招呼、闲聊、自我介绍等无需检索即可回答的寒暄",
        kind="system",
        examples=["你好", "你是谁", "谢谢"],
    ),
    IntentLeaf(
        id="system_knowledge",
        path="system > knowledge",
        description=(
            "常识/定义/算术类知识问答（如'什么是线性透视''1+1等于几'），"
            "无需检索即可直接回答"
        ),
        kind="system",
        examples=["什么是线性透视", "油画颜料和丙烯颜料有什么区别"],
    ),
]

_LEAF_BY_ID: dict[str, IntentLeaf] = {leaf.id: leaf for leaf in INTENT_LEAVES}
_CAPABILITY_IDS = {leaf.id for leaf in INTENT_LEAVES if leaf.kind == "capability"}

# capability 叶子 → 建议优先调用的工具（子管线逻辑下沉后的对应物）
_CAPABILITY_TOOL = {
    "comparison": "compare_subjects",
    "timeline": "timeline_by_periods",
    "recommendation": "recommend_with_exclusions",
    "general": "semantic_search / exact_lookup",
}


# ------------------------------------------------------------------ #
# 工具叶子自动对齐（T2）：GENERAL_TOOLS → 意图树 tool 叶子 1:1           #
# ------------------------------------------------------------------ #

_TOOL_LEAVES_CACHE: Optional[list[IntentLeaf]] = None


def get_tool_leaves() -> list[IntentLeaf]:
    """从 GENERAL_TOOLS + register_skills() 自动生成 tool 叶子（懒加载）。

    与工具带保持 1:1 对齐：新增工具后无需手工维护意图树。
    description 取工具 docstring 首段（压缩 ≤180 字符，控制分类 prompt 体积）。
    """
    global _TOOL_LEAVES_CACHE
    if _TOOL_LEAVES_CACHE is None:
        from src.agent.nodes.general import GENERAL_TOOLS

        leaves: list[IntentLeaf] = []
        for t in GENERAL_TOOLS:
            name = getattr(t, "name", "") or ""
            if not name:
                continue
            desc = str(getattr(t, "description", "") or "").strip()
            first_line = desc.splitlines()[0] if desc else f"调用工具 {name}"
            if len(first_line) > 180:
                first_line = first_line[:180] + "..."
            leaves.append(
                IntentLeaf(
                    id=f"tool_{name}",
                    path=f"tool > {name}",
                    description=first_line,
                    kind="tool",
                    tool_name=name,
                )
            )
        _TOOL_LEAVES_CACHE = leaves
    return _TOOL_LEAVES_CACHE


def all_leaves() -> list[IntentLeaf]:
    """静态叶子 + 自动生成工具叶子（工具叶子优先，保留静态 examples）。"""
    by_id = {leaf.id: leaf for leaf in INTENT_LEAVES}
    for tl in get_tool_leaves():
        existing = by_id.get(tl.id)
        by_id[tl.id] = (
            IntentLeaf(
                id=tl.id,
                path=tl.path,
                description=tl.description,
                kind="tool",
                examples=existing.examples,
                tool_name=tl.tool_name,
            )
            if existing
            else tl
        )
    return list(by_id.values())


# ------------------------------------------------------------------ #
# Prompt 构建与解析                                                     #
# ------------------------------------------------------------------ #

CLASSIFIER_SYSTEM_PROMPT = """你是艺术领域 Agent 的意图识别模块。下面列出了所有可用的意图叶子节点，
请判断用户问题与每个叶子的相关程度，为【每一个】叶子输出一个分数。

意图叶子列表：
{intent_list}

要求：
1. 输出 JSON 数组，元素格式：{{"id": "...", "score": 0.0-1.0, "reason": "一句话理由"}}；
2. 必须覆盖列表中的所有 id，不得省略、不得自创 id；
3. score 表示该意图与问题的匹配程度：高度匹配给 0.9 以上，部分相关给 0.5 左右，无关给 0.1 以下；
4. 如果问题只提到某个具体对象（如"OA 系统"），不要给语义相近但无关的意图高分；
5. 只输出 JSON，不要解释，不要 markdown 代码块。

用户问题：
{user_query}"""


def build_classifier_prompt() -> str:
    """按当前意图树生成分类 prompt 的叶子列表段。"""
    lines: list[str] = []
    for leaf in all_leaves():
        lines.append(f"- id={leaf.id}")
        lines.append(f"  path={leaf.path}")
        lines.append(f"  description={leaf.description}")
        lines.append(f"  type={leaf.kind.upper()}")
        if leaf.examples:
            lines.append(f"  examples={' / '.join(leaf.examples)}")
        lines.append("")
    return "\n".join(lines)


def parse_scores(raw: str) -> list[NodeScore]:
    """鲁棒解析 LLM 打分输出；任何畸形/未知 id 都安全跳过。"""
    data = parse_json(raw)

    # 容错：模型可能在外面又包了一层 {"results": [...]}
    if isinstance(data, dict):
        data = data.get("results")
    if not isinstance(data, list):
        return []

    scores: list[NodeScore] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        leaf_id = item.get("id")
        score = item.get("score")
        leaf = get_leaf(str(leaf_id)) if leaf_id is not None else None
        if leaf is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        scores.append(NodeScore(leaf=leaf, score=score, reason=str(item.get("reason") or "")))
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def top_scores(scores: list[NodeScore], top_n: int = 3, min_score: float = 0.3) -> list[NodeScore]:
    """过滤低分并截断，供观察与后续多意图并行使用。"""
    return [s for s in scores if s.score >= min_score][:top_n]


def _as_score_tuple(s) -> tuple[str, str, float, Optional[str]]:
    """兼容 NodeScore 与 state.intent_scores 的 dict 形态。"""
    if isinstance(s, NodeScore):
        return s.leaf.id, s.leaf.kind, s.score, s.leaf.tool_name
    if isinstance(s, dict):
        try:
            score = float(s.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return str(s.get("id") or ""), str(s.get("kind") or ""), score, s.get("tool_name")
    return "", "", 0.0, None


def intent_tool_suggestions(
    scores,
    top_n: int = 3,
    min_score: float = 0.3,
) -> list[str]:
    """把高分意图叶子转成"建议优先考虑的工具"提示，供 agent prompt 注入。

    - tool 叶子直接用其 tool_name；
    - capability 叶子映射到下沉后的能力工具；
    - system 叶子（寒暄等）不产出工具建议。
    接受 NodeScore 列表或 state.intent_scores 的 dict 列表。
    """
    ranked = sorted(scores, key=lambda s: _as_score_tuple(s)[2], reverse=True)
    hints: list[str] = []
    for s in ranked:
        leaf_id, kind, score, tool_name = _as_score_tuple(s)
        if score < min_score:
            continue
        if kind == "system":
            continue
        tool = tool_name or _CAPABILITY_TOOL.get(leaf_id)
        if tool:
            hints.append(f"{tool}（意图 {leaf_id}，{score:.2f}）")
        if len(hints) >= top_n:
            break
    return hints


def _primary_intent(scores: list[NodeScore]) -> str:
    """取分数最高的 capability 叶子作为路由意图；低于阈值或无结果回落 general。"""
    for s in scores:
        if s.leaf.kind == "capability":
            if s.score >= s.leaf.threshold:
                return s.leaf.id
            break
    return "general"


def _primary_route(scores: list[NodeScore]) -> tuple[str, str]:
    """从 LLM 打分推导路由决策（classify 的 LLM 路径）。

    优先级：
      1. system 叶子（寒暄/常识定义）≥0.7 且高于其他 → direct；
      2. tool 叶子 ≥0.6 且高于最佳 capability → tool:<name>；
      3. capability 主意图 → comparison/timeline/recommendation；
      4. general → rag（默认走检索）；
      5. 无有效分数 → rag 兜底。
    """
    if not scores:
        return "rag", "无有效打分，默认走检索"
    best_system = next(
        (s for s in scores if s.leaf.kind == "system" and s.score >= 0.7),
        None,
    )
    best_cap = next(
        (s for s in scores if s.leaf.kind == "capability"),
        None,
    )
    best_tool = next(
        (
            s for s in scores
            if s.leaf.kind == "tool" and s.score >= 0.6
            and (best_cap is None or s.score > best_cap.score)
        ),
        None,
    )
    if best_system is not None and (
        best_cap is None or best_system.score >= best_cap.score
    ):
        return "direct", f"system:{best_system.leaf.id}={best_system.score:.2f}"
    if best_tool is not None:
        return f"tool:{best_tool.leaf.tool_name}", (
            f"tool:{best_tool.leaf.id}={best_tool.score:.2f}"
        )
    if best_cap is None:
        return "rag", "无 capability 打分，默认检索"
    if best_cap.leaf.id == "general":
        return "rag", f"general={best_cap.score:.2f}"
    return best_cap.leaf.id, f"{best_cap.leaf.id}={best_cap.score:.2f}"


def classify_intents(
    query: str,
    llm: Optional[Callable[[str], str]] = None,
) -> tuple[list[NodeScore], str, str, str]:
    """对问题做意图打分，返回 (全部打分, 主意图, 路由决策, 路由理由)。

    llm 可注入（默认 get_deterministic_llm），便于单测。任何失败都回落：
    scores=[] + primary="general" + route="rag"。
    """
    if llm is None:
        from src.utils.llm import get_deterministic_llm

        def _default_llm(prompt: str) -> str:
            return get_deterministic_llm().invoke(prompt).content

        llm = _default_llm

    prompt = CLASSIFIER_SYSTEM_PROMPT.format(
        intent_list=build_classifier_prompt(),
        user_query=query,
    )
    try:
        raw = llm(prompt)
    except Exception:
        return [], "general", "rag", "LLM 分类失败，默认检索"
    scores = parse_scores(raw)
    route, reason = _primary_route(scores)
    return scores, _primary_intent(scores), route, reason


def get_leaf(leaf_id: str) -> Optional[IntentLeaf]:
    """按 id 取叶子节点。"""
    leaf = _LEAF_BY_ID.get(leaf_id)
    if leaf is not None:
        return leaf
    return next((l for l in get_tool_leaves() if l.id == leaf_id), None)


def all_capability_ids() -> set[str]:
    """现有专用分支意图 id（兼容旧路由）。"""
    return set(_CAPABILITY_IDS)
