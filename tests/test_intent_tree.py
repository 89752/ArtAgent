"""意图树打分分类器纯单测：解析容错、阈值过滤、主意图选择、LLM 失败回落。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.intent_tree import (
    INTENT_LEAVES,
    NodeScore,
    classify_intents,
    get_leaf,
    intent_tool_suggestions,
    parse_scores,
    top_scores,
)


def _raw_with_fence():
    return """```json
[{"id": "comparison", "score": 0.92, "reason": "对比"},
 {"id": "timeline", "score": 0.1, "reason": "无关"},
 {"id": "unknown-xx", "score": 0.99},
 {"id": "tool_image_lookup", "score": 0.75}]
```"""


def test_parse_scores_strips_fence_and_sorts_desc():
    scores = parse_scores(_raw_with_fence())
    ids = [s.leaf.id for s in scores]
    assert ids == ["comparison", "tool_image_lookup", "timeline"]  # 未知 id 被跳过
    assert scores[0].score == 0.92


def test_parse_scores_accepts_nested_results_object():
    raw = '{"results": [{"id": "general", "score": 0.8, "reason": "x"}]}'
    scores = parse_scores(raw)
    assert [s.leaf.id for s in scores] == ["general"]


def test_parse_scores_tolerates_garbage():
    assert parse_scores("") == []
    assert parse_scores("not json at all") == []
    assert parse_scores('{"no": "array"}') == []
    assert parse_scores('[{"id": "general"}]') == []  # 缺 score
    assert parse_scores('[{"id": "general", "score": "abc"}]') == []  # score 非法


def test_top_scores_filters_threshold():
    scores = parse_scores(_raw_with_fence())
    kept = top_scores(scores, top_n=1, min_score=0.3)
    assert [s.leaf.id for s in kept] == ["comparison"]


def test_classify_intents_primary_is_top_capability():
    def fake_llm(prompt):
        assert "comparison" in prompt  # 叶子列表进了 prompt
        return _raw_with_fence()

    scores, primary = classify_intents("对比莫奈和梵高", llm=fake_llm)
    assert primary == "comparison"
    assert scores[0].leaf.id == "comparison"


def test_classify_intents_falls_back_when_capability_below_threshold():
    raw = '[{"id": "comparison", "score": 0.1}, {"id": "tool_web_search", "score": 0.95}]'

    scores, primary = classify_intents("某问题", llm=lambda p: raw)
    assert primary == "general"
    assert scores[0].leaf.id == "tool_web_search"


def test_classify_intents_llm_failure_falls_back_to_general():
    def boom(prompt):
        raise RuntimeError("llm down")

    scores, primary = classify_intents("任何问题", llm=boom)
    assert primary == "general"
    assert scores == []


def test_tree_has_four_capability_leaves_and_tool_leaves():
    kinds = {leaf.kind for leaf in INTENT_LEAVES}
    assert kinds == {"capability", "tool", "system"}
    capability_ids = {leaf.id for leaf in INTENT_LEAVES if leaf.kind == "capability"}
    assert capability_ids == {"comparison", "timeline", "recommendation", "general"}
    tool_leaves = [leaf for leaf in INTENT_LEAVES if leaf.kind == "tool"]
    assert all(leaf.tool_name for leaf in tool_leaves)


def test_node_score_to_dict():
    s = NodeScore(leaf=INTENT_LEAVES[0], score=0.91234, reason="r")
    d = s.to_dict()
    assert d["score"] == 0.9123
    assert d["id"] == INTENT_LEAVES[0].id
    assert d["kind"] == INTENT_LEAVES[0].kind


def _leaf(kind: str, leaf_id: str, score: float, tool_name=None) -> dict:
    return {
        "id": leaf_id, "path": f"{kind} > {leaf_id}", "kind": kind,
        "score": score, "reason": "", "tool_name": tool_name,
    }


def test_intent_suggestions_maps_capability_and_tool():
    scores = [
        _leaf("capability", "comparison", 0.93),
        _leaf("tool", "tool_web_search", 0.72, tool_name="web_search"),
        _leaf("system", "system_greeting", 0.95),
        _leaf("capability", "timeline", 0.1),
    ]
    hints = intent_tool_suggestions(scores)
    assert hints[0].startswith("compare_subjects")
    assert "web_search" in hints[1]
    assert len(hints) == 2  # system 跳过、低分 timeline 过滤


def test_intent_suggestions_accepts_nodescore_input():
    scores = [NodeScore(leaf=get_leaf("recommendation"), score=0.88)]
    hints = intent_tool_suggestions(scores)
    assert "recommend_with_exclusions" in hints[0]


def test_intent_suggestions_empty_on_no_scores():
    assert intent_tool_suggestions([]) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] intent_tree 全部 {len(fns)} 个单测通过")
