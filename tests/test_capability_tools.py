"""能力工具（compare_subjects / timeline_by_periods / recommend_with_exclusions）：
schema 与工具守卫集成、注册到 general 工具列表的纯单测（不触发网络）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.nodes.general import GENERAL_TOOLS, TOOL_BY_NAME, _tool_schema
from src.tools.capabilities import (
    compare_subjects,
    group_by_artist,
    recommend_with_exclusions,
    timeline_by_periods,
)
from src.tools.guard import validate_args


def test_three_tools_registered_in_general():
    names = {t.name for t in GENERAL_TOOLS}
    assert {"compare_subjects", "timeline_by_periods", "recommend_with_exclusions"} <= names


def test_compare_schema_required_subjects():
    tool = TOOL_BY_NAME["compare_subjects"]
    schema = _tool_schema(tool)
    assert validate_args(schema, {"subjects": ["Monet", "van Gogh"]}).status == "SUCCESS"
    d = validate_args(schema, {})
    assert d.status == "NEED_CLARIFICATION"
    assert "subjects" in d.missing
    d2 = validate_args(schema, {"subjects": ["Monet"], "dimensions": ["color use"]})
    assert d2.status == "SUCCESS"


def test_timeline_schema_required_subject():
    tool = TOOL_BY_NAME["timeline_by_periods"]
    schema = _tool_schema(tool)
    assert validate_args(schema, {"subject": "Rembrandt"}).status == "SUCCESS"
    d = validate_args(schema, {})
    assert d.status == "NEED_CLARIFICATION"
    assert "subject" in d.missing


def test_recommend_schema_required_both():
    tool = TOOL_BY_NAME["recommend_with_exclusions"]
    schema = _tool_schema(tool)
    d = validate_args(schema, {"preference": "浓烈奔放的风格"})
    assert d.status == "NEED_CLARIFICATION"
    assert set(d.missing) == {"exclude_artists"}
    ok = validate_args(
        schema, {"preference": "浓烈奔放", "exclude_artists": ["van Gogh"]}
    )
    assert ok.status == "SUCCESS"


def test_tool_object_names():
    assert compare_subjects.name == "compare_subjects"
    assert timeline_by_periods.name == "timeline_by_periods"
    assert recommend_with_exclusions.name == "recommend_with_exclusions"


def test_group_by_artist_per_artist_top_titles():
    cands = [
        {"author": "Rubens", "title": "A"},
        {"author": "Rubens", "title": "B"},
        {"author": "Rubens", "title": "C"},
        {"author": "Caravaggio", "title": "D"},
        {"author": "", "title": "no-author"},
    ]
    out = group_by_artist(cands, per_artist=2)
    assert out == {"Rubens": ["A", "B"], "Caravaggio": ["D"]}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] capability_tools 全部 {len(fns)} 个单测通过")
