"""general 节点接线纯单测：会话台账自动登记（不触发 LLM/网络）。"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import ToolMessage

from src.agent.nodes.general import MAX_TOOL_ROUNDS, _ledger_updates, general_tools
from src.agent.state import AgentState


def _msg(name, content):
    return ToolMessage(content=json.dumps(content, ensure_ascii=False), name=name,
                       tool_call_id="c1")


def test_ledger_records_shown_artworks_deduped():
    state = AgentState(user_query="q", shown_artworks=["A"])
    merged = [
        _msg("semantic_search", [{"title": "A"}, {"title": "B", "author": "Monet"}]),
        _msg("exact_lookup", [{"title": "C"}]),
    ]
    updates = _ledger_updates(merged, state)
    assert updates["shown_artworks"] == ["A", "B", "C"]


def test_ledger_records_recommended_artists():
    state = AgentState(user_query="q")
    merged = [
        _msg("recommend_with_exclusions",
             {"features": "f", "liked_artists": [], "candidates": [
                 {"author": "Rubens"}, {"author": "Caravaggio"}, {"author": "Rubens"},
             ]}),
    ]
    updates = _ledger_updates(merged, state)
    assert updates["recommended_artists"] == ["Rubens", "Caravaggio"]


def test_ledger_ignores_unrelated_tools():
    state = AgentState(user_query="q")
    merged = [_msg("web_search", [{"title": "web hit", "url": "x"}])]
    updates = _ledger_updates(merged, state)
    assert updates["shown_artworks"] == []
    assert updates["recommended_artists"] == []


def test_tool_round_cap_blocks_execution():
    from langchain_core.messages import AIMessage

    ai = AIMessage(content="", tool_calls=[
        {"name": "exact_lookup", "args": {"author": "Monet"}, "id": "c1"},
    ])
    state = AgentState(user_query="q", tool_rounds=MAX_TOOL_ROUNDS, messages=[ai])
    with patch("src.agent.nodes.general._guarded_tool_calls") as mock:
        out = general_tools(state)
        mock.assert_not_called()
    assert "LIMIT" in str(out["messages"][0].content)
    assert out["tool_rounds"] == MAX_TOOL_ROUNDS


def test_tool_rounds_increment_on_normal_path():
    from langchain_core.messages import AIMessage

    ai = AIMessage(content="", tool_calls=[
        {"name": "exact_lookup", "args": {"author": "Monet"}, "id": "c9"},
    ])
    state = AgentState(user_query="q", tool_rounds=2, messages=[ai])
    with patch("src.agent.nodes.general._guarded_tool_calls", return_value=([], ["sig1"])):
        out = general_tools(state)
    assert out["tool_rounds"] == 3
    assert out["executed_tool_signatures"] == ["sig1"]


def test_repeat_tool_call_blocked_by_guard():
    from langchain_core.messages import AIMessage

    from src.agent.nodes.general import _guarded_tool_calls

    ai = AIMessage(content="", tool_calls=[
        {"name": "semantic_search", "args": {"query": "莫奈手稿"}, "id": "c1"},
    ])
    sig = 'semantic_search:{"query": "莫奈手稿"}'
    state = AgentState(user_query="q", executed_tool_signatures=[sig], messages=[ai])
    merged, new_sigs = _guarded_tool_calls(state, None)
    assert "REPEAT" in str(merged[0].content)
    assert "read_page_image" in str(merged[0].content)
    assert new_sigs == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] general_wiring 全部 {len(fns)} 个单测通过")
