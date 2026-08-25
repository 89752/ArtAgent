"""Evaluation-set contract checks: no LLM, network, or fixture execution."""

import json
from pathlib import Path
from types import SimpleNamespace

from eval.agent_eval_v2 import _check_behavior, _check_state, _optional_case_reason, _safe_invoke, select_cases
from src.agent.nodes.general import TOOL_BY_NAME


CASES_PATH = Path(__file__).resolve().parent.parent / "eval" / "sets" / "cases.json"


def _args(**overrides):
    values = {
        "answers": None, "facts": False, "tools": False, "behavior": False,
        "adversarial": False, "include_live": False,
        "include_artifact_fixtures": False, "include_multi_agent": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_eval_cases_reference_registered_tools_and_have_unique_ids():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    ids = [str(case["id"]) for case in cases]
    assert len(ids) == len(set(ids))
    requested = {tool for case in cases for tool in case.get("expected_tools", [])}
    assert requested <= set(TOOL_BY_NAME)


def test_stateful_cases_seed_their_own_preconditions_for_chunked_runs():
    cases = {str(case["id"]): case for case in json.loads(CASES_PATH.read_text(encoding="utf-8"))}
    for cid in ("c-034", "c-042", "c-043", "c-044", "c-046", "c-047"):
        assert cases[cid].get("setup"), f"{cid} must not depend on a previous case"


def test_optional_capabilities_are_excluded_from_the_core_acceptance_set():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    excluded = {case["id"]: _optional_case_reason(case, _args()) for case in cases}
    assert excluded["c-035"] == "live_web"
    assert excluded["c-041"] == "artifact_fixture"
    assert excluded["c-052"] == "artifact_fixture"
    assert excluded["c-054"] == "multi_agent"
    core = [case for case in cases if not _optional_case_reason(case, _args())]
    assert len(select_cases(core, _args())) == len(core)


def test_eval_invocation_binds_eval_identity():
    from src.memory.memory_items import get_memory_user_id

    class Graph:
        def invoke(self, payload, config):
            assert payload["user_id"] == "eval-test"
            assert get_memory_user_id() == "eval-test"
            return {"final_answer": "ok"}

    assert _safe_invoke(Graph(), "q", "case") == {"final_answer": "ok"}


def test_answer_and_deleted_memory_assertions_are_not_keyword_only(monkeypatch):
    ok, failures = _check_behavior(
        {"behavior": {"answer_contains_all": ["印象派"]}},
        {"messages": [], "final_answer": "目前没有收藏清单"},
    )
    assert not ok and "回答缺少必含内容：印象派" in failures

    monkeypatch.setattr(
        "src.memory.memory_items.list_memories",
        lambda _user: [{"content": "我喜欢巴洛克风格"}],
    )
    assert "记忆表仍含应删除关键词：巴洛克" in _check_state({"memory_absent": ["巴洛克"]})
