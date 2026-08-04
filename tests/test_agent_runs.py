"""可观测轨迹与指标测试（G8/2.4）。"""

import tempfile
from pathlib import Path

import pytest

from src.observability import runs


@pytest.fixture(autouse=True)
def _isolated_db():
    tmp = Path(tempfile.mkdtemp(prefix="artagent_runs_test_"))
    runs._reset_for_tests(tmp / "obs.db")
    yield
    runs._reset_for_tests()


def test_record_and_list():
    rid = runs.record_run(
        request_id="r1", session_id="s1", intent="comparison",
        steps=[{"node": "classify"}], tools=["semantic_search", "exact_lookup"],
        context_chars=4000, tool_rounds=2, latency_ms=1234.5,
        final_answer_len=200, reflection_triggered=True, web_fallback=True,
    )
    assert rid >= 1
    rows = runs.list_runs()
    assert len(rows) == 1
    assert rows[0]["intent"] == "comparison"
    assert rows[0]["tools"] == ["semantic_search", "exact_lookup"]
    assert rows[0]["latency_ms"] == 1234.5


def test_metrics_summary(monkeypatch):
    monkeypatch.setenv("COST_PER_1K_INPUT_TOKENS", "1.0")
    monkeypatch.setenv("COST_PER_1K_OUTPUT_TOKENS", "2.0")
    for i in range(10):
        runs.record_run(
            session_id=f"s{i}", intent="general", tools=["web_search"],
            context_chars=1000, tool_rounds=1, latency_ms=float(i * 10),
            final_answer_len=100, web_fallback=(i % 2 == 0),
            error=("x" if i == 0 else ""),
        )
    m = runs.metrics(limit=10)
    assert m["count"] == 10
    assert m["latency_ms"]["p50"] == pytest.approx(45.0, abs=1.0)
    assert m["latency_ms"]["p95"] >= 85.0
    assert m["web_fallback_rate"] == pytest.approx(0.5)
    assert m["error_rate"] == pytest.approx(0.1)
    assert m["tool_calls"].get("web_search") == 10
    assert m["est_cost_total"] > 0
