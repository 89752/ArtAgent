"""工具执行治理测试：超时 / 重试 / 失败包装。"""

import json
import time

import pytest

from src.utils import governance


class _FakeTool:
    name = "fake"

    def __init__(self, result="ok", fail_times=0, sleep=0):
        self.result = result
        self.fail_times = fail_times
        self.sleep = sleep
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        if self.sleep:
            time.sleep(self.sleep)
        if self.calls <= self.fail_times:
            raise RuntimeError("boom")
        return self.result


def test_run_with_timeout_ok():
    assert governance.run_with_timeout(lambda: 42, 2) == 42


def test_run_with_timeout_raises():
    with pytest.raises(governance.ToolTimeout):
        governance.run_with_timeout(lambda: time.sleep(5), 0.2)


def test_governed_invoke_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("TOOL_RETRIES", "2")
    tool = _FakeTool(result="done", fail_times=2)
    out = governance.governed_invoke(tool, {})
    assert out == "done"
    assert tool.calls == 3


def test_governed_invoke_returns_error_json(monkeypatch):
    monkeypatch.setenv("TOOL_RETRIES", "0")
    tool = _FakeTool(fail_times=99)
    out = governance.governed_invoke(tool, {})
    data = json.loads(out)
    assert data["status"] == "TOOL_ERROR"
    assert data["tool"] == "fake"


def test_governed_invoke_timeout_returns_error(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_SEC", "0.2")
    monkeypatch.setenv("TOOL_RETRIES", "0")
    tool = _FakeTool(sleep=5)
    out = governance.governed_invoke(tool, {})
    assert json.loads(out)["status"] == "TOOL_ERROR"


def test_truncate_payload_preserves_shape():
    payload = {"title": "short", "items": [{"a": "y" * 1000}, {"b": 2}]}
    shrunk = governance._truncate_payload(payload, limit=200)
    text = json.dumps(shrunk, ensure_ascii=False)
    assert len(text) <= 600  # 近似预算，允许序列化开销
    assert "截断" in str(shrunk["items"][0]["a"])
    assert shrunk["items"][-1].get("truncated") is True


def test_governed_invoke_truncates_long_output(monkeypatch):
    monkeypatch.setenv("TOOL_OUTPUT_MAX_CHARS", "200")
    tool = _FakeTool(result="x" * 5000)
    out = governance.governed_invoke(tool, {})
    assert len(out) <= 500  # 截断后仍在近似预算内
