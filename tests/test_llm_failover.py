"""模型主备降级测试（G9/2.5）。"""

import pytest

from src.utils import llm as llm_mod


class _FakeBackup:
    def __init__(self, model):
        self.model = model

    def invoke(self, *a, **k):
        return f"backup-{self.model}"


def test_failover_switches_to_backup(monkeypatch):
    calls = []
    real_invoke = llm_mod.ChatOpenAI.invoke

    def fake_invoke(self, input, config=None, **kwargs):
        calls.append(self.model_name)
        if len(calls) == 1:
            raise TimeoutError("primary down")
        return "ok"

    monkeypatch.setattr(llm_mod.ChatOpenAI, "invoke", fake_invoke)
    model = llm_mod.FailoverChatOpenAI(
        model="primary-model",
        api_key="x",
        base_url="http://localhost",
        backup_model="backup-model",
        backup_api_key="x",
        backup_base_url="http://localhost",
    )
    assert model.invoke("hi") == "ok"
    assert calls == ["primary-model", "backup-model"]
    monkeypatch.setattr(llm_mod.ChatOpenAI, "invoke", real_invoke)


def test_failover_reraises_without_backup(monkeypatch):
    def fake_invoke(self, input, config=None, **kwargs):
        raise TimeoutError("primary down")

    monkeypatch.setattr(llm_mod.ChatOpenAI, "invoke", fake_invoke)
    model = llm_mod.FailoverChatOpenAI(
        model="primary-model", api_key="x", base_url="http://localhost"
    )
    with pytest.raises(TimeoutError):
        model.invoke("hi")


def test_failover_backup_make_backup_returns_none():
    model = llm_mod.FailoverChatOpenAI(
        model="primary-model", api_key="x", base_url="http://localhost"
    )
    assert model._make_backup() is None
