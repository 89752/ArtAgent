"""LLM 客户端单测：env 配置读取、缺 key 报错、温度缓存。"""

import pytest

from src.utils import llm as llm_mod


def test_get_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        llm_mod.get_llm(temperature=0.777)


def test_get_llm_reads_env_config(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v3-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    model = llm_mod.get_llm(temperature=0.666)
    assert model.model_name == "deepseek-v3-test"
    assert model.temperature == 0.666
    assert "example.test" in str(model.openai_api_base)
    assert model.openai_api_key.get_secret_value() == "sk-test"


def test_deterministic_llm_is_zero_temperature(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    assert llm_mod.get_deterministic_llm().temperature == 0.0
