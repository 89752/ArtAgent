"""配置加载器 + LLM 客户端测试：默认值、env 覆盖、$VAR 解析、边界钳制。"""

import pytest

from src.utils import config
from src.utils import llm as llm_mod


@pytest.fixture(autouse=True)
def _reload_config():
    config.reload_config()
    yield
    config.reload_config()


def test_defaults_without_file(monkeypatch):
    monkeypatch.delenv("TOOL_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    cfg = config.reload_config()
    assert cfg["governance"]["tool_timeout_sec"] == 60
    assert cfg["models"]["llm_model"] == "deepseek-v3"
    assert cfg["memory"]["auto_extract"] is True
    assert cfg["memory"]["extract_interval"] == 1


def test_missing_env_ref_becomes_none(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = config.reload_config()
    assert cfg["models"]["llm_api_key"] is None


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_SEC", "120")
    monkeypatch.setenv("MEMORY_AUTO_EXTRACT", "1")
    cfg = config.reload_config()
    assert cfg["governance"]["tool_timeout_sec"] == "120"
    assert cfg["memory"]["auto_extract"] == "1"


def test_get_int_clamps(monkeypatch):
    monkeypatch.setenv("TOOL_RETRIES", "-5")
    assert config.get_int("governance.tool_retries", 1, lo=0) == 0
    monkeypatch.setenv("TOOL_RETRIES", "not-a-number")
    assert config.get_int("governance.tool_retries", 1, lo=0) == 1


def test_env_ref_resolution(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-123")
    assert config.get("models.llm_api_key") == "sk-test-123"


def test_get_bool_parsing(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "false")
    assert config.get_bool("retrieval.rerank_enabled", True) is False
    monkeypatch.setenv("RERANK_ENABLED", "1")
    assert config.get_bool("retrieval.rerank_enabled", True) is True


def test_get_path_resolves_relative(monkeypatch):
    monkeypatch.setenv("CORE_DATA_PATH", "./data/core/artworks_core.csv")
    path = config.get_path("project.core_data_path", None)
    assert path is not None
    assert path.is_absolute()


# ── LLM 客户端 ──────────────────────────────────────────────
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
