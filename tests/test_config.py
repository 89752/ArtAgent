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
    assert cfg["governance"]["tool_timeout_sec"] == 600  # 技能/子智能体多步循环需要更长超时
    assert cfg["subagents"]["max_concurrent"] == 3
    # 不预设任何模型平台：模型/地址/Key 必须由部署方显式提供
    defaults = config._defaults()["models"]
    assert defaults["llm_api_key"] is None
    assert defaults["llm_base_url"] is None
    assert defaults["llm_model"] is None
    assert defaults["vision_model"] is None
    assert "memory" not in config._defaults()  # 记忆开关由各模块直接读环境变量


def test_missing_env_ref_becomes_none(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = config.reload_config()
    assert cfg["models"]["llm_api_key"] is None


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_SEC", "120")
    monkeypatch.setenv("SUBAGENT_MAX_CONCURRENT", "5")
    cfg = config.reload_config()
    assert cfg["governance"]["tool_timeout_sec"] == "120"
    assert cfg["subagents"]["max_concurrent"] == "5"


def test_get_int_clamps(monkeypatch):
    monkeypatch.setenv("TOOL_RETRIES", "-5")
    assert config.get_int("governance.tool_retries", 1, lo=0) == 0
    monkeypatch.setenv("TOOL_RETRIES", "not-a-number")
    assert config.get_int("governance.tool_retries", 1, lo=0) == 1


def test_env_ref_resolution(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-123")
    assert config.get("models.llm_api_key") == "sk-test-123"


def test_pdf_image_embed_defaults(monkeypatch):
    defaults = config._defaults()["retrieval"]
    assert defaults["pdf_image_embed_provider"] == "dashscope"
    assert defaults["pdf_image_embed_model"] == "tongyi-embedding-vision-plus"
    assert defaults["pdf_image_embed_api_key"] is None


def test_pdf_image_embed_env_override(monkeypatch):
    monkeypatch.setenv("PDF_IMAGE_EMBED_PROVIDER", "openai")
    monkeypatch.setenv("PDF_IMAGE_EMBED_MODEL", "embed-v3")
    cfg = config.reload_config()
    assert cfg["retrieval"]["pdf_image_embed_provider"] == "openai"
    assert cfg["retrieval"]["pdf_image_embed_model"] == "embed-v3"


def test_judge_model_defaults_none(monkeypatch):
    defaults = config._defaults()["models"]
    assert defaults["judge_model"] is None
    assert defaults["judge_api_key"] is None
    assert defaults["judge_base_url"] is None


# ── LLM 客户端 ──────────────────────────────────────────────
def test_get_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        llm_mod.get_llm(temperature=0.777)


def test_get_llm_requires_model_and_base_url(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "none.yaml")
    config.reload_config()
    with pytest.raises(ValueError, match="LLM_MODEL"):
        llm_mod.get_llm(temperature=0.555)


def test_get_llm_reads_env_config(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v3-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    model = llm_mod.get_llm(temperature=0.666)
    assert model.model_name == "deepseek-v3-test"
    assert model.temperature == 0.666
    assert "example.test" in str(model.openai_api_base)
    assert model.openai_api_key.get_secret_value() == "sk-test"


def test_get_vision_llm_falls_back_to_llm_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "none.yaml")
    config.reload_config()
    monkeypatch.setenv("LLM_API_KEY", "sk-llm")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "chat-model")
    monkeypatch.setenv("VISION_MODEL", "vision-model")
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("VISION_BASE_URL", raising=False)
    llm_mod.get_vision_llm.cache_clear()
    try:
        model = llm_mod.get_vision_llm()
        assert model.model_name == "vision-model"
        assert "llm.example.test" in str(model.openai_api_base)
        assert model.openai_api_key.get_secret_value() == "sk-llm"
    finally:
        llm_mod.get_vision_llm.cache_clear()


def test_get_vision_llm_uses_own_keys(monkeypatch):
    monkeypatch.setenv("VISION_API_KEY", "sk-vision")
    monkeypatch.setenv("VISION_BASE_URL", "https://vision.example.test/v1")
    monkeypatch.setenv("VISION_MODEL", "vision-model")
    llm_mod.get_vision_llm.cache_clear()
    try:
        model = llm_mod.get_vision_llm()
        assert "vision.example.test" in str(model.openai_api_base)
        assert model.openai_api_key.get_secret_value() == "sk-vision"
    finally:
        llm_mod.get_vision_llm.cache_clear()


def test_get_vision_llm_falls_back_model_to_chat(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-llm")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "chat-vision-model")
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("VISION_BASE_URL", raising=False)
    llm_mod.get_vision_llm.cache_clear()
    try:
        model = llm_mod.get_vision_llm()
        assert model.model_name == "chat-vision-model"
        assert "llm.example.test" in str(model.openai_api_base)
        assert model.openai_api_key.get_secret_value() == "sk-llm"
    finally:
        llm_mod.get_vision_llm.cache_clear()


def test_deterministic_llm_is_zero_temperature(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    assert llm_mod.get_deterministic_llm().temperature == 0.0


def test_judge_llm_falls_back_to_chat_model(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://chat.example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "chat-model")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    llm_mod.get_judge_llm.cache_clear()
    llm_mod.get_llm.cache_clear()
    try:
        model = llm_mod.get_judge_llm()
        assert model.model_name == "chat-model"
        assert model.temperature == 0.0
    finally:
        llm_mod.get_judge_llm.cache_clear()


def test_judge_llm_uses_own_config(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_API_KEY", "sk-judge")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example.test/v1")
    llm_mod.get_judge_llm.cache_clear()
    try:
        model = llm_mod.get_judge_llm()
        assert model.model_name == "judge-model"
        assert model.temperature == 0.0
        assert "judge.example.test" in str(model.openai_api_base)
        assert model.openai_api_key.get_secret_value() == "sk-judge"
    finally:
        llm_mod.get_judge_llm.cache_clear()
