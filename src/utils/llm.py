"""LLM 客户端统一封装：OpenAI 兼容接口单模型直连（DeepSeek/Qwen 等）。

- 单模型直连：模型 / Base URL / API Key 由 config.yaml 配置
  （models.*），环境变量 LLM_API_KEY 等仍可覆盖；
- 工程纪律：显式超时（180s）+ 有限重试（2 次）；调用失败抛原始异常，
  由上层决定如何处理或提示用户。
"""

from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.utils.config import get, get_int

load_dotenv()


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Return a cached chat model instance.
    lru_cache keys on temperature, so different temperatures get different instances.
    """
    api_key = get("models.llm_api_key")
    base_url = get(
        "models.llm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = get("models.llm_model", "deepseek-v3")

    if not api_key:
        raise ValueError(
            "LLM_API_KEY not found. Please set it in config.yaml or your .env file."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        request_timeout=get_int("models.request_timeout_sec", 180, lo=1),
        max_retries=get_int("models.max_retries", 2, lo=0),
    )


def get_deterministic_llm() -> ChatOpenAI:
    """Temperature=0 for tool calling and routing decisions."""
    return get_llm(temperature=0.0)


@lru_cache(maxsize=1)
def get_vision_llm() -> ChatOpenAI:
    """返回支持视觉的模型实例（读图/看图分析专用）。

    注意与 get_llm 的分工：对话模型是纯文本大脑，所有需要"看见图片"
    的场景（image_lookup analyze、read_page_image）都走这个视觉实例。
    """
    model = get("models.vision_model", "qwen3.5-omni-plus-2026-03-15")
    api_key = get("models.llm_api_key")
    if not api_key:
        raise ValueError(
            "LLM_API_KEY not found. Please set it in config.yaml or your .env file."
        )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=get(
            "models.llm_base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=0.3,
        request_timeout=get_int("models.request_timeout_sec", 180, lo=1),
        max_retries=get_int("models.max_retries", 2, lo=0),
    )
