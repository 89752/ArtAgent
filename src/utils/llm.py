"""
DeepSeek LLM client wrapper using LangChain's OpenAI-compatible interface.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Return a cached DeepSeek chat model instance.
    lru_cache keys on temperature, so different temperatures get different instances.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v3")

    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found. "
            "Please set it in your .env file."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        # 必须有超时：DashScope 偶发"连接挂着但不返回"，无超时会让
        # 整个 graph/服务无限等待（2026-08-01 全量回归两次卡死于此）。
        # 180s 覆盖正常慢响应（最长见过 ~120s），重试消化瞬时抖动。
        request_timeout=180,
        max_retries=2,
    )


def get_deterministic_llm() -> ChatOpenAI:
    """Temperature=0 for tool calling and routing decisions."""
    return get_llm(temperature=0.0)


@lru_cache(maxsize=1)
def get_vision_llm() -> ChatOpenAI:
    """返回支持视觉的模型实例（读图/看图分析专用）。

    注意与 get_llm 的分工：对话模型（现为 glm-4.7）是纯文本大脑，
    所有需要"看见图片"的场景（image_lookup analyze、read_page_image）
    都走这个 qwen omni 实例。
    """
    return ChatOpenAI(
        model="qwen3.5-omni-plus",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=0.3,
        request_timeout=180,  # 同 get_llm：防 DashScope 挂起连接无限等待
        max_retries=2,
    )