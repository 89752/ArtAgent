"""LLM 客户端统一封装：OpenAI 兼容接口单模型直连。

- 单模型直连：模型 / Base URL / API Key 由 config.yaml 配置（models.*），
  环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 等仍可覆盖；
- 不预设任何模型平台：缺少任一必填项时抛出明确错误；
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
    base_url = get("models.llm_base_url")
    model = get("models.llm_model")
    missing = [
        label
        for label, value in (
            ("LLM_API_KEY", api_key),
            ("LLM_BASE_URL", base_url),
            ("LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "缺少模型配置："
            + ", ".join(missing)
            + "（请在 config.yaml 或环境变量中设置，不要预设第三方平台）"
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
def get_judge_llm() -> ChatOpenAI:
    """评估用裁判模型：独立配置（models.judge_* / JUDGE_*），缺省回落对话模型。

    与 get_llm 分工：评测（llm-as-judge）可用更强/更稳的独立模型，
    避免"用被测模型给自己打分"的偏差；未配置时回落对话模型，行为不变。
    """
    model = get("models.judge_model")
    if not model:
        return get_llm(temperature=0.0)
    api_key = get("models.judge_api_key") or get("models.llm_api_key")
    base_url = get("models.judge_base_url") or get("models.llm_base_url")
    missing = [
        label
        for label, value in (
            ("JUDGE_MODEL", model),
            ("JUDGE_API_KEY", api_key),
            ("JUDGE_BASE_URL", base_url),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "缺少裁判模型配置："
            + ", ".join(missing)
            + "（judge_api_key / judge_base_url 可省略，回落对话模型配置）"
        )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        request_timeout=get_int("models.request_timeout_sec", 180, lo=1),
        max_retries=get_int("models.max_retries", 2, lo=0),
    )


@lru_cache(maxsize=1)
def get_vision_llm() -> ChatOpenAI:
    """返回支持视觉的模型实例（读图/看图分析专用）。

    注意与 get_llm 的分工：对话模型是纯文本大脑，所有需要"看见图片"
    的场景（image_lookup analyze、read_page_image）都走这个视觉实例。
    视觉模型可独立配置；缺省回落对话模型（要求 LLM_MODEL 支持图像输入，
    如 qwen3.7-flash / glm-4.6v）。
    """
    model = get("models.vision_model") or get("models.llm_model")
    api_key = get("models.vision_api_key") or get("models.llm_api_key")
    base_url = get("models.vision_base_url") or get("models.llm_base_url")
    missing = [
        label
        for label, value in (
            ("VISION_MODEL", model),
            ("VISION_API_KEY", api_key),
            ("VISION_BASE_URL", base_url),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "缺少视觉模型配置："
            + ", ".join(missing)
            + "（vision_model 缺省回落 LLM_MODEL；vision_api_key/vision_base_url 缺省回落对话模型配置）"
        )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
        request_timeout=get_int("models.request_timeout_sec", 180, lo=1),
        max_retries=get_int("models.max_retries", 2, lo=0),
    )
