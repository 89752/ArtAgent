"""DeepSeek LLM client wrapper using LangChain's OpenAI-compatible interface.

G9/2.5 模型鲁棒性：主备降级。主模型 invoke 失败（超时/额度/网络）时自动切
备份模型（env：LLM_BACKUP_MODEL / LLM_BACKUP_BASE_URL / LLM_BACKUP_API_KEY，
视觉走 VISION_BACKUP_MODEL）。未配置备份则维持原行为（抛原始异常）。
"""

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

load_dotenv()

logger = logging.getLogger("llm")


class FailoverChatOpenAI(ChatOpenAI):
    """主备降级 ChatOpenAI：同步/异步 invoke 失败自动切备份。"""

    _backup_model: str | None = PrivateAttr(default=None)
    _backup_api_key: str | None = PrivateAttr(default=None)
    _backup_base_url: str | None = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        backup_model = kwargs.pop("backup_model", None)
        backup_api_key = kwargs.pop("backup_api_key", None)
        backup_base_url = kwargs.pop("backup_base_url", None)
        super().__init__(**kwargs)
        self._backup_model = backup_model
        self._backup_api_key = backup_api_key
        self._backup_base_url = backup_base_url

    def _make_backup(self):
        if not self._backup_model:
            return None
        return ChatOpenAI(
            model=self._backup_model,
    api_key=self._backup_api_key or os.getenv("LLM_API_KEY"),
            base_url=self._backup_base_url or self.base_url,
            temperature=self.temperature,
            request_timeout=getattr(self, "request_timeout", 180),
            max_retries=1,
        )

    def invoke(self, input, config=None, **kwargs):
        try:
            return super().invoke(input, config=config, **kwargs)
        except Exception as e:  # noqa: BLE001 —— 主模型失败统一走降级
            backup = self._make_backup()
            if backup is None:
                raise
            logger.warning(
                "主模型 %s 调用失败（%s），切换到备份 %s",
                self.model_name, e, self._backup_model,
            )
            return backup.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        try:
            return await super().ainvoke(input, config=config, **kwargs)
        except Exception as e:  # noqa: BLE001
            backup = self._make_backup()
            if backup is None:
                raise
            logger.warning(
                "主模型 %s 调用失败（%s），切换到备份 %s",
                self.model_name, e, self._backup_model,
            )
            return await backup.ainvoke(input, config=config, **kwargs)


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Return a cached DeepSeek chat model instance.
    lru_cache keys on temperature, so different temperatures get different instances.
    """
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v3")

    if not api_key:
        raise ValueError(
            "LLM_API_KEY not found. "
            "Please set it in your .env file."
        )

    return FailoverChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        # 必须有超时：DashScope 偶发"连接挂着但不返回"，无超时会让
        # 整个 graph/服务无限等待（2026-08-01 全量回归两次卡死于此）。
        # 180s 覆盖正常慢响应（最长见过 ~120s），重试消化瞬时抖动。
        request_timeout=180,
        max_retries=2,
        backup_model=os.getenv("LLM_BACKUP_MODEL") or None,
        backup_base_url=os.getenv("LLM_BACKUP_BASE_URL") or None,
        backup_api_key=os.getenv("LLM_BACKUP_API_KEY") or None,
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
    model = os.getenv("VISION_MODEL", "qwen3.5-omni-plus-2026-03-15")
    return FailoverChatOpenAI(
        model=model,
    api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=0.3,
        request_timeout=180,  # 同 get_llm：防 DashScope 挂起连接无限等待
        max_retries=2,
        backup_model=os.getenv("VISION_BACKUP_MODEL") or None,
        backup_base_url=os.getenv("VISION_BACKUP_BASE_URL") or None,
        backup_api_key=os.getenv("VISION_BACKUP_API_KEY") or None,
    )
