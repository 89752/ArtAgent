"""Agent Manifest 定义与加载（规划中）。

AgentSpec = 平台的应用资产：id/name/system_prompt/tools 白名单/skills/
knowledge_base/memory/guardrails/mcp_servers。一个进程注册多个 Agent，
领域差异全部收敛到 manifest，不再改代码。

格式：JSON 原生支持；YAML 在 PyYAML 可用时支持（requirements 已补）。
system_prompt 支持三种形态：
  "@builtin:artagent"  → 复用 src.agent.prompts.SYSTEM_PROMPT（单一事实源）
  "inline text..."     → 直接使用
  "file:path.md"       → 相对 agents/ 目录读取文件
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class McpServerSpec(BaseModel):
    """MCP Server 配置（消费侧）。"""

    name: str
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    timeout_sec: float = 20.0


class MemorySpec(BaseModel):
    enabled: bool = True
    preferences: bool = True
    summary: bool = True


class AgentSpec(BaseModel):
    """一份 Agent 的全部可配置资产。"""

    id: str
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    system_prompt: str = ""  # @builtin:xxx / 内联文本 / file:path
    model_config: dict = Field(default_factory=dict)
    tools: Optional[list[str]] = None  # None = 全部非技能工具
    skills: Optional[list[str]] = None  # None = 全部技能；[] = 禁技能
    knowledge_base: str = "core"
    memory: MemorySpec = Field(default_factory=MemorySpec)
    guardrails: list[str] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)

    def resolved_system_prompt(self, agents_dir: Path | None = None) -> str:
        """解析 system_prompt 到最终字符串。"""
        raw = (self.system_prompt or "").strip()
        if not raw:
            # 兜底：内置艺术 Agent 提示词
            from src.agent.prompts import SYSTEM_PROMPT

            return SYSTEM_PROMPT
        if raw.startswith("@builtin:"):
            name = raw[len("@builtin:") :]
            if name == "artagent":
                from src.agent.prompts import SYSTEM_PROMPT

                return SYSTEM_PROMPT
            raise ValueError(f"未知内置提示词：{raw}")
        if raw.startswith("file:"):
            rel = raw[len("file:") :].strip()
            base = agents_dir or Path.cwd() / "agents"
            return (base / rel).read_text(encoding="utf-8")
        return raw


def parse_agent_spec(text: str, path: Path | None = None) -> AgentSpec:
    """从 JSON 或 YAML 文本解析 AgentSpec。"""
    text = text.strip()
    if not text:
        raise ValueError("空 manifest")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # PyYAML 可选；未安装时 YAML manifest 不可用
        except ImportError as e:
            raise ValueError(
                f"解析 {path or 'manifest'} 失败：非 JSON 且未安装 PyYAML"
            ) from e
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"manifest 顶层必须是对象：{path}")
    spec = AgentSpec.model_validate(data)
    if not spec.id:
        raise ValueError(f"manifest 缺少 id：{path}")
    return spec


def load_agent_spec(path: Path) -> AgentSpec:
    """从文件加载 AgentSpec。"""
    return parse_agent_spec(path.read_text(encoding="utf-8"), path)
