"""MCP 消费侧薄适配器（P0-4）。

目标：把第三方 MCP Server 的工具导入 Agent 工具带，继续走现有工具守卫。
- 优先使用官方 `mcp` 包（stdio/HTTP 全传输）；未安装时使用内置零依赖
  stdio JSON-RPC 客户端（最小 MCP 子集：initialize / tools/list / tools/call）。
- 连接失败不阻断 Agent：import_mcp_tools 捕获异常并降级。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, create_model

from src.platform.agent_spec import McpServerSpec
from src.utils.logging_config import get_logger

logger = get_logger("platform.mcp")

MCP_PROTOCOL_VERSION = "2024-11-05"


class McpError(RuntimeError):
    pass


# ------------------------------------------------------------------ #
# 内置 stdio JSON-RPC 客户端（零依赖降级路径）                        #
# ------------------------------------------------------------------ #


class McpStdioClient:
    """逐行 JSON-RPC 2.0 over stdio 的最小 MCP 客户端。"""

    def __init__(self, spec: McpServerSpec):
        self.spec = spec
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._pending: dict[int, tuple[threading.Event, dict]] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._tools: list[dict] = []
        self._closed = False

    # ---- 生命周期 -------------------------------------------------- #
    def connect(self) -> "McpStdioClient":
        if not self.spec.command:
            raise McpError(f"MCP server {self.spec.name} 未配置 command")
        env = dict(os.environ)
        env.update(self.spec.env or {})
        try:
            self._proc = subprocess.Popen(
                [self.spec.command, *self.spec.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
                cwd=str(Path.cwd()),
            )
        except OSError as e:
            raise McpError(f"无法启动 MCP server {self.spec.name}：{e}") from e
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        init = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "artagent", "version": "0.1.0"},
            },
        )
        if init is None:
            raise McpError(f"MCP server {self.spec.name} initialize 失败")
        self._notify("notifications/initialized", {})
        listed = self._request("tools/list", {}) or {}
        self._tools = list((listed.get("tools") or []))
        logger.info("[mcp] %s 已连接，工具 %d 个", self.spec.name, len(self._tools))
        return self

    def close(self) -> None:
        self._closed = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass
        with self._lock:
            for event, _ in self._pending.values():
                event.set()
            self._pending.clear()

    # ---- JSON-RPC -------------------------------------------------- #
    def _next_id_value(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def _request(self, method: str, params: dict) -> Optional[dict]:
        if self._proc is None or self._proc.poll() is not None:
            raise McpError(f"MCP server {self.spec.name} 进程已退出")
        req_id = self._next_id_value()
        event = threading.Event()
        result: dict = {}
        with self._lock:
            self._pending[req_id] = (event, result)
        line = json.dumps(
            {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params},
            ensure_ascii=False,
        )
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            with self._lock:
                self._pending.pop(req_id, None)
            raise McpError(f"MCP 写入失败：{e}") from e
        if not event.wait(timeout=self.spec.timeout_sec):
            with self._lock:
                self._pending.pop(req_id, None)
            raise McpError(f"MCP 请求超时：{method}")
        if result.get("error"):
            raise McpError(f"MCP {method} 错误：{result['error']}")
        return result.get("result")

    def _notify(self, method: str, params: dict) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        line = json.dumps(
            {"jsonrpc": "2.0", "method": method, "params": params},
            ensure_ascii=False,
        )
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _read_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict) or "id" not in msg:
                continue
            with self._lock:
                pair = self._pending.pop(msg["id"], None)
            if pair:
                pair[1].update(msg)
                pair[0].set()

    # ---- MCP 工具 -------------------------------------------------- #
    def list_tools(self) -> list[dict]:
        return list(self._tools)

    def call_tool(self, name: str, args: dict) -> str:
        result = self._request(
            "tools/call", {"name": name, "arguments": args or {}}
        ) or {}
        parts = []
        for part in result.get("content") or []:
            if part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
        text = "\n".join(parts)
        if result.get("isError"):
            raise McpError(f"MCP 工具 {name} 执行错误：{text}")
        return text or "（无返回）"


# ------------------------------------------------------------------ #
# 门面：官方 mcp 包优先，内置 stdio 降级                              #
# ------------------------------------------------------------------ #


def _mcp_package_available() -> bool:
    try:
        import mcp  # noqa: F401

        return True
    except ImportError:
        return False


class McpOfficialClient:
    """官方 mcp 包客户端（stdio；每次调用建立会话，首版够用）。"""

    def __init__(self, spec: McpServerSpec):
        self.spec = spec
        self._tools: list[dict] = []

    @staticmethod
    def _run(coro):
        import asyncio

        try:
            asyncio.get_running_loop()
            raise McpError("官方 mcp 客户端不支持在事件循环线程内同步调用")
        except RuntimeError:
            return asyncio.run(coro)

    def connect(self) -> "McpOfficialClient":
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.spec.command,
            args=self.spec.args,
            env={**os.environ, **(self.spec.env or {})},
        )

        async def _connect():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self._tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "inputSchema": t.inputSchema,
                        }
                        for t in listed.tools
                    ]

        try:
            self._run(_connect())
        except Exception as e:  # noqa: BLE001
            raise McpError(f"官方 MCP 客户端连接 {self.spec.name} 失败：{e}") from e
        return self

    def close(self) -> None:
        pass

    def list_tools(self) -> list[dict]:
        return list(self._tools)

    def call_tool(self, name: str, args: dict) -> str:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.spec.command,
            args=self.spec.args,
            env={**os.environ, **(self.spec.env or {})},
        )

        async def _call():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, args or {})
                    return result

        try:
            result = self._run(_call())
        except McpError:
            raise
        except Exception as e:  # noqa: BLE001
            raise McpError(f"MCP 工具 {name} 调用失败：{e}") from e
        parts = []
        for part in getattr(result, "content", None) or []:
            if getattr(part, "type", "") == "text":
                parts.append(str(getattr(part, "text", "")))
            else:
                parts.append(str(part))
        return "\n".join(parts) or "（无返回）"


class McpToolClient:
    """MCP 客户端门面：官方包可用则用之，否则内置 stdio。"""

    def __init__(self, spec: McpServerSpec, backend: str | None = None):
        self.spec = spec
        self._backend = backend or ("official" if _mcp_package_available() else "stdio")
        self._impl = (
            McpOfficialClient(spec) if self._backend == "official" else McpStdioClient(spec)
        )

    def connect(self) -> "McpToolClient":
        self._impl.connect()
        return self

    def close(self) -> None:
        self._impl.close()

    def list_tools(self) -> list[dict]:
        return self._impl.list_tools()

    def call_tool(self, name: str, args: dict) -> str:
        return self._impl.call_tool(name, args)


# ------------------------------------------------------------------ #
# 导入到 langchain 工具带                                             #
# ------------------------------------------------------------------ #

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_pydantic(json_schema: dict) -> Optional[type[BaseModel]]:
    """把 MCP inputSchema 转成 pydantic 模型（StructuredTool args_schema）。"""
    props = (json_schema or {}).get("properties") or {}
    if not props:
        return None
    required = set((json_schema or {}).get("required") or [])
    fields: dict[str, tuple] = {}
    for name, prop in props.items():
        typ = _TYPE_MAP.get((prop or {}).get("type"), str)
        default = (prop or {}).get("default")
        if name in required:
            fields[name] = (typ, ...)
        else:
            fields[name] = (Optional[typ], default)  # type: ignore[assignment]
    return create_model("McpToolArgs", **fields)


def _safe_tool_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return cleaned or "tool"


def import_mcp_tools(
    servers: list[McpServerSpec],
    client_factory: Callable[[McpServerSpec], Any] | None = None,
) -> list:
    """按 manifest 声明连接 MCP Server，把远程工具包装为 StructuredTool。

    client_factory 供测试注入 fake client；默认走 McpToolClient。
    任一台 server 失败只告警，不阻断 Agent。
    """
    from langchain_core.tools import StructuredTool

    tools: list = []
    for spec in servers:
        if not spec.enabled or not spec.command:
            continue
        factory = client_factory or (lambda s: McpToolClient(s))
        client = factory(spec)
        try:
            client.connect()
        except Exception as e:  # noqa: BLE001
            logger.warning("[mcp] 连接 %s 失败，降级为无 MCP 工具：%s", spec.name, e)
            continue
        try:
            remote_tools = client.list_tools()
        except Exception as e:  # noqa: BLE001
            logger.warning("[mcp] 列举 %s 工具失败：%s", spec.name, e)
            client.close()
            continue
        for remote in remote_tools:
            remote_name = str(remote.get("name") or "")
            if not remote_name:
                continue
            tool_name = f"mcp_{_safe_tool_name(spec.name)}_{_safe_tool_name(remote_name)}"
            schema = _schema_to_pydantic(remote.get("inputSchema") or {})

            def _call(
                args: dict,
                _client=client,
                _name=remote_name,
            ) -> str:
                return _client.call_tool(_name, args)

            tools.append(
                StructuredTool.from_function(
                    func=_call,
                    name=tool_name,
                    description=str(remote.get("description") or f"MCP 工具 {remote_name}"),
                    args_schema=schema,
                )
            )
            logger.info(
                "[mcp] %s 已导入工具 %s", spec.name, tool_name
            )
    return tools
