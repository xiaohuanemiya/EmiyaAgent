"""MCP 客户端管理：stdio + streamable-HTTP 服务器生命周期。

关键约束（Windows/anyio）：
- MCP SDK 的 anyio cancel scope 必须在同一个任务内开闭 —— 每个服务器连接
  由一个专属 asyncio 任务持有（_runner），REPL 任务只通过事件与其通信。
- Windows 上 npm 系命令（npx/node 等解析为 .cmd）需要 cmd /c 垫片。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from emiya_agent.config import Config, MCPServerConfig

CONNECT_TIMEOUT = 30
CALL_TIMEOUT = 120


def build_stdio_params(spec: MCPServerConfig, cwd: str | None = None) -> StdioServerParameters:
    """构造 stdio 启动参数；Windows 下 .cmd/.bat 命令自动加 cmd /c 垫片。"""
    assert spec.command
    command = spec.command
    args = list(spec.args)
    if sys.platform == "win32":
        resolved = shutil.which(command)
        if resolved and resolved.lower().endswith((".cmd", ".bat")):
            args = ["/c", command, *args]
            command = "cmd"
    env = {**os.environ, **spec.env}
    return StdioServerParameters(command=command, args=args, env=env, cwd=cwd)


def content_to_text(result: Any) -> str:
    """MCP CallToolResult.content → 文本。"""
    parts: list[str] = []
    for item in getattr(result, "content", None) or []:
        itype = getattr(item, "type", "")
        if itype == "text":
            parts.append(item.text)
        elif itype == "image":
            parts.append("[image content]")
        elif itype == "resource":
            resource = getattr(item, "resource", None)
            text = getattr(resource, "text", None)
            parts.append(text if text is not None else "[binary resource]")
        else:
            parts.append(f"[{itype or 'unknown'} content]")
    return "\n".join(parts) or "(empty result)"


class MCPServerConn:
    """一个 MCP 服务器连接；专属任务持有全部异步上下文。"""

    def __init__(
        self,
        name: str,
        spec: MCPServerConfig,
        cwd: str | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.spec = spec
        self.cwd = cwd
        self.log_dir = log_dir
        self.session: ClientSession | None = None
        self.tools: list[Any] = []  # mcp.types.Tool
        self.error: str | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self.session is not None and self.error is None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._runner(), name=f"mcp-{self.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=CONNECT_TIMEOUT)
        except (asyncio.TimeoutError, TimeoutError):
            self.error = f"连接超时（{CONNECT_TIMEOUT}s）"
            await self.stop()

    async def _runner(self) -> None:
        errlog = None
        try:
            if self.spec.url:
                client_cm = streamablehttp_client(self.spec.url)
                async with client_cm as (read, write, _):
                    await self._serve(read, write)
            elif self.spec.command:
                params = build_stdio_params(self.spec, self.cwd)
                errlog = self._open_errlog()
                async with stdio_client(params, errlog=errlog) as (read, write):
                    await self._serve(read, write)
            else:
                self.error = "配置需给 command（stdio）或 url（HTTP）"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.session = None
            self._ready.set()
            if errlog is not None:
                errlog.close()

    def _open_errlog(self):
        """stdio 服务器的 stderr 写日志文件，不污染终端。"""
        if self.log_dir is None:
            return open(os.devnull, "w", encoding="utf-8")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return open(
            self.log_dir / f"mcp-{self.name}.log", "a", encoding="utf-8", errors="replace"
        )

    async def _serve(self, read: Any, write: Any) -> None:
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            self.tools = list(result.tools)
            self.session = session
            self._ready.set()
            await self._stop.wait()  # 保持连接直到关闭

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """调用工具，返回 (文本, is_error)。"""
        if self.session is None:
            return f"Error: MCP 服务器 {self.name} 未连接", True
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, arguments), timeout=CALL_TIMEOUT
            )
        except (asyncio.TimeoutError, TimeoutError):
            return f"Error: MCP 调用超时（{CALL_TIMEOUT}s）: {tool_name}", True
        except Exception as exc:
            return f"Error: MCP 调用失败: {type(exc).__name__}: {exc}", True
        return content_to_text(result), bool(getattr(result, "isError", False))

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (asyncio.TimeoutError, TimeoutError):
                self._task.cancel()
            self._task = None


class MCPManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.conns: dict[str, MCPServerConn] = {}

    async def start_all(self) -> None:
        """并发连接全部配置的服务器；单个失败不影响其他。"""
        log_dir = self.config.project_dir() / "logs"
        for name, spec in self.config.mcp_servers.items():
            self.conns[name] = MCPServerConn(
                name, spec, cwd=str(self.config.cwd), log_dir=log_dir
            )
        if self.conns:
            await asyncio.gather(*(c.start() for c in self.conns.values()))

    async def stop_all(self) -> None:
        if self.conns:
            await asyncio.gather(
                *(c.stop() for c in self.conns.values()), return_exceptions=True
            )

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "connected": conn.connected,
                "tools": len(conn.tools),
                "error": conn.error,
                "kind": "http" if conn.spec.url else "stdio",
            }
            for name, conn in self.conns.items()
        }
