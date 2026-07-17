"""Shell 执行工具：Windows 用 PowerShell，POSIX 用 bash。

超时默认 120s / 上限 600s；finally 中 terminate+kill 防僵尸进程（Proactor 循环）。
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, ClassVar

from emiya_agent.tools.base import Tool, ToolContext, ToolResult

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600


def _decode(data: bytes) -> str:
    """Windows 控制台输出可能是 GBK；先试 UTF-8，失败回退本地编码。"""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        import locale

        return data.decode(locale.getpreferredencoding(False), errors="replace")


class ShellTool(Tool):
    name: ClassVar[str] = "shell"
    description: ClassVar[str] = (
        "Execute a shell command and return combined stdout/stderr. "
        "On Windows this runs PowerShell; on POSIX it runs bash. "
        "The working directory persists as the project directory for every call. "
        "Use for running programs, tests, git, package managers, etc."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout_seconds": {
                "type": "integer",
                "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT})",
            },
        },
        "required": ["command"],
    }
    read_only: ClassVar[bool] = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args.get("command", "")
        if not command.strip():
            return ToolResult(content="Error: command 为空", is_error=True)
        timeout = min(int(args.get("timeout_seconds") or DEFAULT_TIMEOUT), MAX_TIMEOUT)

        if sys.platform == "win32":
            argv = ["powershell", "-NoProfile", "-Command", command]
        else:
            argv = ["bash", "-c", command]

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ctx.cwd),
            env={**os.environ},
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return ToolResult(
                content=f"Error: 命令超时（{timeout}s）已被终止: {command}",
                is_error=True,
            )
        finally:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except (asyncio.TimeoutError, TimeoutError):
                    proc.kill()
                    await proc.wait()

        output = _decode(stdout or b"").strip()
        if proc.returncode != 0:
            output = f"{output}\n[exit code: {proc.returncode}]".strip()
            return ToolResult(content=output or f"[exit code: {proc.returncode}]", is_error=True)
        return ToolResult(content=output or "(no output)")
