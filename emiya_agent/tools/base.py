"""Tool 抽象：定义、执行上下文、结果。"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel

from emiya_agent.config import Config
from emiya_agent.errors import ToolError

MAX_OUTPUT_CHARS = 30_000


class UIBridge(Protocol):
    """工具与终端交互的最小接口（Repl 实现；测试用 FakeUI）。"""

    async def ask(self, question: str, options: list[str] | None = None) -> str: ...

    def notify(self, text: str) -> None: ...


@dataclass
class ToolContext:
    cwd: Path
    config: Config
    ui: UIBridge
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    state: dict[str, Any] = field(default_factory=dict)  # 回合间共享状态（todo 等）


class ToolResult(BaseModel):
    content: str
    is_error: bool = False


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]]
    read_only: ClassVar[bool] = False  # 只读 ⇒ explore 子agent可用
    interactive: ClassVar[bool] = False  # 交互式 ⇒ 永不并行（会抢终端输入）

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


def truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """超长输出保留头尾。"""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... [truncated {omitted} chars] ...\n{tail}"


def resolve_path(ctx: ToolContext, path_str: str) -> Path:
    """相对路径基于 cwd 解析；返回绝对路径。"""
    if not path_str or not path_str.strip():
        raise ToolError("path 不能为空")
    p = Path(path_str)
    return (p if p.is_absolute() else ctx.cwd / p).resolve()
