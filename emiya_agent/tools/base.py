"""Tool 抽象：定义、执行上下文、结果。"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel

from emiya_agent.config import Config
from emiya_agent.errors import ToolError
from emiya_agent.providers.base import ToolSpec

MAX_OUTPUT_CHARS = 30_000
PermissionAction = Literal["allow", "ask", "deny"]


class UIBridge(Protocol):
    """工具与终端交互的最小接口（Repl 实现；测试用 FakeUI）。"""

    async def ask(self, question: str, options: list[str] | None = None) -> str: ...

    def notify(self, text: str) -> None: ...

    async def confirm_tool(self, name: str, args_brief: str) -> bool: ...


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
    # 审批门控：None = 跟随 not read_only；显式 False 可豁免（todo/memory 等低危写）
    requires_approval: ClassVar[bool | None] = None

    def to_spec(self) -> ToolSpec:
        """导出模型可见规格；有动态配置的工具可覆盖此方法。"""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def approval_required(self, args: dict[str, Any]) -> bool:
        """当前调用是否需要用户确认；工具可按 action 细分读写权限。"""
        if self.requires_approval is not None:
            return self.requires_approval
        return not self.read_only

    def permission_override(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> PermissionAction | None:
        """可选的细粒度调用策略；None 回退到全局 tool_approval。"""
        return None

    async def approval_required_for(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> bool:
        """动态审批判断；需要解析真实目标的工具可异步覆盖。"""
        return self.approval_required(args)

    async def approval_preview(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> str | None:
        """可选的审批说明；用于展示经工具解析后的真实执行目标。"""
        return None

    async def approval_cancelled(self, args: dict[str, Any], ctx: ToolContext) -> None:
        """Release one-shot approval state after denial or channel failure."""

        return None

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
    """相对路径基于 cwd 解析，并默认限制在工作区内。

    文件工具没有操作系统级沙箱。若默认允许绝对路径或 ``..`` 越界，
    只读工具会在无需审批的情况下把工作区外的凭据送入模型上下文。
    需要跨目录访问时，用户必须在用户级配置显式开启
    ``allow_external_paths``；项目配置本身受信任门控，不能静默放宽边界。
    """
    if not path_str or not path_str.strip():
        raise ToolError("path 不能为空")
    p = Path(path_str)
    resolved = (p if p.is_absolute() else ctx.cwd / p).resolve()
    if ctx.config.allow_external_paths:
        return resolved
    workspace = ctx.cwd.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise ToolError(
            f"路径超出工作区: {resolved}（工作区: {workspace}）。"
            "如确有需要，请在用户级配置设置 allow_external_paths = true"
        ) from None
    return resolved
