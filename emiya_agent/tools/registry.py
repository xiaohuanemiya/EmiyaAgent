"""工具注册表：注册、子集、规格导出（按名称排序保证缓存稳定）、分发执行。"""

from __future__ import annotations

from emiya_agent.providers.base import ToolSpec
from emiya_agent.tools.base import (
    PermissionAction,
    Tool,
    ToolContext,
    truncate_output,
)
from emiya_agent.types import ToolResultBlock, ToolUseBlock


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def subset(
        self,
        *,
        read_only_only: bool = False,
        exclude: frozenset[str] | set[str] = frozenset(),
    ) -> "ToolRegistry":
        sub = ToolRegistry()
        for name, tool in self._tools.items():
            if name in exclude:
                continue
            if read_only_only and not tool.read_only:
                continue
            sub.register(tool)
        return sub

    def specs(self) -> list[ToolSpec]:
        return [tool.to_spec() for _, tool in sorted(self._tools.items())]

    def is_read_only(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.read_only

    def is_parallel_safe(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.read_only and not tool.interactive

    def needs_approval(self, name: str, args: dict | None = None) -> bool:
        tool = self._tools.get(name)
        if tool is None:
            return False
        return tool.approval_required(args or {})

    def permission_decision(
        self, name: str, args: dict, ctx: ToolContext
    ) -> PermissionAction:
        """工具级规则优先；否则按全局审批模式与工具副作用属性决定。"""
        tool = self._tools.get(name)
        if tool is None:
            return "allow"
        override = tool.permission_override(args, ctx)
        if override is not None:
            return override
        if ctx.config.tool_approval == "ask" and tool.approval_required(args):
            return "ask"
        return "allow"

    async def permission_decision_for(
        self, name: str, args: dict, ctx: ToolContext
    ) -> PermissionAction:
        """支持需要异步解析真实执行目标的工具审批。"""
        tool = self._tools.get(name)
        if tool is None:
            return "allow"
        override = tool.permission_override(args, ctx)
        if override is not None:
            return override
        if ctx.config.tool_approval == "ask" and await tool.approval_required_for(
            args, ctx
        ):
            return "ask"
        return "allow"

    async def approval_preview(
        self, name: str, args: dict, ctx: ToolContext
    ) -> str | None:
        tool = self._tools.get(name)
        if tool is None:
            return None
        return await tool.approval_preview(args, ctx)

    async def approval_cancelled(self, name: str, args: dict, ctx: ToolContext) -> None:
        tool = self._tools.get(name)
        if tool is not None:
            await tool.approval_cancelled(args, ctx)

    async def dispatch(self, call: ToolUseBlock, ctx: ToolContext) -> ToolResultBlock:
        """执行一个工具调用；任何异常都转为 is_error 结果，绝不向上抛。"""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResultBlock(
                tool_use_id=call.id,
                content=f"Error: unknown tool {call.name!r}",
                is_error=True,
            )
        try:
            result = await tool.run(call.input, ctx)
            return ToolResultBlock(
                tool_use_id=call.id,
                content=truncate_output(result.content),
                is_error=result.is_error,
            )
        except Exception as exc:  # 工具内部错误 → 交给模型自行调整
            return ToolResultBlock(
                tool_use_id=call.id,
                content=f"Error: {type(exc).__name__}: {exc}",
                is_error=True,
            )
