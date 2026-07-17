"""MCP 工具 → 内部 Tool 桥接：mcp__<server>__<tool> 命名。"""

from __future__ import annotations

from typing import Any

from emiya_agent.mcp_client.manager import MCPManager, MCPServerConn
from emiya_agent.tools.base import Tool, ToolContext, ToolResult


class MCPTool(Tool):
    # MCP 工具副作用未知 → 保守：不并行、不给只读子agent
    read_only = False

    def __init__(
        self,
        conn: MCPServerConn,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        self._conn = conn
        self._tool_name = tool_name
        # 实例属性遮蔽 ClassVar —— 每个 MCP 工具一个实例
        self.name = f"mcp__{conn.name}__{tool_name}"
        self.description = f"[MCP:{conn.name}] {description or tool_name}"
        self.input_schema = input_schema or {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        text, is_error = await self._conn.call(self._tool_name, args)
        return ToolResult(content=text, is_error=is_error)


def build_mcp_tools(manager: MCPManager) -> list[MCPTool]:
    tools: list[MCPTool] = []
    for conn in manager.conns.values():
        if not conn.connected:
            continue
        for t in conn.tools:
            schema = getattr(t, "inputSchema", None) or {}
            tools.append(
                MCPTool(conn, t.name, getattr(t, "description", "") or "", dict(schema))
            )
    return tools
