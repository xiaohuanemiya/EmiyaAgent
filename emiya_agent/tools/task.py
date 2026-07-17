"""spawn_subagent 工具：把 SubagentSpawner 暴露给模型。"""

from __future__ import annotations

from typing import Any, ClassVar

from emiya_agent.tools.base import Tool, ToolContext, ToolResult


class SpawnSubagentTool(Tool):
    name: ClassVar[str] = "spawn_subagent"
    description: ClassVar[str] = (
        "Delegate a self-contained task to a subagent with an isolated context. "
        "Roles: 'explore' (read-only codebase search, fast/cheap — use for broad "
        "fan-out searches where you only need conclusions), 'plan' (read-only "
        "design work on a strong model), 'general' (full tools, end-to-end "
        "execution). The prompt must be self-contained — the subagent cannot "
        "see this conversation. You only receive its final report."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["explore", "plan", "general"]},
            "prompt": {
                "type": "string",
                "description": "Self-contained task description for the subagent",
            },
        },
        "required": ["role", "prompt"],
    }
    read_only: ClassVar[bool] = False  # 输出流占用终端，串行执行
    interactive: ClassVar[bool] = True

    def __init__(self, spawner) -> None:  # SubagentSpawner（避免循环导入不标注类型）
        self._spawner = spawner

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        report = await self._spawner.spawn(args["role"], args["prompt"], ctx)
        is_error = report.startswith("Error:") or report.startswith("Subagent refused")
        return ToolResult(content=report, is_error=is_error)
