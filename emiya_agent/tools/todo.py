"""任务清单工具：模型维护待办进度，UI 实时展示。状态存于 ToolContext.state。"""

from __future__ import annotations

from typing import Any, ClassVar

from emiya_agent.tools.base import Tool, ToolContext, ToolResult

STATE_KEY = "todos"
_STATUS_ICON = {"pending": "○", "in_progress": "◐", "completed": "●"}


class TodoWriteTool(Tool):
    name: ClassVar[str] = "todo_write"
    description: ClassVar[str] = (
        "Maintain the task list for the current job. Pass the FULL updated list "
        "each time. Use for multi-step tasks so the user can follow progress. "
        'Each item: {"content": str, "status": "pending"|"in_progress"|"completed"}.'
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }
    read_only: ClassVar[bool] = False
    requires_approval: ClassVar[bool | None] = False  # 仅内部状态，低危豁免审批

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        todos = args.get("todos") or []
        ctx.state[STATE_KEY] = todos
        lines = [
            f"{_STATUS_ICON.get(t.get('status', 'pending'), '○')} {t.get('content', '')}"
            for t in todos
        ]
        rendered = "\n".join(lines) or "(empty)"
        ctx.ui.notify(rendered)
        done = sum(1 for t in todos if t.get("status") == "completed")
        return ToolResult(content=f"Todo list updated ({done}/{len(todos)} completed):\n{rendered}")
