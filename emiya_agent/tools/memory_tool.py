"""memory 工具：agent 读写项目持久记忆。"""

from __future__ import annotations

from typing import Any, ClassVar

from emiya_agent.memory.manager import INDEX_FILE, MemoryManager
from emiya_agent.tools.base import Tool, ToolContext, ToolResult


class MemoryTool(Tool):
    name: ClassVar[str] = "memory"
    description: ClassVar[str] = (
        "Read/write persistent project memory that survives across sessions. "
        "Files live in .emiyacode/memory/; MEMORY.md is the index loaded at "
        "session start (keep it a short list of one-line pointers). "
        "Save durable facts: user preferences, project decisions, gotchas. "
        "Do NOT save what the code/git history already records. "
        "After writing a new memory file, update MEMORY.md too."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read", "write", "delete"],
            },
            "name": {
                "type": "string",
                "description": f"File name, e.g. '{INDEX_FILE}' or 'api-conventions.md'",
            },
            "content": {"type": "string", "description": "Content for write"},
        },
        "required": ["action"],
    }
    read_only: ClassVar[bool] = False  # write/delete 有副作用
    # 只写 .emiyacode/memory/ 且有路径穿越防护，低危豁免审批
    requires_approval: ClassVar[bool | None] = False

    def __init__(self, manager: MemoryManager) -> None:
        self._mm = manager

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = args["action"]
        name = args.get("name") or ""
        try:
            if action == "list":
                files = self._mm.list_files()
                return ToolResult(content="\n".join(files) or "(no memory files)")
            if action == "read":
                return ToolResult(content=self._mm.read(name))
            if action == "write":
                content = args.get("content")
                if content is None:
                    return ToolResult(content="Error: write 需要 content", is_error=True)
                self._mm.write(name, content)
                return ToolResult(content=f"已写入记忆: {name}")
            if action == "delete":
                self._mm.delete(name)
                return ToolResult(content=f"已删除记忆: {name}")
            return ToolResult(content=f"Error: 未知 action {action!r}", is_error=True)
        except (ValueError, FileNotFoundError, OSError) as exc:
            return ToolResult(content=f"Error: {exc}", is_error=True)
