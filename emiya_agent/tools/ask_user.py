"""向用户提问工具：agent 执行中途请求澄清（阻塞循环等待输入）。"""

from __future__ import annotations

from typing import Any, ClassVar

from emiya_agent.tools.base import Tool, ToolContext, ToolResult


class AskUserTool(Tool):
    name: ClassVar[str] = "ask_user"
    description: ClassVar[str] = (
        "Ask the user a clarifying question and wait for the answer. "
        "Use ONLY when genuinely blocked on a decision only the user can make. "
        "Provide options when the choices are enumerable."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of choices",
            },
        },
        "required": ["question"],
    }
    read_only: ClassVar[bool] = True  # 不改系统状态
    interactive: ClassVar[bool] = True  # 抢终端输入，永不并行

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        question = args["question"]
        options = args.get("options")
        answer = await ctx.ui.ask(question, options)
        return ToolResult(content=f"User answered: {answer}")
