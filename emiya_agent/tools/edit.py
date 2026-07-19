"""精确字符串替换编辑工具。

不变量：old_string 必须唯一匹配（除非 replace_all）；保留文件原换行风格（CRLF/LF）。
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from emiya_agent.tools.base import Tool, ToolContext, ToolResult, resolve_path


class EditFileTool(Tool):
    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = (
        "Replace an exact string in a file. old_string must match exactly once "
        "(including whitespace/indentation) unless replace_all is true. "
        "Read the file first to get the exact text."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence (default false)",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }
    read_only: ClassVar[bool] = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args["path"])
        old = args["old_string"]
        new = args["new_string"]
        replace_all = bool(args.get("replace_all", False))

        if old == new:
            return ToolResult(content="Error: old_string 与 new_string 相同", is_error=True)
        if not old:
            return ToolResult(content="Error: old_string 不能为空", is_error=True)
        if not path.is_file():
            return ToolResult(content=f"Error: 文件不存在: {path}", is_error=True)

        try:
            raw = await asyncio.to_thread(
                path.read_text, encoding="utf-8", newline=""  # newline="" 保留原样
            )
        except UnicodeDecodeError:
            return ToolResult(content=f"Error: {path} 不是 UTF-8 文本文件", is_error=True)

        # 统一在 LF 域内匹配，写回时还原 CRLF
        uses_crlf = "\r\n" in raw
        content = raw.replace("\r\n", "\n") if uses_crlf else raw
        old_lf = old.replace("\r\n", "\n")
        new_lf = new.replace("\r\n", "\n")

        count = content.count(old_lf)
        if count == 0:
            return ToolResult(
                content="Error: old_string 未在文件中找到（注意空白与缩进需完全一致）",
                is_error=True,
            )
        if count > 1 and not replace_all:
            return ToolResult(
                content=f"Error: old_string 出现 {count} 次，不唯一。"
                "请扩大上下文使其唯一，或设置 replace_all=true",
                is_error=True,
            )

        updated = content.replace(old_lf, new_lf) if replace_all else content.replace(
            old_lf, new_lf, 1
        )
        if uses_crlf:
            updated = updated.replace("\n", "\r\n")
        await asyncio.to_thread(path.write_text, updated, encoding="utf-8", newline="")
        n = count if replace_all else 1
        return ToolResult(content=f"已替换 {n} 处: {path}")
