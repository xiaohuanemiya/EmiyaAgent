"""文件系统工具：read_file / write_file / list_dir。统一 UTF-8。"""

from __future__ import annotations

from typing import Any, ClassVar

from emiya_agent.tools.base import Tool, ToolContext, ToolResult, resolve_path

MAX_LINES = 2000
MAX_LINE_LEN = 2000


class ReadFileTool(Tool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read a text file. Returns numbered lines (cat -n style). "
        "Use offset/limit for large files."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (absolute or relative to cwd)"},
            "offset": {"type": "integer", "description": "1-based start line (default 1)"},
            "limit": {"type": "integer", "description": f"Max lines to read (default {MAX_LINES})"},
        },
        "required": ["path"],
    }
    read_only: ClassVar[bool] = True

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args["path"])
        if not path.is_file():
            return ToolResult(content=f"Error: 文件不存在: {path}", is_error=True)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                content=f"Error: {path} 不是 UTF-8 文本文件（可能是二进制）",
                is_error=True,
            )
        lines = text.splitlines()
        offset = max(int(args.get("offset") or 1), 1)
        limit = int(args.get("limit") or MAX_LINES)
        window = lines[offset - 1 : offset - 1 + limit]
        if not window and lines:
            return ToolResult(
                content=f"Error: offset={offset} 超出文件行数 {len(lines)}", is_error=True
            )
        numbered = [
            f"{i:6d}\t{line[:MAX_LINE_LEN]}"
            for i, line in enumerate(window, start=offset)
        ]
        body = "\n".join(numbered) if numbered else "(empty file)"
        if offset - 1 + limit < len(lines):
            body += f"\n... ({len(lines) - offset + 1 - limit} more lines, use offset to continue)"
        return ToolResult(content=body)


class WriteFileTool(Tool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Write content to a file (UTF-8), creating parent directories if needed. "
        "Overwrites existing files — prefer edit_file for partial changes."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    read_only: ClassVar[bool] = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        content = args["content"]
        path.write_text(content, encoding="utf-8", newline="\n")
        return ToolResult(content=f"已写入 {path}（{len(content)} 字符）")


class ListDirTool(Tool):
    name: ClassVar[str] = "list_dir"
    description: ClassVar[str] = "List entries of a directory (directories end with /)."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path (default: cwd)"},
        },
    }
    read_only: ClassVar[bool] = True

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args.get("path") or ".")
        if not path.is_dir():
            return ToolResult(content=f"Error: 目录不存在: {path}", is_error=True)
        entries = sorted(
            path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
        lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries[:500]]
        if len(entries) > 500:
            lines.append(f"... ({len(entries) - 500} more)")
        return ToolResult(content="\n".join(lines) or "(empty directory)")
