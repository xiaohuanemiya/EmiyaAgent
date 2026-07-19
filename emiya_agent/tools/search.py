"""搜索工具：glob（文件名模式）+ grep（内容正则，优先 ripgrep）。"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any, ClassVar

from emiya_agent.tools.base import Tool, ToolContext, ToolResult, resolve_path

IGNORED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".emiyacode", "dist"}
MAX_MATCHES = 200
MAX_FILES = 100


def _iter_files(root: Path, pattern: str = "**/*") -> list[Path]:
    out = []
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        if any(part in IGNORED_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


class GlobTool(Tool):
    name: ClassVar[str] = "glob"
    description: ClassVar[str] = (
        'Find files by glob pattern, e.g. "**/*.py" or "src/**/*.ts". '
        "Results sorted by modification time (newest first)."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Base directory (default: cwd)"},
        },
        "required": ["pattern"],
    }
    read_only: ClassVar[bool] = True

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = resolve_path(ctx, args.get("path") or ".")

        def _scan() -> list:
            files = _iter_files(root, args["pattern"])
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return files

        files = await asyncio.to_thread(_scan)  # 大目录树遍历不阻塞事件循环
        if not files:
            return ToolResult(content="(no matches)")
        lines = [str(p) for p in files[:MAX_FILES]]
        if len(files) > MAX_FILES:
            lines.append(f"... ({len(files) - MAX_FILES} more)")
        return ToolResult(content="\n".join(lines))


class GrepTool(Tool):
    name: ClassVar[str] = "grep"
    description: ClassVar[str] = (
        "Search file contents with a regular expression. "
        'Optional glob filter, e.g. "*.py". Returns file:line:text matches.'
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern"},
            "path": {"type": "string", "description": "Base directory (default: cwd)"},
            "glob": {"type": "string", "description": 'File filter like "*.py"'},
        },
        "required": ["pattern"],
    }
    read_only: ClassVar[bool] = True

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = resolve_path(ctx, args.get("path") or ".")
        pattern = args["pattern"]
        file_glob = args.get("glob")

        rg = shutil.which("rg")
        if rg:
            return await self._rg(rg, pattern, root, file_glob)
        return await asyncio.to_thread(self._python_grep, pattern, root, file_glob)

    async def _rg(
        self, rg: str, pattern: str, root: Path, file_glob: str | None
    ) -> ToolResult:
        argv = [rg, "-n", "--no-heading", "-S", "-m", "50", pattern, str(root)]
        if file_glob:
            argv[1:1] = ["--glob", file_glob]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        if proc.returncode not in (0, 1):  # 1 = 无匹配
            return ToolResult(
                content=f"Error: ripgrep 失败: {stderr.decode('utf-8', 'replace')}",
                is_error=True,
            )
        text = stdout.decode("utf-8", "replace").strip()
        lines = text.splitlines()[:MAX_MATCHES]
        return ToolResult(content="\n".join(lines) or "(no matches)")

    def _python_grep(
        self, pattern: str, root: Path, file_glob: str | None
    ) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(content=f"Error: 非法正则: {exc}", is_error=True)
        matches: list[str] = []
        for f in _iter_files(root, f"**/{file_glob}" if file_glob else "**/*"):
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{f}:{lineno}:{line.strip()[:300]}")
                    if len(matches) >= MAX_MATCHES:
                        return ToolResult(content="\n".join(matches) + "\n... (capped)")
        return ToolResult(content="\n".join(matches) or "(no matches)")
