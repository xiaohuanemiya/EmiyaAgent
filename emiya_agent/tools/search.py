"""搜索工具：glob（文件名模式）+ grep（内容正则，要求安全 ripgrep）。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, ClassVar

from emiya_agent.config import select_environment
from emiya_agent.tools.base import Tool, ToolContext, ToolResult, resolve_path
from emiya_agent.tools.shell import (
    _ContainmentError,
    _read_contained_process,
    _spawn_process_tree,
)

IGNORED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".emiyacode", "dist"}
MAX_MATCHES = 200
MAX_FILES = 100
RG_TIMEOUT_SECONDS = 60
_RG_ENV_ALLOWLIST = (
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_*",
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_rg_executable(workspace: Path) -> Path | None:
    """Resolve rg only from absolute PATH directories outside the workspace.

    ``shutil.which`` follows Windows' current-directory-first lookup semantics,
    which lets an untrusted checkout provide ``rg.exe`` to an approval-free
    read-only tool.  Resolve an absolute executable here and pass that exact
    path to the contained subprocess instead.
    """

    workspace_lexical = workspace.absolute()
    try:
        workspace_resolved = workspace.resolve(strict=True)
    except OSError:
        workspace_resolved = workspace_lexical
    names = ("rg.exe", "rg.com") if os.name == "nt" else ("rg",)
    for raw_entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = raw_entry.strip().strip('"')
        if not entry:
            continue
        directory = Path(os.path.expandvars(entry))
        if not directory.is_absolute():
            continue
        for name in names:
            candidate = directory / name
            if _is_within(candidate.absolute(), workspace_lexical):
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError, ValueError):
                continue
            if _is_within(resolved, workspace_resolved):
                continue
            if not resolved.is_file():
                continue
            if os.name != "nt" and not os.access(resolved, os.X_OK):
                continue
            return resolved
    return None


def _rg_environment() -> dict[str, str]:
    """Return the minimal non-secret environment needed by a native rg."""

    return select_environment(_RG_ENV_ALLOWLIST)


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

        rg = await asyncio.to_thread(_resolve_rg_executable, ctx.cwd)
        if rg:
            return await self._rg(
                rg,
                pattern,
                root,
                file_glob,
                cwd=ctx.cwd,
                env=_rg_environment(),
            )
        return ToolResult(
            content=(
                "Error: 未找到工作区外的安全 ripgrep；为避免免审批搜索读取链接目标或"
                "执行无界正则，已禁用 Python 回退。请安装 ripgrep 并将其绝对目录加入 PATH。"
            ),
            is_error=True,
        )

    async def _rg(
        self,
        rg: str | Path,
        pattern: str,
        root: Path,
        file_glob: str | None,
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> ToolResult:
        # --no-config 避免工作站的 RIPGREP_CONFIG_PATH 改变安全语义；
        # -e 明确把模型输入标记为 pattern，-- 明确结束选项，防止
        # ``--pre=COMMAND`` 一类字符串被 rg 当成可执行选项。
        executable = Path(rg)
        if not executable.is_absolute():
            return ToolResult(
                content="Error: ripgrep executable must be an absolute path",
                is_error=True,
            )
        argv = [
            str(executable),
            "--no-config",
            "-n",
            "--no-heading",
            "-S",
            "-m",
            "50",
        ]
        if file_glob:
            argv.extend(["--glob", file_glob])
        argv.extend(["-e", pattern, "--", str(root)])
        try:
            tree = await _spawn_process_tree(argv, cwd=cwd, env=env)
        except _ContainmentError as exc:
            return ToolResult(
                content=f"Error: ripgrep process containment failed: {exc}",
                is_error=True,
            )
        except OSError as exc:
            return ToolResult(
                content=f"Error: ripgrep could not start: {type(exc).__name__}: {exc}",
                is_error=True,
            )

        try:
            stdout, omitted = await asyncio.wait_for(
                _read_contained_process(tree), timeout=RG_TIMEOUT_SECONDS
            )
        except (asyncio.TimeoutError, TimeoutError):
            return ToolResult(
                content=f"Error: ripgrep timed out after {RG_TIMEOUT_SECONDS}s",
                is_error=True,
            )
        finally:
            await tree.terminate()

        raw_text = stdout.decode("utf-8", "replace").strip()
        truncation_note = (
            f"[ripgrep output truncated: {omitted} bytes omitted]" if omitted else ""
        )
        error_text = (
            f"{raw_text}\n{truncation_note}"
            if raw_text and truncation_note
            else raw_text or truncation_note
        )
        if tree.process.returncode not in (0, 1):  # 1 = 无匹配
            return ToolResult(
                content=(
                    f"Error: ripgrep failed with exit code "
                    f"{tree.process.returncode}: {error_text}"
                ),
                is_error=True,
            )
        lines = raw_text.splitlines()[:MAX_MATCHES]
        if truncation_note:
            lines.append(truncation_note)
        return ToolResult(content="\n".join(lines) or "(no matches)")
