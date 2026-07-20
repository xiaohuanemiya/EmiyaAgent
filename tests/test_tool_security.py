"""文件工作区边界与外部进程参数安全。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from emiya_agent.config import Config
from emiya_agent.errors import ConfigError, ToolError
from emiya_agent.tools.base import ToolContext, resolve_path
from emiya_agent.tools.search import GrepTool


def test_resolve_path_rejects_workspace_escape(tmp_path: Path, quiet_console):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = ToolContext(cwd=workspace, config=Config(cwd=workspace), ui=None)  # type: ignore[arg-type]

    with pytest.raises(ToolError, match="超出工作区"):
        resolve_path(ctx, "../secret.txt")
    with pytest.raises(ToolError, match="超出工作区"):
        resolve_path(ctx, str(tmp_path / "absolute-secret.txt"))


def test_resolve_path_rejects_symlink_escape(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    ctx = ToolContext(
        cwd=workspace,
        config=Config(cwd=workspace),
        ui=None,  # type: ignore[arg-type]
    )

    with pytest.raises(ToolError, match="超出工作区"):
        resolve_path(ctx, "link.txt")


def test_resolve_path_can_be_enabled_from_trusted_config(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = Config(cwd=workspace, allow_external_paths=True)
    ctx = ToolContext(cwd=workspace, config=cfg, ui=None)  # type: ignore[arg-type]

    assert resolve_path(ctx, "../secret.txt") == (tmp_path / "secret.txt").resolve()


@pytest.mark.parametrize("name", ["memory", "sessions", "logs"])
def test_runtime_directory_rejects_symlink(
    name: str, tmp_path: Path
):
    project_dir = tmp_path / ".emiyacode"
    project_dir.mkdir()
    target = tmp_path / f"real-{name}"
    target.mkdir()
    link = project_dir / name
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(ConfigError, match="符号链接|junction"):
        Config(cwd=tmp_path).project_runtime_dir(name)


def test_runtime_directory_name_cannot_escape_workspace(tmp_path: Path):
    with pytest.raises(ConfigError, match="非法运行时目录名"):
        Config(cwd=tmp_path).project_runtime_dir("../outside")


@pytest.mark.asyncio
async def test_ripgrep_pattern_cannot_be_parsed_as_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: tuple[str, ...] = ()

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

        def kill(self):
            raise AssertionError("completed process must not be killed")

        async def wait(self):
            return 0

    async def fake_exec(*argv, **kwargs):
        nonlocal captured
        captured = tuple(argv)
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    pattern = "--pre=powershell -Command bad"

    result = await GrepTool()._rg("rg", pattern, tmp_path, None)

    assert result.is_error is False
    assert "--no-config" in captured
    assert captured[captured.index("-e") + 1] == pattern
    assert captured[captured.index("--") + 1] == str(tmp_path)
