"""文件工作区边界与外部进程参数安全。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from emiya_agent.config import Config
from emiya_agent.errors import ConfigError, ToolError
from emiya_agent.tools import search as search_module
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
def test_runtime_directory_rejects_symlink(name: str, tmp_path: Path):
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
    captured_env: dict[str, str] = {}

    class FakeProcess:
        returncode = 0

    class FakeTree:
        process = FakeProcess()
        terminated = 0

        async def terminate(self):
            self.terminated += 1

    tree = FakeTree()

    async def fake_spawn(argv, *, cwd, env):
        nonlocal captured, captured_env
        captured = tuple(argv)
        captured_env = env
        assert cwd == tmp_path
        return tree

    async def fake_read(received_tree):
        assert received_tree is tree
        return b"", 0

    monkeypatch.setattr(search_module, "_spawn_process_tree", fake_spawn)
    monkeypatch.setattr(search_module, "_read_contained_process", fake_read)
    pattern = "--pre=powershell -Command bad"

    result = await GrepTool()._rg(
        tmp_path / "safe-bin" / "rg",
        pattern,
        tmp_path,
        None,
        cwd=tmp_path,
        env={"LANG": "C"},
    )

    assert result.is_error is False
    assert Path(captured[0]).is_absolute()
    assert "--no-config" in captured
    assert captured[captured.index("-e") + 1] == pattern
    assert captured[captured.index("--") + 1] == str(tmp_path)
    assert captured_env == {"LANG": "C"}
    assert tree.terminated >= 1


def _fake_rg(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / ("rg.exe" if os.name == "nt" else "rg")
    executable.write_bytes(b"fake")
    if os.name != "nt":
        executable.chmod(0o700)
    return executable


def test_ripgrep_resolution_ignores_relative_and_workspace_path_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_rg = _fake_rg(workspace)
    safe_rg = _fake_rg(tmp_path / "safe-bin")
    monkeypatch.chdir(workspace)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((".", str(workspace), str(safe_rg.parent))),
    )

    assert search_module._resolve_rg_executable(workspace) == safe_rg.resolve()

    monkeypatch.setenv("PATH", os.pathsep.join((".", str(workspace))))
    assert search_module._resolve_rg_executable(workspace) is None
    assert workspace_rg.is_file()


def test_ripgrep_environment_excludes_path_and_ambient_credentials(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PATH", "sensitive-path")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("LANG", "C.UTF-8")

    env = search_module._rg_environment()

    assert env.get("LANG") == "C.UTF-8"
    assert "PATH" not in env
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env


@pytest.mark.asyncio
async def test_workspace_ripgrep_fails_closed_without_safe_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _fake_rg(workspace)
    source = workspace / "example.txt"
    source.write_text("private fallback marker\n", encoding="utf-8")
    monkeypatch.setenv("PATH", str(workspace))
    ctx = ToolContext(
        cwd=workspace,
        config=Config(cwd=workspace),
        ui=None,  # type: ignore[arg-type]
    )

    result = await GrepTool().run({"pattern": "fallback marker"}, ctx)

    assert result.is_error is True
    assert "ripgrep" in result.content
    assert "private fallback marker" not in result.content


@pytest.mark.asyncio
async def test_ripgrep_output_is_bounded_and_reports_omitted_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class FakeProcess:
        returncode = 0

    class FakeTree:
        process = FakeProcess()

        async def terminate(self):
            return None

    async def fake_spawn(argv, *, cwd, env):
        del argv, cwd, env
        return FakeTree()

    output = b"".join(f"file:{index}:hit\n".encode() for index in range(500))

    async def fake_read(tree):
        del tree
        return output, 1234

    monkeypatch.setattr(search_module, "_spawn_process_tree", fake_spawn)
    monkeypatch.setattr(search_module, "_read_contained_process", fake_read)

    result = await GrepTool()._rg(
        tmp_path / "safe-rg",
        "hit",
        tmp_path,
        None,
        cwd=tmp_path,
        env={},
    )

    lines = result.content.splitlines()
    assert result.is_error is False
    assert len(lines) == search_module.MAX_MATCHES + 1
    assert lines[-1] == "[ripgrep output truncated: 1234 bytes omitted]"


@pytest.mark.asyncio
async def test_ripgrep_timeout_and_cancellation_terminate_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    started = asyncio.Event()
    trees = []

    class FakeProcess:
        returncode = None

    class FakeTree:
        process = FakeProcess()

        def __init__(self):
            self.terminated = 0

        async def terminate(self):
            self.terminated += 1
            self.process.returncode = -9

    async def fake_spawn(argv, *, cwd, env):
        del argv, cwd, env
        tree = FakeTree()
        trees.append(tree)
        return tree

    async def blocked_read(tree):
        del tree
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(search_module, "_spawn_process_tree", fake_spawn)
    monkeypatch.setattr(search_module, "_read_contained_process", blocked_read)
    monkeypatch.setattr(search_module, "RG_TIMEOUT_SECONDS", 0.001)

    timeout_result = await GrepTool()._rg(
        tmp_path / "safe-rg",
        "x",
        tmp_path,
        None,
        cwd=tmp_path,
        env={},
    )
    assert timeout_result.is_error is True
    assert "timed out" in timeout_result.content
    assert trees[-1].terminated >= 1

    monkeypatch.setattr(search_module, "RG_TIMEOUT_SECONDS", 60)
    started.clear()
    task = asyncio.create_task(
        GrepTool()._rg(
            tmp_path / "safe-rg",
            "x",
            tmp_path,
            None,
            cwd=tmp_path,
            env={},
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert trees[-1].terminated >= 1
