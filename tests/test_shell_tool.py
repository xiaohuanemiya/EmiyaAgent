"""shell 工具：真实子进程（PowerShell/bash）、退出码、超时。"""

import sys

import pytest

from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.shell import ShellTool

from tests.conftest import FakeUI


@pytest.fixture
def ctx(tmp_path, config):
    return ToolContext(cwd=tmp_path, config=config, ui=FakeUI())


@pytest.mark.asyncio
async def test_echo(ctx):
    result = await ShellTool().run({"command": "echo hello-emiya"}, ctx)
    assert not result.is_error
    assert "hello-emiya" in result.content


@pytest.mark.asyncio
async def test_nonzero_exit_is_error(ctx):
    result = await ShellTool().run({"command": "exit 3"}, ctx)
    assert result.is_error
    assert "exit code: 3" in result.content


@pytest.mark.asyncio
async def test_cwd_is_project_dir(tmp_path, ctx):
    (tmp_path / "probe.txt").write_text("x", encoding="utf-8")
    cmd = "dir probe.txt" if sys.platform == "win32" else "ls probe.txt"
    result = await ShellTool().run({"command": cmd}, ctx)
    assert not result.is_error
    assert "probe.txt" in result.content


@pytest.mark.asyncio
async def test_timeout_kills_process(ctx):
    sleep_cmd = "Start-Sleep -Seconds 30" if sys.platform == "win32" else "sleep 30"
    result = await ShellTool().run(
        {"command": sleep_cmd, "timeout_seconds": 1}, ctx
    )
    assert result.is_error
    assert "超时" in result.content


@pytest.mark.asyncio
async def test_empty_command_is_error(ctx):
    result = await ShellTool().run({"command": "  "}, ctx)
    assert result.is_error
