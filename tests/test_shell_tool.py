"""shell 工具：真实子进程（PowerShell/bash）、退出码、超时。"""

import asyncio
import base64
import signal
import sys

import pytest

from emiya_agent.config import ShellPermissionRule
from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.tools import shell as shell_module
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
async def test_clean_shell_exit_still_kills_background_child(ctx, tmp_path):
    started = tmp_path / "background-started"
    escaped = tmp_path / "background-escaped"
    if sys.platform == "win32":
        child_script = (
            "Set-Content -LiteralPath 'background-started' -Value started; "
            "Start-Sleep -Milliseconds 900; "
            "Set-Content -LiteralPath 'background-escaped' -Value escaped"
        )
        encoded = base64.b64encode(child_script.encode("utf-16le")).decode()
        command = (
            "$psi = New-Object System.Diagnostics.ProcessStartInfo; "
            "$psi.FileName = Join-Path $PSHOME 'powershell.exe'; "
            f"$psi.Arguments = '-NoProfile -EncodedCommand {encoded}'; "
            "$psi.UseShellExecute = $false; $psi.CreateNoWindow = $true; "
            "$psi.RedirectStandardOutput = $true; "
            "$psi.RedirectStandardError = $true; "
            "$child = [System.Diagnostics.Process]::Start($psi); "
            "for ($i = 0; $i -lt 100 -and "
            "-not (Test-Path -LiteralPath 'background-started'); $i++) { "
            "Start-Sleep -Milliseconds 20 }; "
            "$child.Dispose(); "
            "if (-not (Test-Path -LiteralPath 'background-started')) { exit 7 }"
        )
    else:
        command = (
            "(printf started > background-started; sleep 0.9; "
            "printf escaped > background-escaped) "
            ">child.stdout 2>child.stderr & "
            "for i in {1..100}; do "
            "[ -f background-started ] && exit 0; sleep 0.02; done; exit 7"
        )

    started_at = asyncio.get_running_loop().time()
    result = await ShellTool().run(
        {"command": command, "timeout_seconds": 5}, ctx
    )
    elapsed = asyncio.get_running_loop().time() - started_at
    assert not result.is_error, result.content
    assert started.exists()
    assert not escaped.exists(), (
        f"shell did not return before the child delay ({elapsed:.3f}s)"
    )
    await asyncio.sleep(1.1)
    assert not escaped.exists()


@pytest.mark.asyncio
async def test_empty_command_is_error(ctx):
    result = await ShellTool().run({"command": "  "}, ctx)
    assert result.is_error


@pytest.mark.asyncio
async def test_shell_environment_requires_explicit_opt_in(
    ctx, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("EMIYA_TEST_SECRET", "leaked")
    if sys.platform == "win32":
        command = (
            "if ($env:EMIYA_TEST_SECRET) { 'leaked' } else { 'clean' }"
        )
    else:
        command = "if [ -n \"$EMIYA_TEST_SECRET\" ]; then echo leaked; else echo clean; fi"

    clean = await ShellTool().run({"command": command}, ctx)
    assert clean.content == "clean"

    ctx.config.shell_inherit_env.append("EMIYA_TEST_SECRET")
    inherited = await ShellTool().run({"command": command}, ctx)
    assert inherited.content == "leaked"


def test_shell_rules_override_global_approval(ctx):
    registry = ToolRegistry()
    registry.register(ShellTool())
    ctx.config.shell_rules = [
        ShellPermissionRule(action="deny", pattern=r"rm .*"),
        ShellPermissionRule(action="allow", pattern=r"git status"),
        ShellPermissionRule(action="ask", pattern=r"deploy .*"),
    ]
    ctx.config.tool_approval = "auto"

    assert registry.permission_decision("shell", {"command": "rm -rf x"}, ctx) == "deny"
    assert registry.permission_decision("shell", {"command": "git status"}, ctx) == "allow"
    assert registry.permission_decision("shell", {"command": "deploy prod"}, ctx) == "ask"
    assert registry.permission_decision("shell", {"command": "echo ok"}, ctx) == "allow"


@pytest.mark.asyncio
async def test_windows_shell_is_suspended_until_job_assignment(
    ctx, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []
    captured_options = {}

    class EmptyReader:
        async def read(self, size: int) -> bytes:
            del size
            return b""

    class FakeProcess:
        pid = 4242
        returncode = 0
        stdout = EmptyReader()

        async def wait(self) -> int:
            return 0

    async def fake_exec(*argv, **options):
        del argv
        captured_options.update(options)
        events.append("spawned-suspended")
        return FakeProcess()

    class TrackingJob:
        def __init__(self, pid: int) -> None:
            assert pid == 4242
            assert events == ["spawned-suspended"]
            events.append("job-assigned")
            self.handle = 1

        def close(self) -> bool:
            events.append("job-closed")
            self.handle = None
            return True

    def resume(pid: int) -> None:
        assert pid == 4242
        assert events == ["spawned-suspended", "job-assigned"]
        events.append("resumed")

    monkeypatch.setattr(shell_module.sys, "platform", "win32")
    monkeypatch.setattr(
        shell_module, "_windows_system_executable", lambda name: name
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(shell_module, "_WindowsJob", TrackingJob)
    monkeypatch.setattr(shell_module, "_resume_windows_process", resume)

    result = await ShellTool().run({"command": "Write-Output ok"}, ctx)

    assert not result.is_error
    assert captured_options["creationflags"] & shell_module._CREATE_SUSPENDED
    assert events == [
        "spawned-suspended",
        "job-assigned",
        "resumed",
        "job-closed",
    ]


@pytest.mark.asyncio
async def test_cancel_during_windows_spawn_kills_suspended_process(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    spawn_started = asyncio.Event()
    release_spawn = asyncio.Event()
    resume_calls = 0

    class FakeProcess:
        pid = 4343
        returncode = None

        def __init__(self) -> None:
            self.kills = 0

        def kill(self) -> None:
            self.kills += 1
            self.returncode = -9

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process = FakeProcess()

    async def delayed_exec(*argv, **options):
        del argv, options
        spawn_started.set()
        await release_spawn.wait()
        return process

    def resume(pid: int) -> None:
        nonlocal resume_calls
        del pid
        resume_calls += 1

    monkeypatch.setattr(shell_module.sys, "platform", "win32")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_exec)
    monkeypatch.setattr(shell_module, "_resume_windows_process", resume)

    task = asyncio.create_task(
        shell_module._spawn_process_tree(  # noqa: SLF001
            ["powershell.exe", "-Command", "echo never"],
            cwd=tmp_path,
            env={},
        )
    )
    await spawn_started.wait()
    task.cancel()
    release_spawn.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.kills == 1
    assert resume_calls == 0


@pytest.mark.asyncio
async def test_windows_job_failure_kills_process_without_resuming(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    class FakeProcess:
        pid = 4399
        returncode = None

        def __init__(self) -> None:
            self.kills = 0

        def kill(self) -> None:
            self.kills += 1
            self.returncode = -9

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process = FakeProcess()

    async def fake_exec(*argv, **options):
        del argv, options
        return process

    class FailedJob:
        def __init__(self, pid: int) -> None:
            assert pid == process.pid
            self.handle = None

    def must_not_resume(pid: int) -> None:
        del pid
        raise AssertionError("an uncontained process must not be resumed")

    monkeypatch.setattr(shell_module.sys, "platform", "win32")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(shell_module, "_WindowsJob", FailedJob)
    monkeypatch.setattr(
        shell_module, "_resume_windows_process", must_not_resume
    )

    with pytest.raises(shell_module._ContainmentError):  # noqa: SLF001
        await shell_module._spawn_process_tree(  # noqa: SLF001
            ["powershell.exe", "-Command", "echo never"],
            cwd=tmp_path,
            env={},
        )
    assert process.kills == 1


@pytest.mark.asyncio
async def test_cancel_during_windows_job_failure_cleanup_stays_cancelled(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    class FakeProcess:
        pid = 4400
        returncode = None

        def __init__(self) -> None:
            self.kills = 0

        def kill(self) -> None:
            self.kills += 1

        async def wait(self) -> int:
            cleanup_started.set()
            await release_cleanup.wait()
            self.returncode = -9
            return self.returncode

    process = FakeProcess()

    async def fake_exec(*argv, **options):
        del argv, options
        return process

    class FailedJob:
        def __init__(self, pid: int) -> None:
            assert pid == process.pid
            self.handle = None

    monkeypatch.setattr(shell_module.sys, "platform", "win32")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(shell_module, "_WindowsJob", FailedJob)

    task = asyncio.create_task(
        shell_module._spawn_process_tree(  # noqa: SLF001
            ["powershell.exe", "-Command", "echo never"],
            cwd=tmp_path,
            env={},
        )
    )
    await cleanup_started.wait()
    task.cancel()
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.kills == 1
    assert process.returncode == -9


@pytest.mark.asyncio
async def test_posix_exited_leader_cleanup_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[int, int]] = []

    class ExitedProcess:
        pid = 4444
        returncode = 0

        async def wait(self) -> int:
            return 0

    def killpg(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(shell_module.sys, "platform", "linux")
    monkeypatch.setattr(shell_module.os, "killpg", killpg, raising=False)
    tree = shell_module._ProcessTree(ExitedProcess(), None)  # type: ignore[arg-type]  # noqa: SLF001

    await tree.terminate()
    await tree.terminate()

    assert calls == [
        (4444, signal.SIGTERM),
        (4444, 0),
    ]


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "win32", reason="requires a real suspended PowerShell"
)
async def test_windows_resume_failure_never_executes_real_powershell(
    ctx, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    marker = tmp_path / "shell-must-not-run"

    def fail_resume(pid: int) -> None:
        del pid
        raise OSError("simulated ResumeThread failure")

    monkeypatch.setattr(shell_module, "_resume_windows_process", fail_resume)
    command = f"Set-Content -LiteralPath '{marker}' -Value ran"

    result = await ShellTool().run({"command": command}, ctx)
    await asyncio.sleep(0.1)

    assert result.is_error
    assert "收容失败" in result.content
    assert not marker.exists()
