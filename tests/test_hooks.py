from __future__ import annotations

import asyncio
import os
import sys

import pytest
from pydantic import ValidationError

from conftest import FakeProvider, FakeUI, assistant_turn
import emiya_agent.hooks.runner as hook_runner_module
from emiya_agent.agent.loop import AgentLoop
from emiya_agent.agent.subagent import SubagentSpawner
from emiya_agent.agent.usage import UsageTracker
from emiya_agent.config import Config
from emiya_agent.hooks import (
    HookCommandConfig,
    HookDecision,
    HookDiagnostic,
    HookEvent,
    HookRunner,
)
from emiya_agent.hooks.runner import MAX_HOOK_OUTPUT_BYTES, _HookOutcome
from emiya_agent.tools.base import Tool, ToolContext, ToolResult
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.types import (
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    user_message,
)
from emiya_agent.ui.render import QuietRenderer


def _make_executable(path) -> None:
    path.write_bytes(b"")
    if os.name != "nt":
        path.chmod(0o700)


def test_hook_bare_executable_uses_only_absolute_path_entries(tmp_path) -> None:
    project = tmp_path / "project"
    safe_bin = tmp_path / "safe-bin"
    project.mkdir()
    safe_bin.mkdir()
    filename = "guard.EXE" if os.name == "nt" else "guard"
    malicious = project / filename
    expected = safe_bin / filename
    _make_executable(malicious)
    _make_executable(expected)

    argv = hook_runner_module._resolve_hook_argv(
        ("guard", "--check"),
        cwd=project,
        env={"PATH": str(safe_bin), "PATHEXT": ".EXE"},
    )

    assert argv == (str(expected.resolve()), "--check")


def test_hook_bare_executable_ignores_relative_path_entries(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    filename = "guard.EXE" if os.name == "nt" else "guard"
    _make_executable(project / filename)

    with pytest.raises(FileNotFoundError, match="absolute allowed PATH"):
        hook_runner_module._resolve_hook_argv(
            ("guard",),
            cwd=project,
            env={"PATH": ".", "PATHEXT": ".EXE"},
        )


class _MarkTool(Tool):
    name = "mark"
    description = "record execution"
    input_schema = {"type": "object", "properties": {}}
    read_only = True

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, args, ctx) -> ToolResult:
        self.calls += 1
        return ToolResult(content="executed")


class _TouchTool(Tool):
    name = "touch"
    description = "change policy-visible state"
    input_schema = {"type": "object", "properties": {}}
    read_only = False

    def __init__(self, marker) -> None:
        self.marker = marker
        self.calls = 0

    async def run(self, args, ctx) -> ToolResult:
        self.calls += 1
        self.marker.write_text("changed", encoding="utf-8")
        return ToolResult(content="changed")


class _ProviderLookup:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider

    def get(self, route: str):
        assert route == "fake/model"
        return self.provider, "model"


def _tool_loop(tmp_path, hook: HookCommandConfig, tool: _MarkTool):
    provider = FakeProvider(
        [
            assistant_turn(
                [ToolUseBlock(id="call-1", name="mark", input={})],
                stop_reason="tool_use",
            ),
            assistant_turn([TextBlock(text="done")]),
        ]
    )
    registry = ToolRegistry()
    registry.register(tool)
    runner = HookRunner([hook], cwd=tmp_path, env={})
    loop = AgentLoop(
        provider,
        "model",
        "fake/model",
        registry,
        QuietRenderer().render,
        hooks=runner,
    )
    config = Config(cwd=tmp_path, tool_approval="auto")
    ui = FakeUI(approvals=[False])
    return loop, provider, ToolContext(cwd=tmp_path, config=config, ui=ui), ui


def _python_hook(
    hook_id: str,
    source: str,
    *arguments: str,
    matcher: str = ".*",
    timeout: float = 2,
    on_error: str = "warn",
) -> HookCommandConfig:
    return HookCommandConfig(
        id=hook_id,
        event=HookEvent.PRE_TOOL_USE,
        matcher=matcher,
        command=[sys.executable, "-I", "-c", source, *arguments],
        timeout=timeout,
        on_error=on_error,
    )


def _runner(tmp_path, *configs, env=None) -> HookRunner:
    return HookRunner(configs, cwd=tmp_path, env=env or {})


def test_hook_config_validates_protocol_fields() -> None:
    with pytest.raises(ValidationError):
        HookCommandConfig(
            id="bad",
            event=HookEvent.PRE_TOOL_USE,
            matcher="[",
            command=[sys.executable],
        )
    with pytest.raises(ValidationError):
        HookCommandConfig(
            id="bad",
            event=HookEvent.PRE_TOOL_USE,
            command=[],
        )
    with pytest.raises(ValidationError):
        HookCommandConfig(
            id="bad",
            event=HookEvent.PRE_TOOL_USE,
            command=[sys.executable],
            timeout=121,
        )


async def test_allow_never_elevates_existing_ask(tmp_path) -> None:
    config = _python_hook(
        "allow",
        "print('{\"decision\":\"allow\",\"reason\":\"looks safe\"}')",
    )
    result = await _runner(tmp_path, config).run(
        HookEvent.PRE_TOOL_USE,
        "shell",
        {"command": "echo ok"},
        initial_decision=HookDecision.ASK,
        initial_reason="normal approval is required",
    )

    assert result.decision == HookDecision.ASK
    assert result.reason == "normal approval is required"
    assert result.diagnostics[0].decision == HookDecision.ALLOW


async def test_json_deny_and_exit_two_both_block(tmp_path) -> None:
    json_deny = _python_hook(
        "policy",
        "print('{\"decision\":\"deny\",\"reason\":\"policy denied\"}')",
    )
    exit_two = _python_hook(
        "command",
        "import sys; print('command denied', file=sys.stderr); raise SystemExit(2)",
    )
    result = await _runner(tmp_path, json_deny, exit_two).run(
        HookEvent.PRE_TOOL_USE,
        "write",
        {},
        initial_decision=HookDecision.ALLOW,
    )

    assert result.decision == HookDecision.DENY
    assert result.reason == "policy denied; command denied"
    assert [item.status for item in result.diagnostics] == ["blocked", "blocked"]


async def test_nonzero_exit_honors_warn_and_block(tmp_path) -> None:
    source = "raise SystemExit(9)"
    warning = _python_hook("warning", source, on_error="warn")
    blocking = _python_hook("blocking", source, on_error="block")
    result = await _runner(tmp_path, warning, blocking).run(
        HookEvent.PRE_TOOL_USE,
        "shell",
        {},
        initial_decision=HookDecision.ALLOW,
    )

    assert result.decision == HookDecision.DENY
    assert result.diagnostics[0].status == "warning"
    assert result.diagnostics[1].status == "error"


async def test_timeout_honors_block_policy(tmp_path) -> None:
    config = _python_hook(
        "slow",
        "import time; time.sleep(10)",
        timeout=0.05,
        on_error="block",
    )
    result = await _runner(tmp_path, config).run(
        HookEvent.PRE_TOOL_USE,
        "shell",
        {},
        initial_decision=HookDecision.ALLOW,
    )

    assert result.decision == HookDecision.DENY
    assert result.diagnostics[0].status == "timeout"
    assert "timed out" in result.diagnostics[0].reason


async def test_runner_uses_only_explicit_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EMIYA_HOOK_HIDDEN", "must-not-leak")
    source = (
        "import json,os; print(json.dumps({"
        "'decision':'ask','reason':os.getenv('EMIYA_HOOK_VISIBLE','')+'|'"
        "+str(os.getenv('EMIYA_HOOK_HIDDEN'))}))"
    )
    config = _python_hook("environment", source)
    result = await _runner(
        tmp_path,
        config,
        env={"EMIYA_HOOK_VISIBLE": "passed"},
    ).run(
        HookEvent.PRE_TOOL_USE,
        "read",
        {},
        initial_decision=HookDecision.ALLOW,
    )

    assert result.decision == HookDecision.ASK
    assert result.reason == "passed|None"


async def test_command_arguments_are_literal_and_stdin_is_json(tmp_path) -> None:
    literal = "; echo this-is-not-a-shell"
    source = (
        "import json,sys; data=json.load(sys.stdin); "
        "print(json.dumps({'decision':'ask','reason':"
        "sys.argv[1]+'|'+data['tool_name']+'|'+data['payload']['value']}))"
    )
    config = _python_hook("literal", source, literal)
    result = await _runner(tmp_path, config).run(
        HookEvent.PRE_TOOL_USE,
        "edit",
        {"value": "stdin-ok"},
        initial_decision=HookDecision.ALLOW,
    )

    assert result.reason == f"{literal}|edit|stdin-ok"
    assert not (tmp_path / "this-is-not-a-shell").exists()


async def test_stdout_and_stderr_are_independently_bounded(tmp_path) -> None:
    source = (
        "import os; "
        "os.write(1,b'x'*70000); os.write(2,b'y'*71000)"
    )
    config = _python_hook("bounded", source, on_error="warn")
    result = await _runner(tmp_path, config).run(
        HookEvent.PRE_TOOL_USE,
        "shell",
        {},
        initial_decision=HookDecision.ALLOW,
    )

    diagnostic = result.diagnostics[0]
    assert len(diagnostic.stdout.encode()) == MAX_HOOK_OUTPUT_BYTES
    assert len(diagnostic.stderr.encode()) == MAX_HOOK_OUTPUT_BYTES
    assert diagnostic.stdout_omitted_bytes == 70000 - MAX_HOOK_OUTPUT_BYTES
    assert diagnostic.stderr_omitted_bytes == 71000 - MAX_HOOK_OUTPUT_BYTES
    assert diagnostic.status == "warning"


async def test_matching_hooks_run_concurrently_but_report_config_order(
    tmp_path, monkeypatch
) -> None:
    slow = _python_hook("first", "pass", matcher="shell|write")
    fast = _python_hook("second", "pass", matcher="shell")
    skipped = _python_hook("skipped", "raise SystemExit(2)", matcher="read")
    both_started = asyncio.Event()
    release = asyncio.Event()
    started_ids: list[str] = []

    async def fake_run_one(self, config, event, stdin):
        del self, stdin
        started_ids.append(config.id)
        if len(started_ids) == 2:
            both_started.set()
        await release.wait()
        return _HookOutcome(
            HookDecision.ASK,
            HookDiagnostic(
                hook_id=config.id,
                event=event,
                status="ok",
                decision=HookDecision.ASK,
                reason=config.id,
                exit_code=0,
            ),
        )

    monkeypatch.setattr(HookRunner, "_run_one", fake_run_one)

    task = asyncio.create_task(
        _runner(tmp_path, slow, fast, skipped).run(
            HookEvent.PRE_TOOL_USE,
            "shell",
            {},
            initial_decision=HookDecision.ALLOW,
        )
    )
    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
    finally:
        release.set()
    result = await task

    assert started_ids == ["first", "second"]
    assert [item.hook_id for item in result.diagnostics] == ["first", "second"]
    assert result.reason == "first; second"


async def test_empty_stdout_is_valid_noop_and_malformed_json_warns(tmp_path) -> None:
    noop = _python_hook("noop", "pass")
    malformed = _python_hook("malformed", "print('not json')", on_error="warn")
    result = await _runner(tmp_path, noop, malformed).run(
        HookEvent.PRE_TOOL_USE,
        "shell",
        {},
        initial_decision=HookDecision.ALLOW,
    )

    assert result.decision == HookDecision.ALLOW
    assert result.diagnostics[0].status == "ok"
    assert result.diagnostics[1].status == "warning"


async def test_cancellation_terminates_hook(tmp_path) -> None:
    marker = tmp_path / "should-not-exist"
    started = tmp_path / "child-started"
    child_source = (
        "import pathlib,sys,time; time.sleep(.6); "
        "pathlib.Path(sys.argv[1]).write_text('leaked')"
    )
    source = (
        "import pathlib,subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-I','-c',sys.argv[3],sys.argv[1]]); "
        "pathlib.Path(sys.argv[2]).write_text('ready'); time.sleep(10)"
    )
    config = _python_hook(
        "cancel",
        source,
        str(marker),
        str(started),
        child_source,
        timeout=10,
    )
    task = asyncio.create_task(
        _runner(tmp_path, config).run(
            HookEvent.PRE_TOOL_USE,
            "shell",
            {},
            initial_decision=HookDecision.ALLOW,
        )
    )
    for _ in range(100):
        if started.exists():
            break
        await asyncio.sleep(0.01)
    assert started.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.8)

    assert not marker.exists()


async def test_clean_exit_terminates_redirected_background_child(tmp_path) -> None:
    marker = tmp_path / "background-should-not-survive"
    child_source = (
        "import pathlib,sys,time; time.sleep(.6); "
        "pathlib.Path(sys.argv[1]).write_text('leaked')"
    )
    source = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-I','-c',sys.argv[2],sys.argv[1]],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL,close_fds=True)"
    )
    config = _python_hook(
        "background",
        source,
        str(marker),
        child_source,
    )

    result = await _runner(tmp_path, config).run(
        HookEvent.PRE_TOOL_USE,
        "shell",
        {},
        initial_decision=HookDecision.ALLOW,
    )
    await asyncio.sleep(0.8)

    assert result.decision == HookDecision.ALLOW
    assert not marker.exists()


async def test_nonfinite_hook_payload_honors_block_policy(tmp_path) -> None:
    hook = HookCommandConfig(
        id="strict-json",
        event=HookEvent.POST_TOOL_USE,
        command=[sys.executable, "-I", "-c", "pass"],
        on_error="block",
    )

    result = await _runner(tmp_path, hook).run(
        HookEvent.POST_TOOL_USE,
        "mark",
        {"value": float("nan")},
        initial_decision=HookDecision.ALLOW,
    )

    assert result.decision == HookDecision.DENY
    assert result.diagnostics[0].status == "error"
    assert "not strict JSON" in result.reason


async def test_agent_loop_pre_hook_denies_without_executing_tool(tmp_path) -> None:
    hook = _python_hook(
        "deny-mark",
        "print('{\"decision\":\"deny\",\"reason\":\"policy\"}')",
        matcher="^mark$",
    )
    tool = _MarkTool()
    loop, provider, ctx, _ui = _tool_loop(tmp_path, hook, tool)
    messages = [user_message("run")]

    result = await loop.run(messages, system="system", ctx=ctx)

    assert result.final_text == "done"
    assert tool.calls == 0
    tool_result = provider.requests[1].messages[-1].content[0]
    assert tool_result.is_error
    assert "PreToolUse hook" in tool_result.content


async def test_agent_loop_pre_hook_can_raise_allow_to_ask(tmp_path) -> None:
    hook = _python_hook(
        "ask-mark",
        "print('{\"decision\":\"ask\",\"reason\":\"review\"}')",
        matcher="^mark$",
    )
    tool = _MarkTool()
    loop, _provider, ctx, ui = _tool_loop(tmp_path, hook, tool)

    await loop.run([user_message("run")], system="system", ctx=ctx)

    assert tool.calls == 0
    assert ui.approval_requests == ["mark"]


async def test_agent_loop_post_hook_reports_error_after_execution(tmp_path) -> None:
    hook = HookCommandConfig(
        id="post-check",
        event=HookEvent.POST_TOOL_USE,
        matcher="^mark$",
        command=[
            sys.executable,
            "-I",
            "-c",
            "print('{\"decision\":\"deny\",\"reason\":\"bad output\"}')",
        ],
    )
    tool = _MarkTool()
    loop, provider, ctx, _ui = _tool_loop(tmp_path, hook, tool)

    await loop.run([user_message("run")], system="system", ctx=ctx)

    assert tool.calls == 1
    tool_result = provider.requests[1].messages[-1].content[0]
    assert tool_result.is_error
    assert "already executed" in tool_result.content
    assert "bad output" in tool_result.content


async def test_mixed_batch_pre_hook_observes_prior_write_effects(tmp_path) -> None:
    marker = tmp_path / "policy-state"
    hook = _python_hook(
        "state-policy",
        (
            "import pathlib,sys; "
            "print('{\"decision\":\"deny\",\"reason\":\"state changed\"}' "
            "if pathlib.Path(sys.argv[1]).exists() else "
            "'{\"decision\":\"allow\"}')"
        ),
        str(marker),
        matcher="^mark$",
    )
    touch = _TouchTool(marker)
    mark = _MarkTool()
    provider = FakeProvider(
        [
            assistant_turn(
                [
                    ToolUseBlock(id="first", name="touch", input={}),
                    ToolUseBlock(id="second", name="mark", input={}),
                ],
                stop_reason="tool_use",
            ),
            assistant_turn([TextBlock(text="done")]),
        ]
    )
    registry = ToolRegistry()
    registry.register(touch)
    registry.register(mark)
    loop = AgentLoop(
        provider,
        "model",
        "fake/model",
        registry,
        QuietRenderer().render,
        hooks=HookRunner([hook], cwd=tmp_path, env={}),
    )
    ctx = ToolContext(
        cwd=tmp_path,
        config=Config(cwd=tmp_path, tool_approval="auto"),
        ui=FakeUI(),
    )

    result = await loop.run([user_message("run")], system="system", ctx=ctx)

    assert result.final_text == "done"
    assert touch.calls == 1
    assert mark.calls == 0
    second_result = provider.requests[1].messages[-1].content[1]
    assert second_result.is_error
    assert "state changed" in second_result.content


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_agent_loop_rejects_nonfinite_tool_input(tmp_path, value) -> None:
    tool = _MarkTool()
    provider = FakeProvider(
        [
            assistant_turn(
                [ToolUseBlock(id="bad", name="mark", input={"value": value})],
                stop_reason="tool_use",
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(tool)
    loop = AgentLoop(
        provider,
        "model",
        "fake/model",
        registry,
        QuietRenderer().render,
    )
    messages = [user_message("run")]
    ctx = ToolContext(
        cwd=tmp_path,
        config=Config(cwd=tmp_path, tool_approval="auto"),
        ui=FakeUI(),
    )

    result = await loop.run(messages, system="system", ctx=ctx)

    assert result.stop_reason == "error"
    assert tool.calls == 0
    assert messages == [user_message("run")]


async def test_post_hook_infrastructure_error_honors_block(
    tmp_path, monkeypatch
) -> None:
    hook = HookCommandConfig(
        id="post-block",
        event=HookEvent.POST_TOOL_USE,
        matcher="^mark$",
        command=[sys.executable, "-I", "-c", "pass"],
        on_error="block",
    )
    tool = _MarkTool()
    loop, provider, ctx, _ui = _tool_loop(tmp_path, hook, tool)
    original_run = loop.hooks.run

    async def broken_run(event, *args, **kwargs):
        if event == HookEvent.POST_TOOL_USE:
            raise RuntimeError("runner broke")
        return await original_run(event, *args, **kwargs)

    monkeypatch.setattr(loop.hooks, "run", broken_run)
    await loop.run([user_message("run")], system="system", ctx=ctx)

    assert tool.calls == 1
    tool_result = provider.requests[1].messages[-1].content[0]
    assert tool_result.is_error
    assert "infrastructure error" in tool_result.content


async def test_injected_hooks_apply_inside_subagent(tmp_path) -> None:
    hook = _python_hook(
        "deny-subagent-mark",
        "print('{\"decision\":\"deny\",\"reason\":\"subagent policy\"}')",
        matcher="^mark$",
    )
    tool = _MarkTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = FakeProvider(
        [
            assistant_turn(
                [ToolUseBlock(id="sub-call", name="mark", input={})],
                stop_reason="tool_use",
            ),
            assistant_turn([TextBlock(text="done")]),
        ]
    )
    config = Config(
        cwd=tmp_path,
        tool_approval="auto",
        routes={"subagent_explore": "fake/model"},
    )
    spawner = SubagentSpawner(
        _ProviderLookup(provider),
        config,
        registry,
        QuietRenderer(),
        UsageTracker(config),
        hooks=HookRunner([hook], cwd=tmp_path, env={}),
    )
    ctx = ToolContext(cwd=tmp_path, config=config, ui=FakeUI())

    report = await spawner.spawn("explore", "run", ctx, quiet=True)

    assert report == "done"
    assert tool.calls == 0


async def test_tool_result_observer_failure_preserves_protocol_pairing(
    tmp_path,
) -> None:
    hook = _python_hook("noop", "pass", matcher="^mark$")
    tool = _MarkTool()
    loop, provider, ctx, _ui = _tool_loop(tmp_path, hook, tool)

    def broken_observer(call, result) -> None:
        del call, result
        raise BrokenPipeError("event stream closed")

    loop.on_tool_result = broken_observer
    messages = [user_message("run")]

    result = await loop.run(messages, system="system", ctx=ctx)

    assert result.final_text == "done"
    assert tool.calls == 1
    paired = provider.requests[1].messages[-1]
    assert paired.role == "user"
    assert isinstance(paired.content[0], ToolResultBlock)
    assert paired.content[0].tool_use_id == "call-1"


async def test_hook_notification_failure_preserves_protocol_pairing(
    tmp_path,
) -> None:
    hook = _python_hook(
        "warning",
        "print('not json')",
        matcher="^mark$",
        on_error="warn",
    )
    tool = _MarkTool()
    loop, provider, ctx, _ui = _tool_loop(tmp_path, hook, tool)

    def broken_notify(text: str) -> None:
        del text
        raise BrokenPipeError("terminal closed")

    ctx.ui.notify = broken_notify

    result = await loop.run(
        [user_message("run")], system="system", ctx=ctx
    )

    assert result.final_text == "done"
    assert tool.calls == 1
    paired = provider.requests[1].messages[-1]
    assert paired.role == "user"
    assert paired.content[0].tool_use_id == "call-1"


@pytest.mark.parametrize("invalid_part", ["id", "name", "input"])
async def test_agent_loop_rejects_non_utf8_json_strings(
    tmp_path, invalid_part
) -> None:
    tool = _MarkTool()
    call = ToolUseBlock(
        id="\ud800" if invalid_part == "id" else "call-1",
        name="\ud800" if invalid_part == "name" else "mark",
        input={"nested": ["\ud800"]} if invalid_part == "input" else {},
    )
    provider = FakeProvider(
        [assistant_turn([call], stop_reason="tool_use")]
    )
    registry = ToolRegistry()
    registry.register(tool)
    loop = AgentLoop(
        provider,
        "model",
        "fake/model",
        registry,
        QuietRenderer().render,
    )
    messages = [user_message("run")]
    ctx = ToolContext(
        cwd=tmp_path,
        config=Config(cwd=tmp_path, tool_approval="auto"),
        ui=FakeUI(),
    )

    result = await loop.run(messages, system="system", ctx=ctx)

    assert result.stop_reason == "error"
    assert tool.calls == 0
    assert messages == [user_message("run")]


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows Job Object containment only"
)
async def test_windows_containment_failure_never_starts_hook(
    tmp_path, monkeypatch
) -> None:
    marker = tmp_path / "hook-must-not-run"

    class FailedJob:
        def __init__(self, pid: int) -> None:
            del pid
            self.handle = None

        def close(self) -> None:
            return None

    monkeypatch.setattr(hook_runner_module, "_WindowsJob", FailedJob)
    config = _python_hook(
        "uncontained",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ran')",
        str(marker),
        on_error="warn",
    )

    result = await _runner(tmp_path, config).run(
        HookEvent.PRE_TOOL_USE,
        "shell",
        {},
        initial_decision=HookDecision.ALLOW,
    )
    await asyncio.sleep(0.1)

    assert result.decision == HookDecision.DENY
    assert "containment failed" in result.reason
    assert not marker.exists()


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows Job Object containment only"
)
async def test_windows_resume_failure_closes_job_and_never_starts_hook(
    tmp_path, monkeypatch
) -> None:
    marker = tmp_path / "resume-failure-must-not-run"
    real_job = hook_runner_module._WindowsJob
    jobs = []

    class TrackingJob(real_job):
        def __init__(self, pid: int) -> None:
            super().__init__(pid)
            jobs.append(self)

    def fail_resume(pid: int) -> None:
        del pid
        raise OSError("simulated ResumeThread failure")

    monkeypatch.setattr(hook_runner_module, "_WindowsJob", TrackingJob)
    monkeypatch.setattr(
        hook_runner_module, "_resume_windows_process", fail_resume
    )
    config = _python_hook(
        "resume-failure",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ran')",
        str(marker),
        on_error="warn",
    )

    result = await _runner(tmp_path, config).run(
        HookEvent.PRE_TOOL_USE,
        "shell",
        {},
        initial_decision=HookDecision.ALLOW,
    )
    await asyncio.sleep(0.1)

    assert result.decision == HookDecision.DENY
    assert "could not resume" in result.reason
    assert jobs and jobs[0].handle is None
    assert not marker.exists()


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows suspended-spawn cancellation only"
)
async def test_windows_spawn_cancellation_reaps_suspended_hook(
    tmp_path, monkeypatch
) -> None:
    marker = tmp_path / "cancelled-spawn-must-not-run"
    spawned = asyncio.Event()
    release = asyncio.Event()
    processes = []
    real_spawn = asyncio.create_subprocess_exec
    delay_first_spawn = True

    async def delayed_spawn(*args, **kwargs):
        nonlocal delay_first_spawn
        process = await real_spawn(*args, **kwargs)
        if delay_first_spawn:
            delay_first_spawn = False
            processes.append(process)
            spawned.set()
            await release.wait()
        return process

    monkeypatch.setattr(
        hook_runner_module.asyncio,
        "create_subprocess_exec",
        delayed_spawn,
    )
    config = _python_hook(
        "cancel-spawn",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ran')",
        str(marker),
    )
    task = asyncio.create_task(
        _runner(tmp_path, config).run(
            HookEvent.PRE_TOOL_USE,
            "shell",
            {},
            initial_decision=HookDecision.ALLOW,
        )
    )
    await asyncio.wait_for(spawned.wait(), timeout=2)

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)

    assert processes and processes[0].returncode is not None
    assert not marker.exists()
