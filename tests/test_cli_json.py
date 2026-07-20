from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace
from typing import Any

import pytest

import emiya_agent.cli as cli_module
import emiya_agent.ui.repl as repl_module
from emiya_agent.agent.control import ExitCode
from emiya_agent.agent.loop import LoopResult
from emiya_agent.config import Config
from emiya_agent.ui.json_output import JsonEventSink


def _install_fake_repl(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Config:
    config = Config(
        cwd=tmp_path,
        routes={"executor": "fake/test-model"},
    )
    monkeypatch.setattr(cli_module, "load_config", lambda _cwd: config)

    class FakeRepl:
        def __init__(
            self,
            received_config: Config,
            *,
            automation: bool,
            output_mode: str,
            event_sink: JsonEventSink | None,
            controller: Any,
        ) -> None:
            assert received_config is config
            assert automation is True
            assert output_mode in {"json", "stream-json"}
            self.event_sink = event_sink
            self.controller = controller
            self.route = received_config.route("executor")
            self.session_id = "fake-session"
            self.last_result: LoopResult | None = None
            self.last_error: str | None = None
            self.last_error_kind: str | None = None

        async def one_shot(self, prompt: str) -> bool:
            assert prompt == "finish the task"
            self.last_result = LoopResult(
                final_text="done \u2713",
                stop_reason="end_turn",
                served_by="fake-provider",
            )
            return True

    monkeypatch.setattr(repl_module, "Repl", FakeRepl)
    return config


def test_json_mode_writes_exactly_one_clean_json_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_repl(monkeypatch, tmp_path)

    code = cli_module.run(["-p", "finish the task", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["type"] == "final"
    assert payload["status"] == "success"
    assert payload["output"] == "done \u2713"
    assert payload["exit_code"] == 0
    assert captured.out == (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    assert "\x1b" not in captured.out
    assert captured.err == ""


def test_stream_json_is_strict_ndjson_with_monotonic_sequence_numbers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_repl(monkeypatch, tmp_path)

    code = cli_module.run(["-p", "finish the task", "--stream-json"])

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    events = [json.loads(line) for line in lines]
    assert code == 0
    assert len(events) == 2
    assert [event["type"] for event in events] == ["run_start", "final"]
    assert [event["seq"] for event in events] == [1, 2]
    assert all(
        current["seq"] < following["seq"]
        for current, following in zip(events, events[1:])
    )
    assert all(line.startswith("{") and line.endswith("}") for line in lines)
    assert events[-1]["output"] == "done \u2713"
    assert captured.err == ""


@pytest.mark.parametrize(
    "args",
    [
        ["--json"],
        ["--stream-json"],
        ["--max-turns", "1"],
        ["--timeout", "1"],
        ["--max-budget-usd", "1"],
    ],
)
def test_machine_output_and_limits_require_prompt(
    args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.run(args)

    assert exc_info.value.code == int(ExitCode.USAGE)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "require -p/--prompt" in captured.err


@pytest.mark.parametrize(
    ("error_kind", "stop_reason", "timed_out", "expected"),
    [
        ("config", None, False, ExitCode.CONFIG_ERROR),
        ("provider", None, False, ExitCode.PROVIDER_ERROR),
        (None, "refusal", False, ExitCode.REFUSAL),
        (None, "max_turns", False, ExitCode.MAX_TURNS),
        (None, "max_iterations", False, ExitCode.MAX_TURNS),
        (None, "budget_exceeded", False, ExitCode.BUDGET_EXCEEDED),
        (None, "budget_unavailable", False, ExitCode.CONFIG_ERROR),
        (None, "max_tokens", False, ExitCode.MAX_TOKENS),
        (None, "error", False, ExitCode.PROTOCOL_ERROR),
        (None, "timeout", False, ExitCode.TIMEOUT),
        (None, "end_turn", True, ExitCode.TIMEOUT),
        (None, "interrupted", False, ExitCode.INTERRUPTED),
    ],
)
def test_exit_mapping_is_stable(
    error_kind: str | None,
    stop_reason: str | None,
    timed_out: bool,
    expected: ExitCode,
) -> None:
    result = (
        LoopResult(final_text="", stop_reason=stop_reason)
        if stop_reason is not None
        else None
    )
    repl = SimpleNamespace(last_error_kind=error_kind, last_result=result)

    actual = cli_module._exit_for_result(repl, timed_out=timed_out)

    assert actual is expected
    assert int(actual) == int(expected)


@pytest.mark.asyncio
async def test_automation_ask_and_confirmation_fail_closed() -> None:
    stream = io.StringIO()
    repl = object.__new__(repl_module.Repl)
    repl.automation = True
    repl.event_sink = JsonEventSink(stream)

    with pytest.raises(RuntimeError, match="unavailable in one-shot mode"):
        await repl.ask("need input", ["yes", "no"])

    assert await repl.confirm_tool("shell", "dangerous command") is False
    event = json.loads(stream.getvalue())
    assert event["type"] == "approval_required"
    assert event["tool"] == "shell"
    assert event["approved"] is False


@pytest.mark.asyncio
async def test_timeout_wins_when_inner_coroutine_swallows_cancellation() -> None:
    class ExpiredAfterWait:
        def __init__(self) -> None:
            self.reads = 0

        @property
        def remaining_timeout(self) -> float:
            self.reads += 1
            return 0.001 if self.reads == 1 else 0.0

    class CancellationSwallowingRepl:
        async def one_shot(self, _prompt: str) -> bool:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                return True
            raise AssertionError("sleep unexpectedly completed")

    ok, timed_out = await cli_module._one_shot(
        CancellationSwallowingRepl(),
        "prompt",
        ExpiredAfterWait(),
    )

    assert ok is True
    assert timed_out is True
