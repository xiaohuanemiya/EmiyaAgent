"""非交互执行与敏感输入的安全契约。"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
from prompt_toolkit.history import DummyHistory

import emiya_agent.ui.repl as repl_module
from emiya_agent.agent.loop import LoopResult
from emiya_agent.config import Config, load_config
from emiya_agent.ui import commands
from emiya_agent.ui.repl import Repl


@pytest.mark.asyncio
async def test_one_shot_does_not_implicitly_disable_approval(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    repl = Repl(load_config(tmp_path))
    observed: list[tuple[str, str]] = []

    async def no_connect():
        return None

    async def execute(text: str):
        observed.append((text, repl.config.tool_approval))
        return "done"

    async def no_stop():
        return None

    monkeypatch.setattr(repl, "_connect_mcp", no_connect)
    monkeypatch.setattr(repl, "_execute_turn", execute)
    monkeypatch.setattr(repl.mcp, "stop_all", no_stop)

    assert await repl.one_shot("task") is True
    assert observed == [("task", "ask")]


@pytest.mark.parametrize(
    "text",
    [
        "你好",
        "请重构认证模块并补充测试",
        "fix the login bug",
    ],
)
@pytest.mark.asyncio
async def test_repl_routes_all_plain_input_to_main_agent(
    tmp_path, monkeypatch: pytest.MonkeyPatch, text: str
):
    repl = Repl(load_config(tmp_path))
    observed: list[str] = []

    async def execute(raw: str):
        observed.append(raw)
        return "done"

    monkeypatch.setattr(repl, "_execute_turn", execute)

    await repl.handle_input(text)

    assert observed == [text]


def test_legacy_language_and_spec_commands_are_removed():
    assert "lang" not in commands.COMMANDS
    assert "spec" not in commands.COMMANDS
    assert not hasattr(Repl, "spec_turn")


@pytest.mark.asyncio
async def test_model_command_updates_executor_target_route(quiet_console):
    class FakeProviders:
        def get(self, route: str):
            return object(), route.split("/", 1)[-1]

    class FakeRepl:
        console = quiet_console
        providers = FakeProviders()

        def __init__(self) -> None:
            self.route = "openai/old"
            self.config = Config(routes={"executor": self.route})

    repl = FakeRepl()
    assert await commands.dispatch(repl, "/model deepseek/new") is False  # type: ignore[arg-type]
    assert repl.route == "deepseek/new"
    assert repl.config.route("executor") == "deepseek/new"


@pytest.mark.asyncio
async def test_one_shot_reports_execution_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    repl = Repl(load_config(tmp_path))

    async def no_connect():
        return None

    async def fail(_text: str):
        return None

    async def no_stop():
        return None

    monkeypatch.setattr(repl, "_connect_mcp", no_connect)
    monkeypatch.setattr(repl, "_execute_turn", fail)
    monkeypatch.setattr(repl.mcp, "stop_all", no_stop)

    assert await repl.one_shot("task") is False


def test_protocol_error_is_reported_as_execution_failure(quiet_console):
    repl = object.__new__(Repl)
    repl.console = quiet_console

    result = repl._report_result(LoopResult(final_text="partial", stop_reason="error"))

    assert result is None
    assert "非法" in quiet_console.file.getvalue()


@pytest.mark.asyncio
async def test_secret_prompt_is_masked_and_uses_no_disk_history(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    repl = Repl(load_config(tmp_path))
    repl.session = object()  # type: ignore[assignment]
    captured = {}

    class FakePromptSession:
        def __init__(self, *, history):
            captured["history"] = history

        async def prompt_async(self, prompt, *, is_password):
            captured["prompt"] = prompt
            captured["is_password"] = is_password
            return "secret-value"

    monkeypatch.setattr(repl_module, "PromptSession", FakePromptSession)
    monkeypatch.setattr(repl_module, "patch_stdout", nullcontext)

    assert await repl.read_secret("key > ") == "secret-value"
    assert isinstance(captured["history"], DummyHistory)
    assert captured["is_password"] is True
