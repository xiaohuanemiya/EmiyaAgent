"""生命周期 Hook：UserPromptSubmit 拦截、matcher 主题语义、观测型事件接线。"""

from __future__ import annotations

import json
import sys

import pytest

from conftest import FakeProvider, assistant_turn
from emiya_agent.config import Config
from emiya_agent.hooks import (
    DECISION_EVENTS,
    HookCommandConfig,
    HookEvent,
)
from emiya_agent.types import TextBlock, user_message
from emiya_agent.ui.render import QuietRenderer
from emiya_agent.ui.repl import Repl

DENY_SOURCE = 'print(\'{"decision":"deny","reason":"blocked by test"}\')'
MARKER_SOURCE = (
    "import pathlib, sys; "
    "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')"
)


def _hook(
    event: HookEvent,
    source: str,
    *arguments: str,
    matcher: str = ".*",
    on_error: str = "warn",
) -> HookCommandConfig:
    return HookCommandConfig(
        id=f"hook-{event.value.lower()}",
        event=event,
        matcher=matcher,
        command=[sys.executable, "-I", "-c", source, *arguments],
        timeout=30,
        on_error=on_error,
    )


class _Lookup:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider

    def get(self, route: str):
        return self.provider, "model"


def _repl(tmp_path, hooks: list[HookCommandConfig], provider: FakeProvider) -> Repl:
    config = Config(
        cwd=tmp_path,
        routes={
            role: "fake/model"
            for role in (
                "executor",
                "translator",
                "compactor",
                "subagent_explore",
                "subagent_plan",
                "subagent_general",
            )
        },
        hooks=hooks,
    )
    repl = Repl(config)
    repl.providers = _Lookup(provider)
    repl.route = "fake/model"
    repl.renderer = QuietRenderer()
    return repl


def test_lifecycle_events_are_valid_config_values() -> None:
    for event in (
        "UserPromptSubmit",
        "SessionStart",
        "SessionEnd",
        "Stop",
        "SubagentStop",
        "PreCompact",
        "PostCompact",
    ):
        config = HookCommandConfig(
            id=f"cfg-{event}", event=event, command=[sys.executable]
        )
        assert config.event == HookEvent(event)


def test_decision_events_are_exactly_the_gating_set() -> None:
    assert DECISION_EVENTS == {
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.PRE_COMPACT,
    }


@pytest.mark.asyncio
async def test_user_prompt_submit_deny_blocks_prompt(tmp_path) -> None:
    provider = FakeProvider([])  # 任何请求都会触发脚本耗尽断言
    repl = _repl(
        tmp_path, [_hook(HookEvent.USER_PROMPT_SUBMIT, DENY_SOURCE)], provider
    )

    result = await repl._execute_turn("delete everything")

    assert result is None
    assert repl.messages == []
    assert provider.requests == []
    assert repl.last_error is not None
    assert "blocked by test" in repl.last_error


@pytest.mark.asyncio
async def test_user_prompt_submit_matcher_targets_prompt_text(tmp_path) -> None:
    provider = FakeProvider([assistant_turn([TextBlock(text="done")])])
    repl = _repl(
        tmp_path,
        [_hook(HookEvent.USER_PROMPT_SUBMIT, DENY_SOURCE, matcher="deploy")],
        provider,
    )

    allowed = await repl._execute_turn("hello world")
    assert allowed == "done"
    assert len(provider.requests) == 1

    blocked = await repl._execute_turn("deploy to production")
    assert blocked is None
    assert len(provider.requests) == 1  # 第二次未到达模型


@pytest.mark.asyncio
async def test_stop_hook_observes_stop_reason(tmp_path) -> None:
    marker = tmp_path / "stop-payload.json"
    provider = FakeProvider([assistant_turn([TextBlock(text="done")])])
    repl = _repl(
        tmp_path,
        [_hook(HookEvent.STOP, MARKER_SOURCE, str(marker))],
        provider,
    )

    result = await repl._execute_turn("just answer")

    assert result == "done"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["event"] == "Stop"
    assert payload["tool_name"] == "end_turn"  # 事件主题 = stop reason
    assert payload["payload"]["stop_reason"] == "end_turn"
    assert payload["payload"]["route"] == "fake/model"


@pytest.mark.asyncio
async def test_pre_compact_deny_skips_compaction(tmp_path) -> None:
    provider = FakeProvider([])
    repl = _repl(
        tmp_path, [_hook(HookEvent.PRE_COMPACT, DENY_SOURCE)], provider
    )
    repl.messages[:] = [user_message(f"turn {index}") for index in range(10)]
    before = list(repl.messages)

    await repl.maybe_compact(force=True)

    assert repl.messages == before  # 压缩被 hook 拦截，历史未变


@pytest.mark.asyncio
async def test_one_shot_fires_session_start_and_end(tmp_path) -> None:
    start_marker = tmp_path / "session-start.json"
    end_marker = tmp_path / "session-end.json"
    provider = FakeProvider([assistant_turn([TextBlock(text="ok")])])
    repl = _repl(
        tmp_path,
        [
            _hook(HookEvent.SESSION_START, MARKER_SOURCE, str(start_marker)),
            _hook(HookEvent.SESSION_END, MARKER_SOURCE, str(end_marker)),
        ],
        provider,
    )

    assert await repl.one_shot("say ok") is True

    start_payload = json.loads(start_marker.read_text(encoding="utf-8"))
    assert start_payload["event"] == "SessionStart"
    assert start_payload["tool_name"] == "one-shot"
    end_payload = json.loads(end_marker.read_text(encoding="utf-8"))
    assert end_payload["event"] == "SessionEnd"
    assert end_payload["payload"]["reason"] == "exit"


@pytest.mark.asyncio
async def test_observe_only_stop_deny_does_not_change_result(tmp_path) -> None:
    provider = FakeProvider([assistant_turn([TextBlock(text="unchanged")])])
    repl = _repl(tmp_path, [_hook(HookEvent.STOP, DENY_SOURCE)], provider)

    result = await repl._execute_turn("run")

    assert result == "unchanged"  # Stop 是观测型事件，deny 不改变控制流
    assert repl.last_result is not None
    assert repl.last_result.stop_reason == "end_turn"
