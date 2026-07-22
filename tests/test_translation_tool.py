from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from emiya_agent.agent.control import RunController, RunLimits
from emiya_agent.agent.specialist_prompts import (
    COMPACT_SYSTEM,
    TRANSLATION_SYSTEM,
)
from emiya_agent.agent.subagent import SubagentSpawner
from emiya_agent.agent.usage import UsageTracker
from emiya_agent.config import Config, ProviderConfig
from emiya_agent.providers.base import MessageDone
from emiya_agent.tools.base import Tool, ToolContext, ToolResult
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.tools.task import SpawnSubagentTool
from emiya_agent.tools.translate import TranslateForModelTool
from emiya_agent.types import Message, TextBlock, Usage
from emiya_agent.ui.render import QuietRenderer

from tests.conftest import FakeProvider, FakeUI, assistant_turn


class FakeRegistry:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider
        self.routes: list[str] = []

    def get(self, route: str):
        self.routes.append(route)
        model = {
            "deepseek/translator": "translator-model",
            "deepseek/compactor": "compactor-model",
        }.get(route, "delegated-model")
        return self.provider, model


class SentinelTool(Tool):
    """A root-only tool that must never leak into an internal specialist."""

    name: ClassVar[str] = "sentinel_root_tool"
    description: ClassVar[str] = "Must not be visible to internal specialists"
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
    }
    read_only: ClassVar[bool] = True
    requires_approval: ClassVar[bool | None] = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(content="sentinel")


def translation_config(tmp_path: Path) -> Config:
    return Config(
        cwd=tmp_path,
        routes={
            "executor": "openai/executor",
            "translator": "deepseek/translator",
            "compactor": "deepseek/compactor",
            "subagent_explore": "deepseek/explorer",
            "subagent_plan": "openai/planner",
            "subagent_general": "openai/general",
        },
        providers={
            "openai": ProviderConfig(api_key_env="OPENAI_API_KEY", lang="en"),
            "deepseek": ProviderConfig(api_key_env="DEEPSEEK_API_KEY", lang="zh"),
        },
    )


def build_spawner(
    config: Config,
    provider: FakeProvider,
    *,
    controller: RunController | None = None,
    shared_context: str = "",
) -> tuple[SubagentSpawner, FakeRegistry, UsageTracker]:
    providers = FakeRegistry(provider)
    usage = UsageTracker(config)
    tools = ToolRegistry()
    tools.register(SentinelTool())
    spawner = SubagentSpawner(
        providers,  # type: ignore[arg-type]
        config,
        tools,
        QuietRenderer(),  # type: ignore[arg-type]
        usage,
        shared_context=shared_context,
        controller=controller,
    )
    return spawner, providers, usage


def tool_context(config: Config) -> ToolContext:
    return ToolContext(cwd=config.cwd, config=config, ui=FakeUI())


def test_translation_spec_exposes_live_target_model_languages(tmp_path: Path):
    config = translation_config(tmp_path)
    provider = FakeProvider([])
    spawner, _, _ = build_spawner(config, provider)
    registry = ToolRegistry()
    registry.register(TranslateForModelTool(spawner, config))

    first = registry.specs()[0].description
    assert "plan: openai/planner [en]" in first
    assert "explore: deepseek/explorer [zh]" in first

    config.routes["subagent_plan"] = "deepseek/new-planner"
    second = registry.specs()[0].description
    assert "plan: deepseek/new-planner [zh]" in second
    assert "plan: openai/planner [en]" not in second


@pytest.mark.asyncio
async def test_translation_runs_as_isolated_internal_subagent(tmp_path: Path):
    config = translation_config(tmp_path)
    used = Usage(input_tokens=12, output_tokens=7)
    provider = FakeProvider(
        [assistant_turn([TextBlock(text="Fix the authentication bug")], usage=used)]
    )
    controller = RunController(config, RunLimits(max_turns=1))
    spawner, providers, usage = build_spawner(
        config,
        provider,
        controller=controller,
        shared_context="SECRET PROJECT CONTEXT MUST NOT LEAK",
    )
    executor_used = Usage(input_tokens=30, output_tokens=10)
    usage.add("openai/executor", executor_used)
    tool = TranslateForModelTool(spawner, config)
    source = "修复认证模块的 bug"

    result = await tool.run(
        {"target_role": "plan", "text": source},
        tool_context(config),
    )

    assert result.is_error is False
    assert providers.routes == ["deepseek/translator"]
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.model == "translator-model"
    assert request.system == TRANSLATION_SYSTEM.rstrip()
    assert "SECRET PROJECT CONTEXT MUST NOT LEAK" not in request.system
    assert request.tools == []
    assert request.effort == "low"
    assert len(request.messages) == 1
    payload = json.loads(request.messages[0].text())
    assert payload == {
        "target_role": "plan",
        "target_model": "openai/planner",
        "target_language": "en",
        "text": source,
    }
    output = json.loads(result.content)
    assert output == {
        "target_role": "plan",
        "target_model": "openai/planner",
        "target_language": "en",
        "translated_text": "Fix the authentication bug",
    }
    assert usage.last_for("deepseek/translator") == Usage()
    assert usage.last_for("deepseek/translator", scope="internal:translator") == used
    assert usage.total(scope="internal:translator") == used
    assert usage.last_for("openai/executor") == executor_used
    assert usage.last_route == "openai/executor"
    assert usage.last_usage == executor_used
    assert usage.total() == executor_used + used
    assert controller.turns == 1
    assert controller.usage == used


@pytest.mark.asyncio
async def test_compactor_runs_as_isolated_internal_subagent(tmp_path: Path):
    config = translation_config(tmp_path)
    used = Usage(input_tokens=20, output_tokens=8)
    provider = FakeProvider(
        [
            assistant_turn(
                [TextBlock(text="## Resume\n- Keep the invariant")], usage=used
            )
        ]
    )
    controller = RunController(config, RunLimits(max_turns=1))
    spawner, providers, usage = build_spawner(
        config,
        provider,
        controller=controller,
        shared_context="SECRET PROJECT CONTEXT MUST NOT LEAK",
    )
    transcript = "user: continue\nassistant: inspecting the repository"

    result = await spawner.spawn_internal(
        "compactor",
        transcript,
        tool_context(config),
    )

    assert result == "## Resume\n- Keep the invariant"
    assert providers.routes == ["deepseek/compactor"]
    request = provider.requests[0]
    assert request.model == "compactor-model"
    assert request.system == COMPACT_SYSTEM.rstrip()
    assert "SECRET PROJECT CONTEXT MUST NOT LEAK" not in request.system
    assert request.messages[0].text() == transcript
    assert request.tools == []
    assert request.effort == "low"
    assert usage.last_for("deepseek/compactor") == Usage()
    assert usage.last_for("deepseek/compactor", scope="internal:compactor") == used
    assert usage.total(scope="internal:compactor") == used
    assert usage.total() == used
    assert controller.turns == 1
    assert controller.usage == used


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["translator", "compactor"])
async def test_internal_roles_are_not_publicly_spawnable(
    tmp_path: Path,
    role: str,
):
    config = translation_config(tmp_path)
    provider = FakeProvider([])
    spawner, providers, _ = build_spawner(config, provider)

    report = await spawner.spawn(role, "do internal work", tool_context(config))

    assert report.startswith("Error: unknown delegated subagent role")
    assert providers.routes == []
    assert provider.requests == []
    public_roles = SpawnSubagentTool.input_schema["properties"]["role"]["enum"]
    assert set(public_roles) == {"explore", "plan", "general"}
    assert role not in public_roles


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn", "expected"),
    [
        (
            MessageDone(
                message=Message(role="assistant", content=[TextBlock(text="partial")]),
                usage=Usage(input_tokens=3, output_tokens=1),
                stop_reason="max_tokens",
            ),
            "translator subagent stopped with max_tokens",
        ),
        (
            MessageDone(
                message=Message(role="assistant", content=[]),
                usage=Usage(input_tokens=3),
                stop_reason="refusal",
                refusal="policy",
            ),
            "translator subagent stopped with refusal: policy",
        ),
        (
            MessageDone(
                message=Message(role="assistant", content=[TextBlock(text="  \n")]),
                usage=Usage(input_tokens=3),
                stop_reason="end_turn",
            ),
            "translator subagent stopped with empty_response",
        ),
    ],
)
async def test_translation_turns_incomplete_specialist_results_into_tool_errors(
    tmp_path: Path,
    turn: MessageDone,
    expected: str,
):
    config = translation_config(tmp_path)
    provider = FakeProvider([turn])
    spawner, _, _ = build_spawner(config, provider)
    tool = TranslateForModelTool(spawner, config)

    result = await tool.run(
        {"target_role": "plan", "text": "translate me"},
        tool_context(config),
    )

    assert result.is_error is True
    assert expected in result.content
    assert len(provider.requests) == 1
    assert provider.requests[0].tools == []


@pytest.mark.asyncio
async def test_translation_caller_cannot_override_target_language(tmp_path: Path):
    config = translation_config(tmp_path)
    provider = FakeProvider([])
    spawner, _, _ = build_spawner(config, provider)
    tool = TranslateForModelTool(spawner, config)

    result = await tool.run(
        {
            "target_role": "plan",
            "target_language": "zh",
            "text": "translate me",
        },
        tool_context(config),
    )

    assert result.is_error is True
    assert "unexpected fields: target_language" in result.content
    assert provider.requests == []
