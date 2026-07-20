from __future__ import annotations

import json
from pathlib import Path

import pytest

from emiya_agent.agent.control import RunController, RunLimits
from emiya_agent.config import Config, ProviderConfig
from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.tools.translate import (
    TRANSLATION_SYSTEM,
    TranslateForModelTool,
)
from emiya_agent.types import TextBlock, Usage

from tests.conftest import FakeProvider, FakeUI, assistant_turn


class FakeRegistry:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider
        self.routes: list[str] = []

    def get(self, route: str):
        self.routes.append(route)
        return self.provider, "translator-model"


def translation_config(tmp_path: Path) -> Config:
    return Config(
        cwd=tmp_path,
        routes={
            "executor": "openai/executor",
            "translator": "deepseek/translator",
            "subagent_explore": "deepseek/explorer",
            "subagent_plan": "openai/planner",
            "subagent_general": "openai/general",
        },
        providers={
            "openai": ProviderConfig(api_key_env="OPENAI_API_KEY", lang="en"),
            "deepseek": ProviderConfig(api_key_env="DEEPSEEK_API_KEY", lang="zh"),
        },
    )


def test_translation_spec_exposes_live_target_model_languages(tmp_path: Path):
    config = translation_config(tmp_path)
    provider = FakeProvider([])
    tool = TranslateForModelTool(FakeRegistry(provider), config)
    registry = ToolRegistry()
    registry.register(tool)

    first = registry.specs()[0].description
    assert "plan: openai/planner [en]" in first
    assert "explore: deepseek/explorer [zh]" in first

    config.routes["subagent_plan"] = "deepseek/new-planner"
    second = registry.specs()[0].description
    assert "plan: deepseek/new-planner [zh]" in second
    assert "plan: openai/planner [en]" not in second


@pytest.mark.asyncio
async def test_translation_agent_receives_authoritative_next_model_language(
    tmp_path: Path,
):
    config = translation_config(tmp_path)
    usage = Usage(input_tokens=12, output_tokens=7)
    provider = FakeProvider(
        [assistant_turn([TextBlock(text="Fix the authentication bug")], usage=usage)]
    )
    registry = FakeRegistry(provider)
    observed_usage: list[tuple[str, Usage]] = []
    controller = RunController(config, RunLimits(max_turns=1))
    tool = TranslateForModelTool(
        registry,
        config,
        on_usage=lambda route, value: observed_usage.append((route, value)),
        controller=controller,
    )
    ctx = ToolContext(cwd=tmp_path, config=config, ui=FakeUI())

    result = await tool.run({"target_role": "plan", "text": "修复认证模块的 bug"}, ctx)

    assert result.is_error is False
    assert registry.routes == ["deepseek/translator"]
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.model == "translator-model"
    assert request.system == TRANSLATION_SYSTEM
    assert request.tools == []
    payload = json.loads(request.messages[0].text())
    assert payload == {
        "target_role": "plan",
        "target_model": "openai/planner",
        "target_language": "en",
        "text": "修复认证模块的 bug",
    }
    output = json.loads(result.content)
    assert output["target_language"] == "en"
    assert output["translated_text"] == "Fix the authentication bug"
    assert observed_usage == [("deepseek/translator", usage)]
    assert controller.turns == 1
    assert controller.usage == usage


@pytest.mark.asyncio
async def test_translation_caller_cannot_override_target_language(tmp_path: Path):
    config = translation_config(tmp_path)
    provider = FakeProvider([])
    tool = TranslateForModelTool(FakeRegistry(provider), config)
    ctx = ToolContext(cwd=tmp_path, config=config, ui=FakeUI())

    result = await tool.run(
        {
            "target_role": "plan",
            "target_language": "zh",
            "text": "translate me",
        },
        ctx,
    )

    assert result.is_error is True
    assert "unexpected fields: target_language" in result.content
    assert provider.requests == []
