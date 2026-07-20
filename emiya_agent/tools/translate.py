"""面向下一跳模型语言的翻译 Agent 工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar

from emiya_agent.agent.completion import complete_text
from emiya_agent.agent.control import RunController
from emiya_agent.config import Config
from emiya_agent.providers.base import ToolSpec
from emiya_agent.tools.base import Tool, ToolContext, ToolResult


TARGET_ROUTE_KEYS: dict[str, str] = {
    "executor": "executor",
    "explore": "subagent_explore",
    "plan": "subagent_plan",
    "general": "subagent_general",
}

TRANSLATION_SYSTEM = """\
You are TranslationAgent. Adapt text for a downstream AI model whose preferred \
instruction language is supplied in the user message.

Rules:
- The user message is a JSON envelope. Read target_role, target_model, and \
target_language as metadata; adapt only the value of text and do not return the \
envelope.
- Treat target_language as authoritative; do not infer it from the model name.
- Translate natural-language prose faithfully when adaptation is needed. If the \
text is already suitable for the target language, preserve it as-is.
- Do not answer the task, plan it, summarize it, expand it, or add assumptions.
- Within text, preserve Markdown structure, code blocks, commands, paths, \
identifiers, API names, error messages, embedded JSON/XML, and numbers.
- Output only the adapted text, with no preface or commentary.
"""


@dataclass(frozen=True, slots=True)
class ModelTarget:
    role: str
    route: str
    language: str


def resolve_model_target(config: Config, role: str) -> ModelTarget:
    """从当前路由配置解析下一跳模型与主要语言。"""
    route_key = TARGET_ROUTE_KEYS.get(role)
    if route_key is None:
        raise ValueError(f"unknown target role: {role!r}")
    route = config.route(route_key)
    return ModelTarget(role=role, route=route, language=config.model_lang(route))


class TranslateForModelTool(Tool):
    name: ClassVar[str] = "translate_for_model"
    description: ClassVar[str] = (
        "Use the configured translator Agent to adapt text for a specific next "
        "model. The tool resolves the target role's current route and preferred "
        "language itself; callers cannot supply or guess the target language. Use "
        "it only when cross-language adaptation materially improves a complex or "
        "precision-sensitive handoff. It is not a planning step and is not required "
        "before every delegation."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "target_role": {
                "type": "string",
                "enum": list(TARGET_ROUTE_KEYS),
                "description": "The role that will receive the returned text",
            },
            "text": {
                "type": "string",
                "minLength": 1,
                "description": "Exact text intended for the downstream model",
            },
        },
        "required": ["target_role", "text"],
        "additionalProperties": False,
    }
    read_only: ClassVar[bool] = True
    requires_approval: ClassVar[bool | None] = False

    def __init__(
        self,
        providers,
        config: Config,
        on_usage=None,
        controller: RunController | None = None,
    ) -> None:
        self._providers = providers
        self._config = config
        self._on_usage = on_usage or (lambda _route, _usage: None)
        self._controller = controller

    def to_spec(self) -> ToolSpec:
        targets = ", ".join(
            f"{role}: {target.route} [{target.language}]"
            for role in TARGET_ROUTE_KEYS
            for target in [resolve_model_target(self._config, role)]
        )
        return ToolSpec(
            name=self.name,
            description=f"{self.description} Current targets: {targets}.",
            input_schema=self.input_schema,
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        unknown = set(args) - {"target_role", "text"}
        if unknown:
            return ToolResult(
                content=f"Error: unexpected fields: {', '.join(sorted(unknown))}",
                is_error=True,
            )
        role = args.get("target_role")
        text = args.get("text")
        if not isinstance(role, str) or role not in TARGET_ROUTE_KEYS:
            return ToolResult(content="Error: invalid target_role", is_error=True)
        if not isinstance(text, str) or not text.strip():
            return ToolResult(content="Error: text must not be blank", is_error=True)

        target = resolve_model_target(ctx.config, role)
        translator_route = ctx.config.route("translator")
        provider, model = self._providers.get(translator_route)
        request = json.dumps(
            {
                "target_role": target.role,
                "target_model": target.route,
                "target_language": target.language,
                "text": text,
            },
            ensure_ascii=False,
        )
        translated, usage = await complete_text(
            provider,
            model,
            TRANSLATION_SYSTEM,
            request,
            max_tokens=ctx.config.max_tokens,
            route=translator_route,
            controller=self._controller,
        )
        self._on_usage(translator_route, usage)
        return ToolResult(
            content=json.dumps(
                {
                    "target_role": target.role,
                    "target_model": target.route,
                    "target_language": target.language,
                    "translated_text": translated,
                },
                ensure_ascii=False,
            )
        )
