"""子 agent：隔离上下文的角色化执行单元。

- 可委派角色：explore、plan、general
- 内部专用角色：translator、compactor（无工具、单轮、静默）

深度限制为 1：子 agent 不能再生成子 agent（工具子集里排除 spawn_subagent）。
根 executor 不属于子 agent；它持有顶层会话、工具编排与生命周期。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from emiya_agent.agent.control import RunController
from emiya_agent.agent.loop import AgentLoop, LoopResult
from emiya_agent.agent.specialist_prompts import (
    COMPACT_SYSTEM,
    TRANSLATION_SYSTEM,
)
from emiya_agent.agent.usage import UsageTracker
from emiya_agent.config import Config, select_environment
from emiya_agent.errors import ProviderError
from emiya_agent.hooks import HookDecision, HookEvent, HookRunner
from emiya_agent.lsp.manager import LSPManager
from emiya_agent.providers.base import MessageDone
from emiya_agent.providers.registry import ProviderRegistry
from emiya_agent.tools.base import ToolContext, UIBridge
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.types import user_message
from emiya_agent.ui.render import QuietRenderer, StreamRenderer

_COMMON = """\
You are a subagent of EmiyaAgent working on a delegated task. \
Work autonomously — you cannot ask the user questions. \
End with a final report that fully answers the delegation prompt; \
the parent agent only sees your final message, not your tool calls. \
Write the report in the SAME language as the delegation prompt \
(中文派活则中文报告) — the parent agent may have a different native language.
"""

EXPLORE_SYSTEM = (
    _COMMON
    + """
Role: EXPLORER. Search and read the codebase to answer the question. \
You have read-only tools. Be thorough but do not speculate — report file paths \
and line references for everything you claim.
"""
)

PLAN_SYSTEM = (
    _COMMON
    + """
Role: PLANNER. Design an implementation approach for the given task. \
You have read-only tools — inspect the code first, then produce a concrete, \
step-by-step plan naming files and functions. Consider trade-offs briefly and \
recommend one approach.
"""
)

GENERAL_SYSTEM = (
    _COMMON
    + """
Role: GENERAL. Complete the delegated task end-to-end with the available tools, \
then report what you did and verified.
"""
)


@dataclass(frozen=True, slots=True)
class SubagentRole:
    name: str
    route_key: str  # config routes 键
    system_prompt: str
    max_iterations: int
    read_only: bool = False
    tools_enabled: bool = True
    include_shared_context: bool = True
    effort: str | None = None
    kind: str = "delegated"


DELEGATED_ROLES: dict[str, SubagentRole] = {
    "explore": SubagentRole(
        "explore",
        "subagent_explore",
        EXPLORE_SYSTEM,
        15,
        read_only=True,
        effort="low",
    ),
    "plan": SubagentRole("plan", "subagent_plan", PLAN_SYSTEM, 25, read_only=True),
    "general": SubagentRole("general", "subagent_general", GENERAL_SYSTEM, 40),
}

INTERNAL_ROLES: dict[str, SubagentRole] = {
    "translator": SubagentRole(
        "translator",
        "translator",
        TRANSLATION_SYSTEM,
        1,
        tools_enabled=False,
        include_shared_context=False,
        effort="low",
        kind="internal",
    ),
    "compactor": SubagentRole(
        "compactor",
        "compactor",
        COMPACT_SYSTEM,
        1,
        tools_enabled=False,
        include_shared_context=False,
        effort="low",
        kind="internal",
    ),
}

# 子 agent 永不可用的工具（交互/递归/进度污染）
EXCLUDED_TOOLS = frozenset(
    {
        "spawn_subagent",
        "subagent_task",
        "translate_for_model",
        "ask_user",
        "todo_write",
    }
)


class SubagentExecutionError(ProviderError):
    """An internal specialist subagent did not produce a complete result."""

    def __init__(
        self,
        role: str,
        stop_reason: str,
        detail: str | None = None,
    ) -> None:
        self.role = role
        self.stop_reason = stop_reason
        self.detail = detail
        message = f"{role} subagent stopped with {stop_reason}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class SubagentSpawner:
    def __init__(
        self,
        providers: ProviderRegistry,
        config: Config,
        tools: ToolRegistry,
        renderer: StreamRenderer,
        usage: UsageTracker,
        shared_context: str = "",
        controller: RunController | None = None,
        hooks: HookRunner | None = None,
    ) -> None:
        self.providers = providers
        self.config = config
        self.tools = tools
        self.renderer = renderer
        self.usage = usage
        self.shared_context = shared_context.strip()
        self.controller = controller
        self.hooks = hooks

    async def spawn(
        self,
        role_name: str,
        prompt: str,
        parent_ctx: ToolContext,
        *,
        cancel: asyncio.Event | None = None,
        quiet: bool = False,
        usage_scope: str | None = None,
        config: Config | None = None,
        tools: ToolRegistry | None = None,
        shared_context: str | None = None,
        ui: UIBridge | None = None,
    ) -> str:
        effective_config = config or self.config
        effective_tools = tools or self.tools
        role = DELEGATED_ROLES.get(role_name)
        if role is None:
            return (
                "Error: unknown delegated subagent role "
                f"{role_name!r}，可用: {', '.join(DELEGATED_ROLES)}"
            )
        effective_usage_scope = usage_scope or f"subagent:{role.name}"

        result = await self._run(
            role,
            prompt,
            parent_ctx,
            cancel=cancel,
            quiet=quiet,
            usage_scope=effective_usage_scope,
            config=effective_config,
            tools=effective_tools,
            shared_context=shared_context,
            ui=ui,
        )

        if result.stop_reason == "refusal":
            return f"Subagent refused: {result.refusal or ''}"
        return result.final_text or "(subagent produced no report)"

    async def spawn_internal(
        self,
        role_name: str,
        prompt: str,
        parent_ctx: ToolContext,
        *,
        cancel: asyncio.Event | None = None,
        usage_scope: str | None = None,
        config: Config | None = None,
    ) -> str:
        """Run a host-selected specialist that is never model-delegatable.

        Internal specialists always receive an empty tool registry, an isolated
        message list, no project/skill context, and a quiet renderer.  A
        non-terminal or empty response is an error rather than usable output.
        """

        role = INTERNAL_ROLES.get(role_name)
        if role is None:
            raise ValueError(f"unknown internal subagent role: {role_name!r}")
        effective_config = config or self.config
        effective_usage_scope = usage_scope or f"internal:{role.name}"
        result = await self._run(
            role,
            prompt,
            parent_ctx,
            cancel=cancel,
            quiet=True,
            usage_scope=effective_usage_scope,
            config=effective_config,
            tools=ToolRegistry(),
            shared_context="",
            ui=None,
        )
        if result.stop_reason != "end_turn":
            detail = result.refusal if result.stop_reason == "refusal" else None
            raise SubagentExecutionError(role.name, result.stop_reason, detail)
        if not result.final_text.strip():
            raise SubagentExecutionError(role.name, "empty_response")
        return result.final_text

    async def _run(
        self,
        role: SubagentRole,
        prompt: str,
        parent_ctx: ToolContext,
        *,
        cancel: asyncio.Event | None,
        quiet: bool,
        usage_scope: str,
        config: Config,
        tools: ToolRegistry,
        shared_context: str | None,
        ui: UIBridge | None,
    ) -> LoopResult:
        effective_config = config
        effective_tools = tools

        route = effective_config.route(role.route_key)
        provider, model = self.providers.get(route)
        sub_tools = (
            effective_tools.subset(
                read_only_only=role.read_only, exclude=EXCLUDED_TOOLS
            )
            if role.tools_enabled
            else ToolRegistry()
        )
        console = getattr(self.renderer, "console", None)
        if not quiet and console is not None:
            console.print(f"[blue]┌─ 子agent [{role.name}] @ {route} ─────────[/blue]")

        renderer = QuietRenderer() if quiet else self.renderer
        hook_env = select_environment(effective_config.hook_inherit_env)
        effective_hooks = (
            self.hooks.for_context(cwd=effective_config.cwd, env=hook_env)
            if self.hooks is not None
            else HookRunner(
                effective_config.hooks,
                cwd=effective_config.cwd,
                env=hook_env,
            )
        )

        async def render(events) -> MessageDone:
            return await renderer.render(events)

        loop = AgentLoop(
            provider,
            model,
            route,
            sub_tools,
            render,
            max_tokens=effective_config.max_tokens,
            effort=role.effort or effective_config.effort,
            max_iterations=role.max_iterations,
            on_usage=lambda used_route, usage: self.usage.add(
                used_route, usage, scope=usage_scope
            ),
            controller=self.controller,
            hooks=effective_hooks,
        )
        # 隔离上下文：新消息列表 + 新 state；共享 cancel 以便一起中断
        lsp = (
            LSPManager(effective_config, effective_config.cwd)
            if role.tools_enabled
            else None
        )
        sub_ctx = ToolContext(
            cwd=effective_config.cwd,
            config=effective_config,
            ui=ui or parent_ctx.ui,
            cancel=cancel or parent_ctx.cancel,
            state={"lsp_manager": lsp} if lsp is not None else {},
        )
        messages = [user_message(prompt)]
        system = role.system_prompt.rstrip()
        if role.include_shared_context:
            context = self.shared_context if shared_context is None else shared_context
            if context:
                system += f"\n\n{context}\n"
        stop_reason = "execution_error"
        try:
            try:
                result = await loop.run(messages, system=system, ctx=sub_ctx)
                stop_reason = result.stop_reason
                if result.stop_reason == "interrupted" and sub_ctx.cancel.is_set():
                    stop_reason = "cancelled"
                    raise asyncio.CancelledError
            except asyncio.CancelledError:
                stop_reason = "cancelled"
                raise
            except KeyboardInterrupt:
                stop_reason = "interrupted"
                raise
        finally:
            try:
                if lsp is not None:
                    await lsp.close_all()
            finally:
                # SubagentStop is observational and must be paired with every
                # started subagent, including provider failures and cancellation.
                if effective_hooks.has_hooks_for(HookEvent.SUBAGENT_STOP):
                    try:
                        outcome = await effective_hooks.run(
                            HookEvent.SUBAGENT_STOP,
                            role.name,
                            {
                                "role": role.name,
                                "kind": role.kind,
                                "route": route,
                                "stop_reason": stop_reason,
                            },
                            initial_decision=HookDecision.ALLOW,
                        )
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        raise
                    except Exception:
                        outcome = None
                    if outcome is not None and not quiet and console is not None:
                        for diagnostic in outcome.diagnostics:
                            if diagnostic.status != "ok":
                                console.print(
                                    f"Hook {diagnostic.hook_id}"
                                    f"（{diagnostic.status}）: {diagnostic.reason}",
                                    markup=False,
                                    style="yellow",
                                )

        if not quiet and console is not None:
            console.print(
                f"[blue]└─ 子agent [{role.name}] 结束（{result.stop_reason}）[/blue]"
            )

        return result
