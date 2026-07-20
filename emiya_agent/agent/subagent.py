"""子 agent：隔离上下文的角色化执行单元。

- explore: 只读搜索/摸底，走便宜模型（DeepSeek）
- plan:    方案设计，走强模型（opus），只读
- general: 完整工具，通用执行

深度限制为 1：子 agent 不能再生成子 agent（工具子集里排除 spawn_subagent）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from emiya_agent.agent.control import RunController
from emiya_agent.agent.loop import AgentLoop
from emiya_agent.agent.usage import UsageTracker
from emiya_agent.config import Config, select_environment
from emiya_agent.hooks import HookRunner
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


@dataclass
class SubagentRole:
    name: str
    route_key: str  # config routes 键
    read_only: bool
    system_prompt: str
    max_iterations: int


ROLES: dict[str, SubagentRole] = {
    "explore": SubagentRole("explore", "subagent_explore", True, EXPLORE_SYSTEM, 15),
    "plan": SubagentRole("plan", "subagent_plan", True, PLAN_SYSTEM, 25),
    "general": SubagentRole("general", "subagent_general", False, GENERAL_SYSTEM, 40),
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
        usage_scope: str = "main",
        config: Config | None = None,
        tools: ToolRegistry | None = None,
        shared_context: str | None = None,
        ui: UIBridge | None = None,
    ) -> str:
        effective_config = config or self.config
        effective_tools = tools or self.tools
        role = ROLES.get(role_name)
        if role is None:
            return (
                f"Error: unknown subagent role {role_name!r}，可用: {', '.join(ROLES)}"
            )

        route = effective_config.route(role.route_key)
        provider, model = self.providers.get(route)
        sub_tools = effective_tools.subset(
            read_only_only=role.read_only, exclude=EXCLUDED_TOOLS
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
            effort="low" if role.name == "explore" else effective_config.effort,
            max_iterations=role.max_iterations,
            on_usage=lambda used_route, usage: self.usage.add(
                used_route, usage, scope=usage_scope
            ),
            controller=self.controller,
            hooks=effective_hooks,
        )
        # 隔离上下文：新消息列表 + 新 state；共享 cancel 以便一起中断
        sub_ctx = ToolContext(
            cwd=effective_config.cwd,
            config=effective_config,
            ui=ui or parent_ctx.ui,
            cancel=cancel or parent_ctx.cancel,
            state={"lsp_manager": LSPManager(effective_config, effective_config.cwd)},
        )
        messages = [user_message(prompt)]
        system = role.system_prompt.rstrip()
        context = self.shared_context if shared_context is None else shared_context
        if context:
            system += f"\n\n{context}\n"
        try:
            result = await loop.run(messages, system=system, ctx=sub_ctx)
        finally:
            await sub_ctx.state["lsp_manager"].close_all()
        if not quiet and console is not None:
            console.print(
                f"[blue]└─ 子agent [{role.name}] 结束（{result.stop_reason}）[/blue]"
            )

        if result.stop_reason == "interrupted" and sub_ctx.cancel.is_set():
            raise asyncio.CancelledError

        if result.stop_reason == "refusal":
            return f"Subagent refused: {result.refusal or ''}"
        return result.final_text or "(subagent produced no report)"
