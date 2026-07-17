"""子 agent：隔离上下文的角色化执行单元。

- explore: 只读搜索/摸底，走便宜模型（DeepSeek）
- plan:    方案设计，走强模型（opus），只读
- general: 完整工具，通用执行

深度限制为 1：子 agent 不能再生成子 agent（工具子集里排除 spawn_subagent）。
"""

from __future__ import annotations

from dataclasses import dataclass

from emiya_agent.agent.loop import AgentLoop
from emiya_agent.agent.usage import UsageTracker
from emiya_agent.config import Config
from emiya_agent.providers.base import MessageDone
from emiya_agent.providers.registry import ProviderRegistry
from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.types import user_message
from emiya_agent.ui.render import StreamRenderer

_COMMON = """\
You are a subagent of EmiyaAgent working on a delegated task. \
Work autonomously — you cannot ask the user questions. \
End with a final report that fully answers the delegation prompt; \
the parent agent only sees your final message, not your tool calls. \
Write the report in the SAME language as the delegation prompt \
(中文派活则中文报告) — the parent agent may have a different native language.
"""

EXPLORE_SYSTEM = _COMMON + """
Role: EXPLORER. Search and read the codebase to answer the question. \
You have read-only tools. Be thorough but do not speculate — report file paths \
and line references for everything you claim.
"""

PLAN_SYSTEM = _COMMON + """
Role: PLANNER. Design an implementation approach for the given task. \
You have read-only tools — inspect the code first, then produce a concrete, \
step-by-step plan naming files and functions. Consider trade-offs briefly and \
recommend one approach.
"""

GENERAL_SYSTEM = _COMMON + """
Role: GENERAL. Complete the delegated task end-to-end with the available tools, \
then report what you did and verified.
"""


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
EXCLUDED_TOOLS = frozenset({"spawn_subagent", "ask_user", "todo_write"})


class SubagentSpawner:
    def __init__(
        self,
        providers: ProviderRegistry,
        config: Config,
        tools: ToolRegistry,
        renderer: StreamRenderer,
        usage: UsageTracker,
    ) -> None:
        self.providers = providers
        self.config = config
        self.tools = tools
        self.renderer = renderer
        self.usage = usage

    async def spawn(self, role_name: str, prompt: str, parent_ctx: ToolContext) -> str:
        role = ROLES.get(role_name)
        if role is None:
            return f"Error: unknown subagent role {role_name!r}，可用: {', '.join(ROLES)}"

        route = self.config.route(role.route_key)
        provider, model = self.providers.get(route)
        sub_tools = self.tools.subset(
            read_only_only=role.read_only, exclude=EXCLUDED_TOOLS
        )
        console = self.renderer.console
        console.print(
            f"[blue]┌─ 子agent [{role.name}] @ {route} ─────────[/blue]"
        )

        async def render(events) -> MessageDone:
            return await self.renderer.render(events)

        loop = AgentLoop(
            provider,
            model,
            route,
            sub_tools,
            render,
            max_tokens=self.config.max_tokens,
            effort="low" if role.name == "explore" else self.config.effort,
            max_iterations=role.max_iterations,
            on_usage=self.usage.add,
        )
        # 隔离上下文：新消息列表 + 新 state；共享 cancel 以便一起中断
        sub_ctx = ToolContext(
            cwd=parent_ctx.cwd,
            config=parent_ctx.config,
            ui=parent_ctx.ui,
            cancel=parent_ctx.cancel,
        )
        messages = [user_message(prompt)]
        result = await loop.run(messages, system=role.system_prompt, ctx=sub_ctx)
        console.print(f"[blue]└─ 子agent [{role.name}] 结束（{result.stop_reason}）[/blue]")

        if result.stop_reason == "refusal":
            return f"Subagent refused: {result.refusal or ''}"
        return result.final_text or "(subagent produced no report)"
