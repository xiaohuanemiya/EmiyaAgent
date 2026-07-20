"""交互式 REPL：prompt_toolkit 收输入，rich 渲染输出。

终端契约：prompt 激活期间不做任何 rich 渲染；一轮流式输出结束后才重新出提示符。
Repl 同时实现 UIBridge（工具的 ask/notify 走这里）。
"""

from __future__ import annotations

import asyncio
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import DummyHistory, FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from emiya_agent import __version__
from emiya_agent.agent.control import RunController
from emiya_agent.agent.completion import complete_text
from emiya_agent.agent.loop import AgentLoop, LoopResult
from emiya_agent.agent.prompts import build_system_prompt, load_agent_md
from emiya_agent.agent.subagent import SubagentSpawner
from emiya_agent.agent.subagent_tasks import SubagentTaskService
from emiya_agent.agent.usage import UsageTracker
from emiya_agent.compaction.manager import CompactionManager
from emiya_agent.config import Config, config_files_exist, select_environment
from emiya_agent.errors import ConfigError, ProviderError
from emiya_agent.hooks import HookDecision, HookEvent, HookRunner
from emiya_agent.hooks.models import HookRunResult
from emiya_agent.lsp.manager import LSPManager
from emiya_agent.mcp_client.adapter import build_mcp_tools
from emiya_agent.mcp_client.manager import MCPManager
from emiya_agent.memory.manager import MemoryManager
from emiya_agent.providers.registry import ProviderRegistry
from emiya_agent.safe_paths import UnsafePathError
from emiya_agent.session.manager import SessionManager
from emiya_agent.session.store import SessionRecordTooLargeError, SessionStore
from emiya_agent.skills.loader import (
    default_skill_dirs,
    discover_skills,
    skills_index_text,
)
from emiya_agent.tools import build_default_registry
from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.memory_tool import MemoryTool
from emiya_agent.tools.skill_tool import SkillTool
from emiya_agent.tools.task import SpawnSubagentTool, SubagentTaskTool
from emiya_agent.tools.translate import TranslateForModelTool
from emiya_agent.types import (
    Message,
    ToolResultBlock,
    ToolUseBlock,
    user_message,
)
from emiya_agent.ui import commands, panels
from emiya_agent.ui.json_output import JsonEventSink, JsonStreamRenderer
from emiya_agent.ui.render import QuietRenderer, StreamRenderer


class Repl:
    def __init__(
        self,
        config: Config,
        *,
        automation: bool = False,
        output_mode: str = "human",
        event_sink: JsonEventSink | None = None,
        controller: RunController | None = None,
    ) -> None:
        if output_mode not in {"human", "json", "stream-json"}:
            raise ValueError(f"unknown output mode: {output_mode}")
        if output_mode == "stream-json" and event_sink is None:
            raise ValueError("stream-json output requires an event sink")
        self.config = config
        self.automation = automation
        self.output_mode = output_mode
        self.event_sink = event_sink
        self.controller = controller
        self.console = Console(file=sys.stderr) if output_mode != "human" else Console()
        self.providers = ProviderRegistry(config)
        self.usage = UsageTracker(config)
        if output_mode == "stream-json":
            assert event_sink is not None
            self.renderer = JsonStreamRenderer(event_sink)
        elif output_mode == "json":
            self.renderer = QuietRenderer()
        else:
            self.renderer = StreamRenderer(self.console)
        self.route = config.route("executor")
        self.messages: list[Message] = []
        self.last_result: LoopResult | None = None
        self.last_error: str | None = None
        self.last_error_kind: str | None = None

        # skills / memory / 子agent / 压缩
        self.skills = discover_skills(default_skill_dirs(config))
        project_dir = config.project_dir()
        self.memory = MemoryManager(config.project_runtime_dir("memory"))
        project_instructions = load_agent_md(config.cwd)
        skill_index = skills_index_text(self.skills)
        memory_index = self.memory.index_text()
        self.system = build_system_prompt(
            config.cwd,
            project_instructions,
            skills_index=skill_index,
            memory_index=memory_index,
        )
        shared_sections = [f"# Environment\n- Working directory: {config.cwd}"]
        if project_instructions:
            shared_sections.append(f"# Project instructions\n{project_instructions}")
        if skill_index:
            shared_sections.append(f"# Skills\n{skill_index}")

        self.tools = build_default_registry()
        self.tools.register(SkillTool(self.skills))
        self.tools.register(MemoryTool(self.memory))
        self.tools.register(
            TranslateForModelTool(
                self.providers,
                config,
                on_usage=self.usage.add,
                controller=controller,
            )
        )
        self.hooks = HookRunner(
            config.hooks,
            cwd=config.cwd,
            env=select_environment(config.hook_inherit_env),
        )
        self.spawner = SubagentSpawner(
            self.providers,
            config,
            self.tools,
            self.renderer,
            self.usage,
            shared_context="\n\n".join(shared_sections),
            controller=controller,
            hooks=self.hooks,
        )
        self.task_service = SubagentTaskService(self.spawner, config)
        self.tools.register(SpawnSubagentTool(self.spawner, self.task_service))
        self.tools.register(SubagentTaskTool(self.task_service))
        self.compactor = CompactionManager(self._compactor_complete)
        self.mcp = MCPManager(config)
        self.lsp = LSPManager(config, config.cwd)

        self.tool_ctx = ToolContext(
            cwd=config.cwd,
            config=config,
            ui=self,
            state={"lsp_manager": self.lsp},
        )
        config.project_runtime_dir("sessions")
        self.session_mgr = SessionManager(project_dir)
        self.store: SessionStore | None = None  # 首轮输入时惰性创建
        self.session_id: str | None = None
        self.first_run = not config_files_exist(config.cwd)
        # 非 TTY（管道/CI）降级为逐行读 stdin；真实终端用 prompt_toolkit
        self.session: PromptSession[str] | None = None
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                history_file = Config.user_dir() / "repl_history.txt"
                self.session = PromptSession(history=FileHistory(str(history_file)))
            except Exception:
                self.session = None

    # ------------------------------------------------------------- 输入

    async def read_line(self, prompt: str = "emiya> ") -> str:
        """读取一行输入。EOF 时抛 EOFError。"""
        if self.session is not None:
            with patch_stdout():
                return await self.session.prompt_async(prompt)
        self.console.print(prompt, end="", markup=False, highlight=False)
        line = await asyncio.to_thread(sys.stdin.readline)
        if line == "":
            raise EOFError
        return line.rstrip("\n").lstrip("\ufeff")

    async def read_secret(self, prompt: str) -> str:
        """读取不回显、也绝不写入持久历史的敏感值。"""
        if self.session is None:
            # 管道/CI 没有可靠的隐藏输入能力；要求改用环境变量。
            raise EOFError("secret input requires an interactive terminal")
        secret_session: PromptSession[str] = PromptSession(history=DummyHistory())
        with patch_stdout():
            return await secret_session.prompt_async(prompt, is_password=True)

    # ------------------------------------------------------------- UIBridge

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        """ask_user 工具的实现：展示问题并等待输入。"""
        if self.automation:
            raise RuntimeError("ask_user is unavailable in one-shot mode")
        self.console.print(
            Panel(
                Text(panels.safe_terminal_text(question)),
                title="Agent 提问",
                border_style="magenta",
            )
        )
        if options:
            for i, opt in enumerate(options, 1):
                self.console.print(f"  {i}. {opt}", markup=False)
            answer = (await self.read_line("选择编号或直接作答 > ")).strip()
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1]
            return answer
        return (await self.read_line("答复 > ")).strip()

    def notify(self, text: str) -> None:
        """todo_write 等工具的进度展示。"""
        panels.print_todos(self.console, text)

    async def confirm_tool(self, name: str, args_brief: str) -> bool:
        """写操作审批（tool_approval = "ask" 时）。a = 本会话改为全部放行。"""
        if self.automation:
            if self.event_sink is not None:
                self.event_sink.emit(
                    "approval_required",
                    tool=name,
                    preview=args_brief,
                    approved=False,
                )
            return False
        self.console.print(
            Panel(
                f"[bold]{escape(name)}[/bold]\n{escape(args_brief)}",
                title="执行确认",
                border_style="yellow",
            )
        )
        try:
            answer = (
                (await self.read_line("[y]允许(回车)  [n]拒绝  [a]本会话全部允许 > "))
                .strip()
                .lower()
            )
        except (KeyboardInterrupt, EOFError):
            return False
        if answer in ("a", "always", "全部"):
            self.config.tool_approval = "auto"
            self.console.print(
                "[dim]本会话切换为自动放行（/approval ask 可改回）。[/dim]"
            )
            return True
        return answer in ("", "y", "yes", "是")

    # ------------------------------------------------------------- 生命周期 Hook

    async def emit_hook_event(
        self,
        event: HookEvent,
        subject: str,
        payload: dict | None = None,
    ) -> HookRunResult | None:
        """运行一个生命周期事件的用户级 Hook。

        基础设施错误绝不打断主流程；返回 None 表示无匹配 Hook 或运行失败。
        决策是否生效由调用方按 DECISION_EVENTS 约定处理，观测型事件只展示诊断。
        """
        if self.hooks is None or not self.hooks.has_hooks_for(event):
            return None
        try:
            result = await self.hooks.run(
                event, subject, payload, initial_decision=HookDecision.ALLOW
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            self.console.print(
                f"[yellow]Hook 基础设施错误（{event.value}）: "
                f"{escape(f'{type(exc).__name__}: {exc}')}[/yellow]"
            )
            return None
        for diagnostic in result.diagnostics:
            if diagnostic.status != "ok":
                self.console.print(
                    f"[yellow]Hook {escape(diagnostic.hook_id)}"
                    f"（{diagnostic.status}）: {escape(diagnostic.reason)}[/yellow]"
                )
        return result

    # ------------------------------------------------------------- 主循环

    async def run(self) -> None:
        await self._maybe_first_run_init()
        self._banner()
        await self._connect_mcp()
        await self.emit_hook_event(
            HookEvent.SESSION_START,
            "startup",
            {"source": "startup", "cwd": str(self.config.cwd)},
        )
        try:
            await self._main_loop()
        finally:
            try:
                await self.emit_hook_event(
                    HookEvent.SESSION_END,
                    "exit",
                    {"reason": "exit", "session_id": self.session_id},
                )
            finally:
                try:
                    await self.task_service.shutdown()
                finally:
                    try:
                        await self.lsp.close_all()
                    finally:
                        try:
                            await self.mcp.stop_all()  # 确保 stdio 子进程被终止
                        finally:
                            self._release_store_writer()
        self.console.print("[dim]再见。[/dim]")

    async def one_shot(self, text: str) -> bool:
        """emiya -p "...": 执行一轮后退出；默认沿用安全审批策略。"""
        self.last_result = None
        self.last_error = None
        self.last_error_kind = None
        try:
            await self._connect_mcp()
            await self.emit_hook_event(
                HookEvent.SESSION_START,
                "one-shot",
                {"source": "one-shot", "cwd": str(self.config.cwd)},
            )
            return await self._execute_turn(text) is not None
        finally:
            try:
                await self.emit_hook_event(
                    HookEvent.SESSION_END,
                    "exit",
                    {"reason": "exit", "session_id": self.session_id},
                )
            finally:
                try:
                    await self.task_service.shutdown()
                finally:
                    try:
                        await self.lsp.close_all()
                    finally:
                        try:
                            await self.mcp.stop_all()
                        finally:
                            self._release_store_writer()

    async def _maybe_first_run_init(self) -> None:
        """首次运行（无任何配置文件）且交互式终端时，询问是否初始化。"""
        if not self.first_run or self.session is None:
            return
        from emiya_agent.ui.init_wizard import run_init_wizard

        try:
            answer = (
                (
                    await self.read_line(
                        "检测到首次运行。现在初始化模型供应商吗？[Y/n] > "
                    )
                )
                .strip()
                .lower()
            )
        except (KeyboardInterrupt, EOFError):
            return
        if answer in ("", "y", "yes", "是"):
            await run_init_wizard(self, first_run=True)
        else:
            from emiya_agent.config import user_config_path, write_config_template

            write_config_template(user_config_path())
            self.console.print(
                "[dim]已跳过。生成了配置模板，可手工编辑或随时运行 /init。[/dim]"
            )
        self.first_run = False

    async def _connect_mcp(self) -> None:
        """启动时连接 MCP 服务器并注册工具（首个请求前完成，保证缓存前缀稳定）。"""
        if not self.config.mcp_servers:
            return
        with self.console.status("[dim]连接 MCP 服务器…[/dim]"):
            await self.mcp.start_all()
        for tool in build_mcp_tools(self.mcp):
            self.tools.register(tool)
        for name, info in self.mcp.status().items():
            if info["connected"]:
                self.console.print(
                    f"[green]✓ MCP {name}[/green]（{info['tools']} 个工具）"
                )
            else:
                self.console.print(
                    f"[red]✗ MCP {name}: {info['error'] or '连接失败'}[/red]"
                )

    async def _main_loop(self) -> None:
        while True:
            try:
                line = await self.read_line()
            except KeyboardInterrupt:
                continue  # 空提示符上 Ctrl+C：忽略
            except EOFError:
                break  # Ctrl+D / Ctrl+Z：退出
            line = line.lstrip("\ufeff").strip()  # 管道输入可能带 BOM
            if not line:
                continue
            if line.startswith("/"):
                if await commands.dispatch(self, line):
                    break
                continue
            await self.handle_input(line)

    def _banner(self) -> None:
        self.console.print(
            f"[bold magenta]EmiyaAgent[/bold magenta] v{__version__}  ·  "
            f"执行模型 [bold]{self.route}[/bold]  ·  Agent 自主规划  ·  /help 查看命令"
        )
        for warning in self.config.load_warnings:
            self.console.print(f"[yellow]⚠ {warning}[/yellow]")
        missing = [
            f"{name}({self.config.providers[name].api_key_env})"
            for name, ok in self.config.key_status().items()
            if not ok
        ]
        if missing:
            self.console.print(
                f"[yellow]⚠ 未检测到 API key: {', '.join(missing)} —— "
                f"对应模型不可用，详见 /config[/yellow]"
            )

    # ------------------------------------------------------------- 输入分发

    async def handle_input(self, text: str) -> None:
        """普通输入原样交给主 Agent；是否规划或派生子 Agent 由模型决定。"""
        await self._execute_turn(text)

    # ------------------------------------------------------------- 压缩

    async def _compactor_complete(self, system: str, text: str) -> str:
        route = self.config.route("compactor")
        provider, model = self.providers.get(route)
        result, usage = await complete_text(
            provider,
            model,
            system,
            text,
            max_tokens=self.config.max_tokens,
            route=route,
            controller=self.controller,
        )
        self.usage.add(route, usage)
        return result

    async def maybe_compact(self, force: bool = False) -> None:
        """超阈值自动压缩；force=True 为 /compact 手动触发。"""
        window = self.config.model_info(self.route).context_window
        if not force and not self.compactor.should_compact(
            self.usage.last_for(self.route), window
        ):
            return
        head, _ = self.compactor.split(self.messages)
        if not head:
            if force:
                self.console.print("[dim]对话还很短，无需压缩。[/dim]")
            return
        trigger = "manual" if force else "auto"
        # PreCompact 是决策型事件：deny 跳过本轮压缩（/compact 可再手动触发）。
        pre_compact = await self.emit_hook_event(
            HookEvent.PRE_COMPACT,
            trigger,
            {"trigger": trigger, "messages": len(self.messages)},
        )
        if pre_compact is not None and pre_compact.decision == HookDecision.DENY:
            self.console.print(
                "[yellow]PreCompact hook 阻止了本轮压缩，继续使用完整历史"
                f"{'：' + escape(pre_compact.reason) if pre_compact.reason else '。'}"
                "[/yellow]"
            )
            return
        try:
            before = len(self.messages)
            with self.console.status("[dim]正在压缩上下文…[/dim]"):
                compacted = await self.compactor.compact(self.messages)
            store = self._ensure_store()
            # 原始消息继续保留在 JSONL 供审计；恢复时从此活跃上下文快照回放。
            store.append_snapshot(compacted)
            # 快照完整落盘后再改变内存，避免磁盘失败造成两边上下文分叉。
            # 保持列表对象身份，确保正在运行的 AgentLoop 立即看到轮内结果。
            self.messages[:] = compacted
            try:
                store.append_meta(
                    "compaction", {"before": before, "after": len(self.messages)}
                )
            except (OSError, UnsafePathError, ValueError):
                # 审计元数据不是恢复基线；快照已成功时不回滚压缩。
                self.console.print(
                    "[yellow]压缩已完成，但审计元数据写入失败。[/yellow]"
                )
            self.console.print(
                f"[dim]✂ 上下文已压缩：{before} → {len(self.messages)} 条消息"
                "（原始记录保留在会话文件中）。[/dim]"
            )
            await self.emit_hook_event(
                HookEvent.POST_COMPACT,
                trigger,
                {
                    "trigger": trigger,
                    "before": before,
                    "after": len(self.messages),
                },
            )
        except (
            ProviderError,
            ConfigError,
            SessionRecordTooLargeError,
            UnsafePathError,
            OSError,
            ValueError,
        ) as exc:
            self.console.print(
                f"[yellow]压缩失败（{exc}），继续使用完整历史。[/yellow]"
            )

    # ------------------------------------------------------------- 会话

    def _ensure_store(self) -> SessionStore:
        if self.store is None:
            self.session_id, self.store = self.session_mgr.new_store(
                acquire_writer=True
            )
        return self.store

    def _release_store_writer(self) -> None:
        if self.store is not None:
            self.store.release_writer()

    def _persist(self, message: Message) -> None:
        self._ensure_store().append_message(message)

    # ------------------------------------------------------------- 执行一轮

    async def _execute_turn(self, text: str) -> str | None:
        """把 text 作为用户消息，运行 agent 工具循环。返回最终文本；失败返回 None。"""
        try:
            provider, model = self.providers.get(self.route)
        except ConfigError as exc:
            self.last_error = str(exc)
            self.last_error_kind = "config"
            self.console.print(f"[red]{exc}[/red]")
            return None

        # UserPromptSubmit 是决策型事件：deny 阻止该提示词进入历史与模型。
        prompt_hook = await self.emit_hook_event(
            HookEvent.USER_PROMPT_SUBMIT,
            text,
            {"prompt": text, "session_id": self.session_id},
        )
        if prompt_hook is not None and prompt_hook.decision == HookDecision.DENY:
            reason = prompt_hook.reason or "UserPromptSubmit hook 拒绝了该提示词"
            self.last_error = f"UserPromptSubmit hook blocked the prompt: {reason}"
            self.last_error_kind = "execution"
            self.console.print(
                f"[yellow]提示词已被 Hook 拦截，未发送: {escape(reason)}[/yellow]"
            )
            return None

        msg = user_message(text)
        self.messages.append(msg)
        self._persist(msg)
        self.tool_ctx.cancel = asyncio.Event()  # 每轮重置

        def on_tool_result(call: ToolUseBlock, result: ToolResultBlock) -> None:
            if self.event_sink is not None:
                self.event_sink.emit(
                    "tool_result",
                    tool_use_id=call.id,
                    tool_name=call.name,
                    input=call.input,
                    content=result.content,
                    is_error=result.is_error,
                )
            panels.print_tool_call(self.console, call)
            panels.print_tool_result(self.console, call, result)

        loop = AgentLoop(
            provider,
            model,
            self.route,
            self.tools,
            self.renderer.render,
            max_tokens=self.config.max_tokens,
            effort=self.config.effort,
            on_message=self._persist,
            on_usage=self.usage.add,
            on_tool_result=on_tool_result,
            before_request=lambda _messages: self.maybe_compact(),
            controller=self.controller,
            hooks=self.hooks,
        )
        try:
            result = await loop.run(
                self.messages, system=self.system, ctx=self.tool_ctx
            )
        except ProviderError as exc:
            self._repair_dangling()
            self.last_error = str(exc)
            self.last_error_kind = "provider"
            self.console.print(f"[red]{exc}[/red]")
            return None
        except (KeyboardInterrupt, EOFError):
            self.tool_ctx.cancel.set()
            self._repair_dangling()
            self.last_result = LoopResult(final_text="", stop_reason="interrupted")
            self.console.print("\n[yellow]已中断本轮。[/yellow]")
            return None
        except Exception as exc:
            # AgentLoop commits protocol messages to memory before invoking
            # the persistence callback. If that callback (or a later
            # integration) fails, repair the JSONL suffix without hiding the
            # original failure.
            try:
                self._repair_dangling("Tool loop aborted after an internal error")
            except Exception as repair_exc:
                exc.add_note(
                    "Additionally failed to repair the session's dangling "
                    f"tool result: {type(repair_exc).__name__}: {repair_exc}"
                )
            raise

        self.last_result = result
        await self.emit_hook_event(
            HookEvent.STOP,
            result.stop_reason,
            {
                "stop_reason": result.stop_reason,
                "route": self.route,
                "session_id": self.session_id,
            },
        )
        text_result = self._report_result(result)
        if not self.automation:
            await self.maybe_compact()
        return text_result

    def _report_result(self, result: LoopResult) -> str | None:
        if result.stop_reason == "refusal":
            self.console.print(f"[red]模型拒绝了该请求: {result.refusal or ''}[/red]")
            return None
        if result.stop_reason == "error":
            self.console.print(
                "[red]模型返回了非法或无法安全回放的工具调用，本轮已终止。[/red]"
            )
            return None
        if result.stop_reason == "interrupted":
            self.console.print("[yellow]已中断本轮。[/yellow]")
            return result.final_text or None

        footer = self.usage.footer()
        if self.session_id:
            footer += f" · 会话 {self.session_id}"
        if result.served_by:
            footer += f" · 由 {result.served_by} 服务"
        if result.stop_reason == "max_tokens":
            footer += " · [yellow]输出触顶被截断[/yellow]"
        if result.stop_reason == "max_iterations":
            footer += " · [yellow]达到迭代上限[/yellow]"
        if result.stop_reason == "max_turns":
            footer += " · [yellow]达到全局请求轮次上限[/yellow]"
        if result.stop_reason == "budget_exceeded":
            footer += " · [yellow]达到费用预算上限[/yellow]"
        if result.stop_reason == "budget_unavailable":
            footer += " · [yellow]缺少模型定价，无法执行预算约束[/yellow]"
        if result.stop_reason == "timeout":
            footer += " · [yellow]达到运行超时[/yellow]"
        self.console.print(f"[dim]{footer}[/dim]")
        return result.final_text

    def _repair_dangling(self, note: str = "Interrupted by user") -> None:
        """防御：若最后一条助手消息带未回应的 tool_use，补合成 tool_result。"""
        if not self.messages:
            return

        last = self.messages[-1]
        if last.role == "assistant":
            assistant = last
            calls = assistant.tool_uses()
            if not calls:
                return
            result_message = Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id=call.id, content=note, is_error=True)
                    for call in calls
                ],
            )
            self.messages.append(result_message)
            assistant_index = len(self.messages) - 2
        elif len(self.messages) >= 2:
            assistant = self.messages[-2]
            calls = assistant.tool_uses() if assistant.role == "assistant" else []
            result_blocks = [
                block for block in last.content if isinstance(block, ToolResultBlock)
            ]
            if (
                not calls
                or len(result_blocks) != len(last.content)
                or [block.tool_use_id for block in result_blocks]
                != [call.id for call in calls]
            ):
                return
            result_message = last
            assistant_index = len(self.messages) - 2
        else:
            return

        store = self._ensure_store()
        persisted = SessionStore.load_messages(store.path)
        before_pair = self.messages[:assistant_index]
        through_assistant = self.messages[: assistant_index + 1]
        through_result = self.messages[: assistant_index + 2]

        # append_message may have completed its write before surfacing an
        # error. Reload first so retrying never duplicates the tool result.
        if persisted == through_result:
            return
        if persisted == before_pair:
            store.append_message(assistant)
            persisted = through_assistant
        if persisted == through_assistant:
            store.append_message(result_message)
            return
        raise RuntimeError(
            "session history diverged while repairing a dangling tool result"
        )
