"""交互式 REPL：prompt_toolkit 收输入，rich 渲染输出。

终端契约：prompt 激活期间不做任何 rich 渲染；一轮流式输出结束后才重新出提示符。
Repl 同时实现 UIBridge（工具的 ask/notify 走这里）。
"""

from __future__ import annotations

import asyncio
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from emiya_agent import __version__
from emiya_agent.agent.loop import AgentLoop, LoopResult
from emiya_agent.agent.prompts import build_system_prompt, load_agent_md
from emiya_agent.agent.subagent import SubagentSpawner
from emiya_agent.agent.usage import UsageTracker
from emiya_agent.compaction.manager import CompactionManager
from emiya_agent.config import CONFIG_DIR_NAME, Config, config_files_exist
from emiya_agent.errors import ConfigError, ProviderError
from emiya_agent.mcp_client.adapter import build_mcp_tools
from emiya_agent.mcp_client.manager import MCPManager
from emiya_agent.memory.manager import MemoryManager
from emiya_agent.pipeline.translation import (
    TranslationPipeline,
    complete_text,
    contains_cjk,
)
from emiya_agent.providers.registry import ProviderRegistry
from emiya_agent.session.manager import SessionManager
from emiya_agent.session.store import SessionStore
from emiya_agent.skills.loader import (
    default_skill_dirs,
    discover_skills,
    skills_index_text,
)
from emiya_agent.tools import build_default_registry
from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.memory_tool import MemoryTool
from emiya_agent.tools.skill_tool import SkillTool
from emiya_agent.tools.task import SpawnSubagentTool
from emiya_agent.types import (
    Message,
    ToolResultBlock,
    ToolUseBlock,
    user_message,
)
from emiya_agent.ui import commands, panels
from emiya_agent.ui.confirm import ConfirmUI
from emiya_agent.ui.render import StreamRenderer


class Repl:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.console = Console()
        self.providers = ProviderRegistry(config)
        self.usage = UsageTracker(config)
        self.renderer = StreamRenderer(self.console)
        self.route = config.route("executor")
        self.messages: list[Message] = []

        # skills / memory / 子agent / 压缩
        self.skills = discover_skills(default_skill_dirs(config))
        self.memory = MemoryManager(config.cwd / CONFIG_DIR_NAME / "memory")
        self.tools = build_default_registry()
        self.tools.register(SkillTool(self.skills))
        self.tools.register(MemoryTool(self.memory))
        self.spawner = SubagentSpawner(
            self.providers, config, self.tools, self.renderer, self.usage
        )
        self.tools.register(SpawnSubagentTool(self.spawner))
        self.compactor = CompactionManager(self._compactor_complete)
        self.mcp = MCPManager(config)

        self.system = build_system_prompt(
            config.cwd,
            load_agent_md(config.cwd),
            skills_index=skills_index_text(self.skills),
            memory_index=self.memory.index_text(),
        )
        self.tool_ctx = ToolContext(cwd=config.cwd, config=config, ui=self)
        self.session_mgr = SessionManager(config.project_dir())
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

    async def read_multiline(self, prompt: str, default: str = "") -> str:
        """多行编辑。TTY 下 prompt_toolkit（Esc 后 Enter 提交）；管道模式读到 EOF 行为止。"""
        if self.session is not None:
            with patch_stdout():
                return await self.session.prompt_async(
                    prompt + "\n",
                    default=default,
                    multiline=True,
                    bottom_toolbar="Esc 后按 Enter 提交 · Ctrl+C 取消",
                )
        self.console.print(
            prompt + "（输入 EOF 单独一行结束；直接 EOF 保留原文）", markup=False
        )
        lines: list[str] = []
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if line == "" or line.strip() == "EOF":
                break
            lines.append(line.rstrip("\n"))
        return "\n".join(lines) if lines else default

    # ------------------------------------------------------------- UIBridge

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        """ask_user 工具的实现：展示问题并等待输入。"""
        self.console.print(Panel(question, title="Agent 提问", border_style="magenta"))
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
        self.console.print(
            Panel(
                f"[bold]{name}[/bold]\n{args_brief}",
                title="执行确认",
                border_style="yellow",
            )
        )
        try:
            answer = (
                await self.read_line("[y]允许(回车)  [n]拒绝  [a]本会话全部允许 > ")
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False
        if answer in ("a", "always", "全部"):
            self.config.tool_approval = "auto"
            self.console.print(
                "[dim]本会话切换为自动放行（/approval ask 可改回）。[/dim]"
            )
            return True
        return answer in ("", "y", "yes", "是")

    # ------------------------------------------------------------- 主循环

    async def run(self) -> None:
        await self._maybe_first_run_init()
        self._banner()
        await self._connect_mcp()
        try:
            await self._main_loop()
        finally:
            await self.mcp.stop_all()  # 确保 stdio 子进程被终止
        self.console.print("[dim]再见。[/dim]")

    async def one_shot(self, text: str) -> None:
        """emiya -p "...": 执行一轮后退出（直通执行，非交互 → 自动放行工具）。"""
        self.config.tool_approval = "auto"
        try:
            await self._connect_mcp()
            await self._execute_turn(text)
        finally:
            await self.mcp.stop_all()

    async def _maybe_first_run_init(self) -> None:
        """首次运行（无任何配置文件）且交互式终端时，询问是否初始化。"""
        if not self.first_run or self.session is None:
            return
        from emiya_agent.ui.init_wizard import run_init_wizard

        try:
            answer = (
                await self.read_line(
                    "检测到首次运行。现在初始化模型供应商吗？[Y/n] > "
                )
            ).strip().lower()
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
        mode = "中文管线" if self.pipeline_enabled else "直通"
        self.console.print(
            f"[bold magenta]EmiyaAgent[/bold magenta] v{__version__}  ·  "
            f"执行模型 [bold]{self.route}[/bold]  ·  {mode}  ·  /help 查看命令"
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

    @property
    def pipeline_enabled(self) -> bool:
        return self.config.language == "zh"

    async def handle_input(self, text: str) -> None:
        if self.pipeline_enabled and contains_cjk(text):
            await self.pipeline_turn(text)
        else:
            await self._execute_turn(text)

    # ------------------------------------------------------------- 压缩

    async def _compactor_complete(self, system: str, text: str) -> str:
        route = self.config.route("compactor")
        provider, model = self.providers.get(route)
        result, usage = await complete_text(
            provider, model, system, text, max_tokens=self.config.max_tokens
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
        try:
            before = len(self.messages)
            with self.console.status("[dim]正在压缩上下文…[/dim]"):
                self.messages = await self.compactor.compact(self.messages)
            self._ensure_store().append_meta(
                "compaction", {"before": before, "after": len(self.messages)}
            )
            self.console.print(
                f"[dim]✂ 上下文已压缩：{before} → {len(self.messages)} 条消息"
                "（原始记录保留在会话文件中）。[/dim]"
            )
        except (ProviderError, ConfigError) as exc:
            self.console.print(f"[yellow]压缩失败（{exc}），继续使用完整历史。[/yellow]")

    # ------------------------------------------------------------- 会话

    def _ensure_store(self) -> SessionStore:
        if self.store is None:
            self.session_id, self.store = self.session_mgr.new_store()
        return self.store

    def _persist(self, message: Message) -> None:
        self._ensure_store().append_message(message)

    # ------------------------------------------------------------- 翻译管线

    async def pipeline_turn(self, text: str) -> None:
        translator_route = self.config.route("translator")
        try:
            t_provider, t_model = self.providers.get(translator_route)
        except ConfigError as exc:
            self.console.print(
                f"[yellow]翻译模型不可用（{exc}），本轮直通执行。[/yellow]"
            )
            await self._execute_turn(text)
            return

        async def translator(system: str, user_text: str) -> str:
            with self.console.status("[dim]DeepSeek 处理中…[/dim]"):
                result, usage = await complete_text(
                    t_provider,
                    t_model,
                    system,
                    user_text,
                    max_tokens=self.config.max_tokens,
                )
            self.usage.add(translator_route, usage)
            return result

        async def executor(spec_en: str) -> str:
            return await self._execute_turn(spec_en) or ""

        confirm = ConfirmUI(self.console, self.read_line, self.read_multiline)
        pipeline = TranslationPipeline(
            translator,
            confirm,
            executor,
            executor_lang=self.config.model_lang(self.route),  # zh 基座免翻译
        )
        try:
            result = await pipeline.run(text)
        except ProviderError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]已取消。[/yellow]")
            return

        if result.aborted:
            self.console.print("[dim]已放弃该需求。[/dim]")
            return
        if result.spec is not None:
            self._ensure_store().append_meta(
                "spec",
                {
                    "requirement_zh": text,
                    "spec": result.spec.spec,
                    "mode": "translate" if result.spec.mirror is not None else "native",
                },
            )
        if result.report_zh.strip():
            self._ensure_store().append_meta(
                "report", {"en": result.report_en, "zh": result.report_zh}
            )
            self.console.print(
                Panel(
                    Markdown(result.report_zh),
                    title="执行报告（中文）",
                    border_style="green",
                )
            )

    # ------------------------------------------------------------- 执行一轮

    async def _execute_turn(self, text: str) -> str | None:
        """把 text 作为用户消息，运行 agent 工具循环。返回最终文本；失败返回 None。"""
        try:
            provider, model = self.providers.get(self.route)
        except ConfigError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return None

        msg = user_message(text)
        self.messages.append(msg)
        self._persist(msg)
        self.tool_ctx.cancel = asyncio.Event()  # 每轮重置

        def on_tool_result(call: ToolUseBlock, result: ToolResultBlock) -> None:
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
        )
        try:
            result = await loop.run(self.messages, system=self.system, ctx=self.tool_ctx)
        except ProviderError as exc:
            self._repair_dangling()
            self.console.print(f"[red]{exc}[/red]")
            return None
        except (KeyboardInterrupt, EOFError):
            self.tool_ctx.cancel.set()
            self._repair_dangling()
            self.console.print("\n[yellow]已中断本轮。[/yellow]")
            return None

        text_result = self._report_result(result)
        await self.maybe_compact()
        return text_result

    def _report_result(self, result: LoopResult) -> str | None:
        if result.stop_reason == "refusal":
            self.console.print(
                f"[red]模型拒绝了该请求: {result.refusal or ''}[/red]"
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
        self.console.print(f"[dim]{footer}[/dim]")
        return result.final_text

    def _repair_dangling(self) -> None:
        """防御：若最后一条助手消息带未回应的 tool_use，补合成 tool_result。"""
        if not self.messages:
            return
        last = self.messages[-1]
        if last.role != "assistant":
            return
        calls = last.tool_uses()
        if not calls:
            return
        repair = Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id=c.id, content="Interrupted by user", is_error=True
                )
                for c in calls
            ],
        )
        self.messages.append(repair)
        self._persist(repair)
