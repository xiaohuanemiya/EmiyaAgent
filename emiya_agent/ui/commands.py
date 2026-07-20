"""斜杠命令。handler(repl, args) -> bool，返回 True 表示退出 REPL。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from rich.markup import escape
from rich.table import Table

from emiya_agent.config import (
    persist_trusted_project,
    project_config_path,
    user_config_path,
)
from emiya_agent.safe_paths import UnsafePathError
from emiya_agent.session.manager import repaired_message_copy
from emiya_agent.session.store import (
    CheckpointExistsError,
    SessionBusyError,
    SessionRecordTooLargeError,
)

if TYPE_CHECKING:
    from emiya_agent.ui.repl import Repl

CommandHandler = Callable[["Repl", str], Awaitable[bool]]

COMMANDS: dict[str, tuple[str, CommandHandler]] = {}


def command(name: str, help_text: str):
    def deco(fn: CommandHandler) -> CommandHandler:
        COMMANDS[name] = (help_text, fn)
        return fn

    return deco


async def dispatch(repl: "Repl", line: str) -> bool:
    parts = line[1:].split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    entry = COMMANDS.get(name)
    if entry is None:
        repl.console.print(f"[red]未知命令 /{name}[/red]，输入 /help 查看可用命令")
        return False
    return await entry[1](repl, args)


@command("help", "显示可用命令")
async def _help(repl: "Repl", args: str) -> bool:
    table = Table(show_header=False, box=None, padding=(0, 2))
    for name, (help_text, _) in sorted(COMMANDS.items()):
        table.add_row(f"[cyan]/{name}[/cyan]", help_text)
    repl.console.print(table)
    return False


@command("model", "查看/切换执行模型，如 /model glm/glm-4.6")
async def _model(repl: "Repl", args: str) -> bool:
    if not args.strip():
        repl.console.print(f"当前执行模型: [bold]{repl.route}[/bold]")
        repl.console.print("已配置模型:")
        for route in repl.config.models:
            marker = " ←" if route == repl.route else ""
            repl.console.print(f"  {route}{marker}")
        repl.console.print(
            "[dim]也可直接用任意 provider/model（如 deepseek/deepseek-reasoner），"
            "provider 须在 [providers] 中定义。[/dim]"
        )
        return False
    new_route = args.strip()
    try:
        repl.providers.get(new_route)  # 校验 provider + key
    except Exception as exc:
        repl.console.print(f"[red]切换失败: {exc}[/red]")
        return False
    repl.route = new_route
    repl.config.routes["executor"] = new_route
    repl.console.print(f"已切换到 [bold]{new_route}[/bold]")
    return False


@command("init", "运行初始化向导：选择供应商、设置端点/key/模型、指派各角色")
async def _init(repl: "Repl", args: str) -> bool:
    from emiya_agent.ui.init_wizard import run_init_wizard

    await run_init_wizard(repl)
    return False


@command(
    "provider",
    "管理 API 供应商：/provider add minimax（预设一步接入）或 add <名> <base_url> [KEY_ENV]",
)
async def _provider(repl: "Repl", args: str) -> bool:
    from emiya_agent.config import (
        DEFAULTS,
        ProviderConfig,
        persist_provider,
        project_config_path,
        remove_provider_from_file,
        user_config_path,
    )
    from emiya_agent.providers.presets import PRESETS

    parts = args.split()
    sub = parts[0].lower() if parts else "list"

    if sub == "list":
        table = Table(box=None, padding=(0, 2))
        table.add_column("供应商", style="cyan")
        table.add_column("端点")
        table.add_column("模型语言")
        table.add_column("凭据")
        for name, pc in repl.config.providers.items():
            source = repl.config.credential_source(name)
            cred = (
                f"[green]✓ {source}[/green]"
                if source
                else f"[red]✗ 需设 {pc.api_key_env}[/red]"
            )
            lang = f"主要语言 {pc.lang}"
            table.add_row(name, pc.base_url or "(官方默认)", lang, cred)
        repl.console.print(table)
        not_added = [n for n in PRESETS if n not in repl.config.providers]
        if not_added:
            repl.console.print(
                f"[dim]可一步接入的预设: {', '.join(not_added)}"
                "（/provider add <名称>）；其他 OpenAI 兼容端点用 "
                "/provider add <名> <base_url> [KEY_ENV][/dim]"
            )
        return False

    if sub == "add":
        if len(parts) < 2:
            repl.console.print(
                "[red]用法: /provider add <名称> [base_url] [API_KEY_ENV][/red]"
            )
            return False
        name = parts[1].lower()
        example_model = "<模型ID>"
        if len(parts) >= 3:
            base_url = parts[2]
            env = parts[3] if len(parts) >= 4 else f"{name.upper()}_API_KEY"
            lang = parts[4] if len(parts) >= 5 and parts[4] in ("zh", "en") else "en"
            pc = ProviderConfig(api_key_env=env, base_url=base_url, lang=lang)
        else:
            preset = PRESETS.get(name)
            if preset is None:
                repl.console.print(
                    f"[red]没有 {name!r} 的预设。可用预设: {', '.join(PRESETS)}；"
                    "或手动指定: /provider add "
                    f"{name} <base_url> [API_KEY_ENV] [zh|en][/red]"
                )
                return False
            pc = ProviderConfig(
                api_key_env=preset.api_key_env,
                base_url=preset.base_url,
                lang=preset.lang,
            )
            example_model = preset.example_model
            repl.console.print(f"[dim]{preset.note}[/dim]")
        if pc.lang == "zh":
            repl.console.print(
                "[dim]主要语言 zh：翻译 Agent 会在向该 provider 的模型交接时"
                "把它作为目标语言。[/dim]"
            )

        repl.config.providers[name] = pc  # 本会话立即生效
        repl.providers.invalidate(name)
        path = user_config_path()
        persist_provider(path, name, pc)
        repl.console.print(
            f"[green]✓ 已添加供应商 {name}[/green]（写入 {path}，全局生效）"
        )
        if repl.config.credential_source(name) is None:
            repl.console.print(
                f'  1. 设置 key: $env:{pc.api_key_env}="..."'
                f'（持久化: setx {pc.api_key_env} "..."）'
            )
        repl.console.print(
            f"  2. 路由到它: /model {name}/{example_model} 或 "
            f"/route translator {name}/{example_model}（模型 ID 见官方文档）"
        )
        return False

    if sub == "remove":
        if len(parts) < 2:
            repl.console.print("[red]用法: /provider remove <名称>[/red]")
            return False
        name = parts[1].lower()
        if name in DEFAULTS["providers"]:
            repl.console.print(
                f"[yellow]{name} 是内置供应商，不可删除（可在配置中覆盖其 base_url/key_env）。[/yellow]"
            )
            return False
        removed = remove_provider_from_file(user_config_path(), name)
        removed = (
            remove_provider_from_file(project_config_path(repl.config.cwd), name)
            or removed
        )
        repl.config.providers.pop(name, None)
        repl.providers.invalidate(name)
        if removed:
            repl.console.print(f"已删除供应商 {name}。")
        else:
            repl.console.print(f"[yellow]配置文件中没有 {name}。[/yellow]")
        return False

    repl.console.print(f"[red]未知子命令 {sub!r}，可用: list / add / remove[/red]")
    return False


@command("route", "查看/设置任意角色的模型路由：/route [角色] [provider/model]")
async def _route(repl: "Repl", args: str) -> bool:
    parts = args.split()
    routes = repl.config.routes
    if not parts:
        repl.console.print("各角色路由（/route 角色 provider/model 修改）:")
        for role, route in routes.items():
            live = repl.route if role == "executor" else route
            repl.console.print(f"  {role:18s} → {live}")
        return False
    role = parts[0]
    if role not in routes:
        repl.console.print(f"[red]未知角色 {role!r}，可用: {', '.join(routes)}[/red]")
        return False
    if len(parts) == 1:
        current = repl.route if role == "executor" else routes[role]
        repl.console.print(f"{role} → {current}")
        return False
    new_route = parts[1]
    try:
        repl.providers.get(new_route)  # 校验 provider + key
    except Exception as exc:
        repl.console.print(f"[red]设置失败: {exc}[/red]")
        return False
    routes[role] = new_route
    if role == "executor":
        repl.route = new_route
    repl.console.print(f"{role} → [bold]{new_route}[/bold]")
    return False


@command("config", "显示当前生效配置与 API key 状态")
async def _config(repl: "Repl", args: str) -> bool:
    cfg = repl.config
    repl.console.print(
        "输入语言: 自动  ·  plan: Agent 自主"
        f"  ·  max_tokens: {cfg.max_tokens}  ·  effort: {cfg.effort}"
    )
    project_state = (
        "已信任"
        if cfg.project_trusted
        else "未信任（项目配置已忽略）"
        if cfg.project_config_ignored
        else "未配置"
    )
    path_scope = "允许外部目录" if cfg.allow_external_paths else "仅工作目录"
    repl.console.print(
        f"安全策略: approval={cfg.tool_approval}  ·  文件={path_scope}  ·  "
        f"项目配置={project_state}"
    )
    repl.console.print(
        f"命令策略: {len(cfg.shell_rules)} 条规则  ·  "
        f"shell 环境白名单={len(cfg.shell_inherit_env)}  ·  "
        f"MCP 环境白名单={len(cfg.mcp_inherit_env)}"
        f" + {len(cfg.mcp_server_inherit_env)} 个服务器覆盖"
    )
    repl.console.print(
        f"Hook: {len(cfg.hooks)} 个用户级命令  ·  "
        f"环境白名单={len(cfg.hook_inherit_env)}"
    )
    lsp_state = "已启用" if cfg.lsp_enabled else "已禁用"
    repl.console.print(
        f"LSP: {lsp_state}  ·  {len(cfg.lsp_servers)} 个配置  ·  "
        f"{repl.lsp.connected_count} 个连接  ·  "
        f"环境白名单={len(cfg.lsp_inherit_env)}"
    )
    repl.console.print("模型路由:")
    for role, route in cfg.routes.items():
        repl.console.print(f"  {role:18s} → {route}")
    repl.console.print("凭据状态:")
    for name in cfg.providers:
        source = cfg.credential_source(name)
        if source:
            repl.console.print(f"  {name:10s} [green]✓[/green]  来源: {source}")
        else:
            env = cfg.providers[name].api_key_env
            hint = f"设置 {env}"
            if name == "anthropic":
                hint += " 或 `ant auth login`"
            repl.console.print(f"  {name:10s} [red]✗ 不可用[/red]（{hint}）")
    return False


@command("hooks", "列出用户级 PreToolUse/PostToolUse 命令 Hook")
async def _hooks(repl: "Repl", args: str) -> bool:
    if not repl.config.hooks:
        repl.console.print(
            "[dim]暂无 Hook；仅从用户级 ~/.emiyacode/config.toml 加载。[/dim]"
        )
        return False
    table = Table(box=None, padding=(0, 2))
    for column in ("ID", "事件", "匹配", "命令", "失败策略"):
        table.add_column(column)
    for hook in repl.config.hooks:
        table.add_row(
            escape(hook.id),
            escape(hook.event.value),
            escape(hook.matcher),
            escape(" ".join(hook.command)),
            escape(hook.on_error),
        )
    repl.console.print(table)
    repl.console.print("[dim]Hook 只能收紧既有权限；allow 不会绕过 ask/deny。[/dim]")
    return False


@command("lsp", "查看 Language Server 配置与活动连接")
async def _lsp(repl: "Repl", args: str) -> bool:
    state = "[green]已启用[/green]" if repl.config.lsp_enabled else "[dim]已禁用[/dim]"
    repl.console.print(f"LSP: {state} · 活动连接: {repl.lsp.connected_count}")
    if not repl.config.lsp_servers:
        repl.console.print("[dim]未配置 language server。[/dim]")
        return False
    table = Table(box=None, padding=(0, 2))
    for column in ("服务器", "命令", "扩展名", "优先级", "状态"):
        table.add_column(column)
    for name, server in sorted(
        repl.config.lsp_servers.items(),
        key=lambda item: (-item[1].priority, item[0]),
    ):
        table.add_row(
            escape(name),
            escape(" ".join((server.command, *server.args))),
            escape(", ".join(sorted(server.extensions))),
            str(server.priority),
            "禁用" if server.disabled else "候选",
        )
    repl.console.print(table)
    repl.console.print(
        "[dim]服务器按文件类型、优先级与本地可执行文件选择；不会自动下载。"
        "首次启动每个 server/root 时按审批策略展示实际命令。[/dim]"
    )
    return False


@command("approval", "工具审批模式：/approval ask（写操作须确认）| auto（直接执行）")
async def _approval(repl: "Repl", args: str) -> bool:
    arg = args.strip().lower()
    if not arg:
        mode = repl.config.tool_approval
        desc = (
            "写操作/shell/MCP 执行前询问"
            if mode == "ask"
            else "默认直接执行（显式 ask/deny 规则仍优先）"
        )
        repl.console.print(f"当前审批模式: [bold]{mode}[/bold]（{desc}）")
        return False
    if arg not in ("ask", "auto"):
        repl.console.print("[red]用法: /approval ask|auto[/red]")
        return False
    repl.config.tool_approval = arg
    cancelled: list[str] = []
    if arg == "ask":
        for record in repl.task_service.list():
            if record.spec.role == "general" and await repl.task_service.cancel(
                record.id
            ):
                cancelled.append(record.id)
        for task_id in cancelled:
            await repl.task_service.wait(task_id)
    repl.console.print(
        "已切换到 [bold]ask[/bold]：shell/写文件/MCP 执行前会请求确认"
        if arg == "ask"
        else "已切换到 [bold]auto[/bold]：默认直接执行；显式 ask/deny 规则仍优先"
    )
    if cancelled:
        repl.console.print(
            f"[yellow]已取消 {len(cancelled)} 个后台 general 任务；"
            "其 worktree 保留供检查。[/yellow]"
        )
    return False


@command("trust", "信任当前项目的 .emiyacode/config.toml（重启后生效）")
async def _trust(repl: "Repl", args: str) -> bool:
    path = project_config_path(repl.config.cwd)
    if not path.is_file():
        repl.console.print("[dim]当前项目没有 .emiyacode/config.toml。[/dim]")
        return False
    if repl.config.project_trusted:
        repl.console.print("[dim]当前项目配置已经受信任。[/dim]")
        return False
    persist_trusted_project(user_config_path(), repl.config.cwd)
    repl.console.print(
        "[yellow]已将当前项目加入用户级信任列表。请检查项目配置后重启；"
        "其中的 provider 端点与 MCP 命令将在下次启动生效。[/yellow]"
    )
    return False


@command("clear", "清空当前对话上下文")
async def _clear(repl: "Repl", args: str) -> bool:
    if repl.store is not None:
        # 保留同一 JSONL 中的旧记录用于审计，但恢复时从空上下文继续回放。
        try:
            repl.store.append_snapshot([])
        except (
            SessionRecordTooLargeError,
            UnsafePathError,
            OSError,
            ValueError,
        ) as exc:
            repl.console.print(f"[red]清空失败，上下文未改变: {escape(str(exc))}[/red]")
            return False
    repl.messages.clear()
    repl.console.print("[dim]对话上下文已清空。[/dim]")
    return False


@command("mcp", "查看 MCP 服务器状态与工具")
async def _mcp(repl: "Repl", args: str) -> bool:
    status = repl.mcp.status()
    if not status:
        repl.console.print(
            "[dim]未配置 MCP 服务器。在 .emiyacode/config.toml 中添加：\n"
            "[mcp_servers.名称]\n"
            'command = "npx"  # stdio 方式\n'
            'args = ["-y", "@modelcontextprotocol/server-everything"]\n'
            '# 或 HTTP 方式: url = "https://example.com/mcp"[/dim]'
        )
        return False
    table = Table(box=None, padding=(0, 2))
    table.add_column("服务器", style="cyan")
    table.add_column("类型")
    table.add_column("状态")
    table.add_column("工具数")
    for name, info in status.items():
        state = (
            "[green]已连接[/green]"
            if info["connected"]
            else f"[red]失败: {info['error'] or '未知'}[/red]"
        )
        table.add_row(name, info["kind"], state, str(info["tools"]))
    repl.console.print(table)
    mcp_tools = [n for n in repl.tools.names() if n.startswith("mcp__")]
    if mcp_tools:
        repl.console.print(f"[dim]MCP 工具: {', '.join(mcp_tools)}[/dim]")
    repl.console.print(
        "[dim]注：MCP 服务器在启动时连接；修改配置后需重启 emiya 生效"
        "（会话中途增删工具会破坏提示词缓存）。[/dim]"
    )
    return False


@command("skills", "列出可用技能")
async def _skills(repl: "Repl", args: str) -> bool:
    if not repl.skills:
        repl.console.print(
            "[dim]暂无技能。在 .emiyacode/skills/<名称>/SKILL.md 创建"
            "（YAML frontmatter 写 name/description）。[/dim]"
        )
        return False
    table = Table(box=None, padding=(0, 2))
    table.add_column("技能", style="cyan")
    table.add_column("描述")
    table.add_column("位置", style="dim")
    for skill in sorted(repl.skills.values(), key=lambda s: s.name):
        table.add_row(skill.name, skill.description, str(skill.path.parent))
    repl.console.print(table)
    return False


@command("memory", "查看项目记忆（索引与文件列表）")
async def _memory(repl: "Repl", args: str) -> bool:
    index = repl.memory.index_text()
    files = repl.memory.list_files()
    if index:
        repl.console.print("[bold]MEMORY.md 索引：[/bold]")
        repl.console.print(index, markup=False)
    if files:
        repl.console.print(f"[dim]记忆文件（{len(files)}）: {', '.join(files)}[/dim]")
    if not index and not files:
        repl.console.print(
            "[dim]暂无项目记忆。agent 会通过 memory 工具自动积累，"
            "存于 .emiyacode/memory/。[/dim]"
        )
    return False


@command("compact", "立即压缩当前对话上下文")
async def _compact(repl: "Repl", args: str) -> bool:
    await repl.maybe_compact(force=True)
    return False


def _task_summary(record) -> tuple[str, str, str, str, str]:
    prompt = record.spec.prompt.strip().splitlines()[0][:80]
    return (
        escape(record.id),
        escape(record.status.value),
        escape(record.spec.role),
        escape(record.spec.isolation),
        escape(prompt),
    )


@command("tasks", "列出后台子agent任务")
async def _tasks(repl: "Repl", args: str) -> bool:
    records = repl.task_service.list()
    if not records:
        repl.console.print("[dim]暂无后台子agent任务。[/dim]")
        return False
    table = Table(box=None, padding=(0, 2))
    for column in ("任务 ID", "状态", "角色", "隔离", "任务"):
        table.add_column(column)
    for record in records:
        table.add_row(*_task_summary(record))
    repl.console.print(table)
    repl.console.print(
        "[dim]/task show|wait|cancel|cleanup <任务ID>；"
        "清理含改动的 worktree 需显式 --force[/dim]"
    )
    return False


@command(
    "task",
    "管理后台任务：/task show|wait|cancel|cleanup <任务ID> [超时秒数|--force]",
)
async def _task(repl: "Repl", args: str) -> bool:
    parts = args.split()
    if len(parts) < 2:
        repl.console.print(
            "[red]用法: /task show|wait|cancel|cleanup "
            "<任务ID> [超时秒数|--force][/red]"
        )
        return False
    action, task_id = parts[0].lower(), parts[1]
    record = repl.task_service.get(task_id)
    if record is None:
        repl.console.print(f"[red]未知任务 {escape(task_id)}[/red]")
        return False

    if action in {"show", "get"}:
        table = Table(show_header=False, box=None, padding=(0, 2))
        for label, value in zip(
            ("任务 ID", "状态", "角色", "隔离", "任务"),
            _task_summary(record),
            strict=True,
        ):
            table.add_row(label, value)
        if record.route:
            table.add_row("模型", escape(record.route))
        worktree = repl.task_service.worktree(task_id)
        if worktree is not None:
            table.add_row("worktree", escape(str(worktree.path)))
            table.add_row("分支", escape(worktree.branch or "(detached)"))
        repl.console.print(table)
        if record.report:
            repl.console.print(record.report, markup=False)
        if record.error:
            repl.console.print(f"[red]{escape(record.error)}[/red]")
        return False

    if action == "wait":
        timeout: float | None = None
        if len(parts) >= 3:
            try:
                timeout = float(parts[2])
            except ValueError:
                repl.console.print("[red]超时必须是秒数。[/red]")
                return False
            if not 0 <= timeout <= 600:
                repl.console.print("[red]超时必须在 0 到 600 秒之间。[/red]")
                return False
        try:
            record = await repl.task_service.wait(task_id, timeout=timeout)
        except TimeoutError:
            current = repl.task_service.get(task_id)
            status = current.status.value if current else "unknown"
            repl.console.print(f"[dim]任务仍在运行（{escape(status)}）。[/dim]")
            return False
        repl.console.print(
            f"任务 {escape(task_id)}: [bold]{escape(record.status.value)}[/bold]"
        )
        output = record.report or record.error
        if output:
            repl.console.print(output, markup=False)
        return False

    if action == "cancel":
        changed = await repl.task_service.cancel(task_id)
        repl.console.print(
            f"已请求取消任务 {escape(task_id)}。"
            if changed
            else f"[dim]任务 {escape(task_id)} 已结束。[/dim]"
        )
        return False

    if action == "cleanup":
        force = "--force" in parts[2:]
        try:
            await repl.task_service.cleanup(task_id, force=force)
        except (KeyError, RuntimeError, ValueError) as exc:
            repl.console.print(f"[red]清理失败: {escape(str(exc))}[/red]")
            return False
        repl.console.print(f"已清理任务 {escape(task_id)} 的 worktree。")
        return False

    repl.console.print(
        f"[red]未知操作 {escape(action)!r}；可用 show/wait/cancel/cleanup[/red]"
    )
    return False


@command("checkpoint", "保存命名检查点：/checkpoint <名称>")
async def _checkpoint(repl: "Repl", args: str) -> bool:
    name = args.strip()
    if not name:
        repl.console.print("[red]用法: /checkpoint <名称>[/red]")
        return False
    if repl.store is None or repl.session_id is None:
        repl.console.print("[red]当前还没有可保存的会话。[/red]")
        return False
    try:
        info = repl.store.append_checkpoint(name, repl.messages)
    except (
        CheckpointExistsError,
        SessionRecordTooLargeError,
        UnsafePathError,
        OSError,
        ValueError,
    ) as exc:
        repl.console.print(f"[red]保存检查点失败: {escape(str(exc))}[/red]")
        return False
    repl.console.print(
        f"[green]已保存检查点 {escape(info.name)}[/green]"
        f"（{info.message_count} 条消息）。"
    )
    return False


@command("checkpoints", "列出当前会话的命名检查点")
async def _checkpoints(repl: "Repl", args: str) -> bool:
    if repl.store is None or repl.session_id is None:
        repl.console.print("[dim]当前还没有会话。[/dim]")
        return False
    try:
        infos = repl.store.list_checkpoints()
    except (UnsafePathError, OSError) as exc:
        repl.console.print(f"[red]读取检查点失败: {escape(str(exc))}[/red]")
        return False
    if not infos:
        repl.console.print("[dim]当前会话暂无命名检查点。[/dim]")
        return False
    table = Table(box=None, padding=(0, 2))
    table.add_column("检查点", style="cyan")
    table.add_column("消息数", justify="right")
    table.add_column("创建时间")
    table.add_column("首条消息")
    for info in infos:
        table.add_row(
            escape(info.name),
            str(info.message_count),
            escape(info.created_at),
            escape(info.preview or "(空)"),
        )
    repl.console.print(table)
    repl.console.print("[dim]/fork <检查点> 从该位置创建独立会话[/dim]")
    return False


@command("fork", "分叉当前会话：/fork [检查点名称]")
async def _fork(repl: "Repl", args: str) -> bool:
    if repl.store is None or repl.session_id is None:
        repl.console.print("[red]当前还没有可分叉的会话。[/red]")
        return False
    checkpoint_name = args.strip() or None
    parent_id = repl.session_id
    try:
        if checkpoint_name is None:
            forked = repl.session_mgr.fork(
                parent_id, repl.messages, acquire_writer=True
            )
        else:
            forked = repl.session_mgr.fork(
                parent_id,
                checkpoint_name=checkpoint_name,
                acquire_writer=True,
            )
    except (
        KeyError,
        RuntimeError,
        SessionRecordTooLargeError,
        UnsafePathError,
        OSError,
        ValueError,
    ) as exc:
        repl.console.print(f"[red]分叉会话失败: {escape(str(exc))}[/red]")
        return False

    child_id, child_store, child_messages = forked
    # 所有读取、校验和写入均成功后再一次性切换，失败路径保持父会话不变。
    parent_store = repl.store
    repl.session_id = child_id
    repl.store = child_store
    repl.messages = child_messages
    parent_store.release_writer()
    source = (
        f"检查点 {escape(checkpoint_name)}"
        if checkpoint_name is not None
        else "当前上下文"
    )
    repl.console.print(
        f"已从{source}分叉为会话 [bold]{escape(child_id)}[/bold]；"
        f"父会话 {escape(parent_id)} 保持不变。"
    )
    return False


@command("sessions", "列出历史会话")
async def _sessions(repl: "Repl", args: str) -> bool:
    infos = repl.session_mgr.list_sessions()
    if not infos:
        repl.console.print("[dim]暂无历史会话。[/dim]")
        return False
    table = Table(box=None, padding=(0, 2))
    table.add_column("会话 ID", style="cyan")
    table.add_column("来源")
    table.add_column("首条消息")
    for info in infos:
        marker = " ←当前" if info.session_id == repl.session_id else ""
        source = ""
        if info.parent_session_id:
            source = info.parent_session_id
            if info.checkpoint_name:
                source += f"@{info.checkpoint_name}"
        table.add_row(
            escape(info.session_id) + marker,
            escape(source or "—"),
            escape(info.preview or "(空)"),
        )
    repl.console.print(table)
    repl.console.print(
        "[dim]/resume <ID> 恢复指定会话；/checkpoint 与 /fork 管理分支[/dim]"
    )
    return False


@command("resume", "恢复会话：/resume [会话ID]（缺省最近一个）")
async def _resume(repl: "Repl", args: str) -> bool:
    target = args.strip() or None
    if repl.session_id is not None and (target is None or target == repl.session_id):
        repl.console.print(f"[dim]会话 {escape(repl.session_id)} 已是当前会话。[/dim]")
        return False
    try:
        resumed = repl.session_mgr.resume(target, acquire_writer=True)
    except (OSError, SessionBusyError, UnsafePathError, ValueError) as exc:
        repl.console.print(
            f"[red]恢复会话失败，当前会话未改变: {escape(str(exc))}[/red]"
        )
        return False
    if resumed is None:
        repl.console.print("[red]未找到可恢复的会话。[/red]")
        return False
    session_id, store, messages = resumed
    candidate_messages, repaired = repaired_message_copy(
        messages,
        "Tool call not executed because the previous session ended before "
        "recording its result",
    )
    if repaired:
        try:
            # 中间 tool_result 也可能因损坏行而缺失；完整快照可让所有
            # 修复位置成为新的恢复基线，而非只在文件尾追加一条结果。
            store.append_snapshot(candidate_messages)
        except (
            SessionRecordTooLargeError,
            UnsafePathError,
            OSError,
            ValueError,
        ) as exc:
            store.release_writer()
            repl.console.print(
                f"[red]恢复会话失败，修复记录无法持久化: {escape(str(exc))}[/red]"
            )
            return False
    # 候选上下文已完整读取且必要修复已落盘后，才切换当前会话。
    previous_store = repl.store
    repl.session_id = session_id
    repl.store = store
    repl.messages = candidate_messages
    if previous_store is not None:
        previous_store.release_writer()
    repl.console.print(
        f"已恢复会话 [bold]{session_id}[/bold]（{len(candidate_messages)} 条消息）。"
    )
    if repaired:
        repl.console.print("[yellow]已修复上次中断留下的未配对工具调用。[/yellow]")
    # 展示最近的对话摘要帮助回忆
    for msg in candidate_messages[-2:]:
        text = msg.text().strip()
        if text:
            role = "你" if msg.role == "user" else "助手"
            preview = text[:200] + ("…" if len(text) > 200 else "")
            repl.console.print(f"{role}: {preview}", markup=False, style="dim")
    return False


@command("exit", "退出")
async def _exit(repl: "Repl", args: str) -> bool:
    return True
