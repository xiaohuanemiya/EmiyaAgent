"""斜杠命令。handler(repl, args) -> bool，返回 True 表示退出 REPL。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from rich.table import Table

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
        table.add_column("语言基座")
        table.add_column("凭据")
        for name, pc in repl.config.providers.items():
            source = repl.config.credential_source(name)
            cred = (
                f"[green]✓ {source}[/green]"
                if source
                else f"[red]✗ 需设 {pc.api_key_env}[/red]"
            )
            lang = "中文(免译)" if pc.lang == "zh" else "英文(启用翻译)"
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
                "[dim]语言基座 zh：路由为执行模型时，中文需求免翻译直接 plan 确认。[/dim]"
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
                f"（持久化: setx {pc.api_key_env} \"...\"）"
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
        removed = remove_provider_from_file(
            project_config_path(repl.config.cwd), name
        ) or removed
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
        repl.console.print(
            f"[red]未知角色 {role!r}，可用: {', '.join(routes)}[/red]"
        )
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
    repl.console.print(f"语言模式: {cfg.language}  ·  max_tokens: {cfg.max_tokens}  ·  effort: {cfg.effort}")
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


@command("approval", "工具审批模式：/approval ask（写操作须确认）| auto（直接执行）")
async def _approval(repl: "Repl", args: str) -> bool:
    arg = args.strip().lower()
    if not arg:
        mode = repl.config.tool_approval
        desc = "写操作/shell/MCP 执行前询问" if mode == "ask" else "全部直接执行"
        repl.console.print(f"当前审批模式: [bold]{mode}[/bold]（{desc}）")
        return False
    if arg not in ("ask", "auto"):
        repl.console.print("[red]用法: /approval ask|auto[/red]")
        return False
    repl.config.tool_approval = arg
    repl.console.print(
        "已切换到 [bold]ask[/bold]：shell/写文件/MCP 执行前会请求确认"
        if arg == "ask"
        else "已切换到 [bold]auto[/bold]：所有工具直接执行（注意风险）"
    )
    return False


@command("lang", "查看/切换语言模式：/lang zh 启用中文管线，/lang en 直通")
async def _lang(repl: "Repl", args: str) -> bool:
    arg = args.strip().lower()
    if not arg:
        mode = "zh（中文管线）" if repl.config.language == "zh" else "en（直通）"
        repl.console.print(f"当前语言模式: {mode}")
        return False
    if arg not in ("zh", "en"):
        repl.console.print("[red]用法: /lang zh|en[/red]")
        return False
    repl.config.language = arg
    repl.console.print(
        "已切换到 [bold]中文管线[/bold]（中文输入将先扩写为英文 spec 并确认）"
        if arg == "zh"
        else "已切换到 [bold]直通[/bold] 模式（输入原样交给执行模型）"
    )
    return False


@command("clear", "清空当前对话上下文")
async def _clear(repl: "Repl", args: str) -> bool:
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
            "# 或 HTTP 方式: url = \"https://example.com/mcp\"[/dim]"
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


@command("sessions", "列出历史会话")
async def _sessions(repl: "Repl", args: str) -> bool:
    infos = repl.session_mgr.list_sessions()
    if not infos:
        repl.console.print("[dim]暂无历史会话。[/dim]")
        return False
    table = Table(box=None, padding=(0, 2))
    table.add_column("会话 ID", style="cyan")
    table.add_column("首条消息")
    for info in infos:
        marker = " ←当前" if info.session_id == repl.session_id else ""
        table.add_row(info.session_id + marker, info.preview or "(空)")
    repl.console.print(table)
    repl.console.print("[dim]/resume <ID> 恢复指定会话，/resume 恢复最近会话[/dim]")
    return False


@command("resume", "恢复会话：/resume [会话ID]（缺省最近一个）")
async def _resume(repl: "Repl", args: str) -> bool:
    target = args.strip() or None
    resumed = repl.session_mgr.resume(target)
    if resumed is None:
        repl.console.print("[red]未找到可恢复的会话。[/red]")
        return False
    session_id, store, messages = resumed
    repl.session_id = session_id
    repl.store = store
    repl.messages = messages
    repl.console.print(
        f"已恢复会话 [bold]{session_id}[/bold]（{len(messages)} 条消息）。"
    )
    # 展示最近的对话摘要帮助回忆
    for msg in messages[-2:]:
        text = msg.text().strip()
        if text:
            role = "你" if msg.role == "user" else "助手"
            preview = text[:200] + ("…" if len(text) > 200 else "")
            repl.console.print(f"{role}: {preview}", markup=False, style="dim")
    return False


@command("exit", "退出")
async def _exit(repl: "Repl", args: str) -> bool:
    return True
