"""首次运行初始化向导。

流程：选择要配置的供应商（可多选/全跳过）→ 逐家设置 base_url（带默认值与
备选端点，如 DeepSeek 的 Anthropic 兼容端点）、API key、模型名 →
给每个智能体角色指派模型 → 写回用户级配置（~/.emiyacode/config.toml）。

/init 命令可随时重跑；首次运行（无配置文件）自动触发询问。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emiya_agent.config import (
    Config,
    ModelInfo,
    ProviderConfig,
    persist_models,
    persist_provider,
    persist_routes,
    user_config_path,
    write_config_template,
)
from emiya_agent.providers.presets import PRESETS


@dataclass
class WizardEntry:
    name: str
    note: str
    api_key_env: str
    base_url: str | None  # None = 官方默认端点
    protocol: str
    lang: str
    example_models: list[str]
    # 备选端点：(标签, url, 协议)
    alt_urls: list[tuple[str, str, str]] = field(default_factory=list)


def build_catalog() -> list[WizardEntry]:
    catalog = [
        WizardEntry(
            name="deepseek",
            note="DeepSeek",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            protocol="openai",
            lang="zh",
            example_models=["deepseek-v4-flash", "deepseek-v4-pro"],
            alt_urls=[
                (
                    "Anthropic 兼容端点",
                    "https://api.deepseek.com/anthropic",
                    "anthropic",
                )
            ],
        ),
        WizardEntry(
            name="glm",
            note="智谱 GLM",
            api_key_env="GLM_API_KEY",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            protocol="openai",
            lang="zh",
            example_models=["glm-4.6"],
        ),
        WizardEntry(
            name="anthropic",
            note="Anthropic Claude 官方",
            api_key_env="ANTHROPIC_API_KEY",
            base_url=None,
            protocol="anthropic",
            lang="en",
            example_models=["claude-fable-5", "claude-opus-4-8"],
        ),
        WizardEntry(
            name="openai",
            note="OpenAI GPT 官方",
            api_key_env="OPENAI_API_KEY",
            base_url=None,
            protocol="openai",
            lang="en",
            example_models=["gpt-5.6"],
        ),
    ]
    for preset in PRESETS.values():
        catalog.append(
            WizardEntry(
                name=preset.name,
                note=preset.note,
                api_key_env=preset.api_key_env,
                base_url=preset.base_url,
                protocol="openai",
                lang=preset.lang,
                example_models=[preset.example_model],
            )
        )
    return catalog


ROLE_ORDER = [
    "executor",
    "translator",
    "compactor",
    "subagent_explore",
    "subagent_plan",
    "subagent_general",
]

ROLE_NOTES = {
    "executor": "主执行 agent（工具循环，干活主力）",
    "translator": "下一跳语言适配（建议便宜模型）",
    "compactor": "上下文压缩摘要（建议便宜模型）",
    "subagent_explore": "只读搜索子 agent（建议便宜模型）",
    "subagent_plan": "方案设计子 agent（建议强模型）",
    "subagent_general": "通用执行子 agent",
}


def parse_selection(text: str, catalog: list[WizardEntry]) -> list[WizardEntry]:
    """解析多选输入：'1,3' / 'deepseek,glm' / 'all'；空串 → []。"""
    text = text.strip().lower()
    if not text:
        return []
    if text in ("all", "全部"):
        return list(catalog)
    by_name = {e.name: e for e in catalog}
    selected: list[WizardEntry] = []
    for token in text.replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        entry = None
        if token.isdigit() and 1 <= int(token) <= len(catalog):
            entry = catalog[int(token) - 1]
        elif token in by_name:
            entry = by_name[token]
        if entry is not None and entry not in selected:
            selected.append(entry)
    return selected


async def run_init_wizard(repl, *, first_run: bool = False) -> None:
    """repl 需提供 console / read_line / config / providers / route 属性。"""
    console = repl.console
    try:
        wrote = await _wizard(repl)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]已取消初始化。[/yellow]")
        wrote = False
    if not wrote and first_run:
        # 生成带注释的模板，避免下次启动重复提醒
        path = user_config_path()
        write_config_template(path)
        console.print(f"[dim]已生成配置模板 {path}；随时可用 /init 重新配置。[/dim]")


async def _wizard(repl) -> bool:
    console = repl.console
    config: Config = repl.config
    catalog = build_catalog()

    console.print("[bold]== EmiyaAgent 初始化向导 ==[/bold]")
    console.print("选择要配置的供应商（可多选，未选中的将跳过）:")
    for i, entry in enumerate(catalog, 1):
        lang = f"主要语言 {entry.lang}"
        console.print(f"  {i}. {entry.name:12s} {entry.note}（{lang}）", markup=False)
    raw = await repl.read_line(
        "编号或名称，逗号分隔（如 1,3 / deepseek,glm / all；回车跳过）> "
    )
    selected = parse_selection(raw, catalog)
    if not selected:
        console.print("[dim]未选择任何供应商。[/dim]")
        return False

    configured: dict[str, ProviderConfig] = {}
    model_routes: list[str] = []
    for entry in selected:
        pc, models = await _configure_provider(repl, entry)
        configured[entry.name] = pc
        model_routes += [f"{entry.name}/{m}" for m in models]

    # 角色指派
    routes = dict(config.routes)
    if model_routes:
        console.print("\n[bold]— 给每个智能体角色指派模型 —[/bold]")
        candidates = list(dict.fromkeys(model_routes))  # 去重保序
        for role in ROLE_ORDER:
            current = routes.get(role, "")
            # 当前路由恰好在本次配置的模型里才保留，否则建议第一个新模型
            suggested = current if current in candidates else candidates[0]
            console.print(f"\n{role} —— {ROLE_NOTES.get(role, '')}", markup=False)
            for i, cand in enumerate(candidates, 1):
                mark = " ←建议" if cand == suggested else ""
                console.print(f"  {i}. {cand}{mark}", markup=False)
            answer = (await repl.read_line(f"选择编号（回车={suggested}）> ")).strip()
            if answer.isdigit() and 1 <= int(answer) <= len(candidates):
                routes[role] = candidates[int(answer) - 1]
            elif answer:
                routes[role] = answer  # 允许直接敲 provider/model
            else:
                routes[role] = suggested

    # 持久化（用户级全局配置）
    path = user_config_path()
    for name, pc in configured.items():
        persist_provider(path, name, pc)
    if model_routes:
        persist_models(path, model_routes)
        persist_routes(path, routes)

    # 本会话立即生效
    config.providers.update(configured)
    for route in model_routes:
        config.models.setdefault(route, ModelInfo())
    config.routes.update(routes)
    for name in configured:
        repl.providers.invalidate(name)
    repl.route = config.route("executor")

    console.print(f"\n[green]✓ 初始化完成，已写入 {path}[/green]")
    console.print("角色路由:")
    for role in ROLE_ORDER:
        console.print(f"  {role:18s} → {config.routes.get(role, '')}", markup=False)
    missing = [
        f"{n}（{configured[n].api_key_env}）"
        for n in configured
        if config.credential_source(n) is None
    ]
    if missing:
        console.print(
            f"[yellow]⚠ 以下供应商还没有可用 key: {', '.join(missing)}[/yellow]"
        )
    return True


async def _configure_provider(
    repl, entry: WizardEntry
) -> tuple[ProviderConfig, list[str]]:
    console = repl.console
    console.print(f"\n[bold]— 配置 {entry.name}（{entry.note}）—[/bold]")

    # 1) base_url + 协议
    base_url = entry.base_url
    protocol = entry.protocol
    default_label = base_url or "官方默认端点"
    console.print(f"端点 base_url（回车 = {default_label}）", markup=False)
    for i, (label, url, _proto) in enumerate(entry.alt_urls, 2):
        console.print(f"  [{i}] {label}: {url}", markup=False)
    answer = (await repl.read_line("URL 或备选编号 > ")).strip()
    if answer:
        picked_alt = None
        if answer.isdigit():
            idx = int(answer) - 2
            if 0 <= idx < len(entry.alt_urls):
                picked_alt = entry.alt_urls[idx]
        if picked_alt is not None:
            base_url, protocol = picked_alt[1], picked_alt[2]
        else:
            base_url = answer
            proto_answer = (
                (await repl.read_line("协议（回车=openai 兼容；a=anthropic 兼容）> "))
                .strip()
                .lower()
            )
            if proto_answer in ("a", "anthropic"):
                protocol = "anthropic"

    # 2) API key
    key_answer = (
        await repl.read_secret(
            f"API key（隐藏输入且不记历史；回车跳过，改用环境变量 {entry.api_key_env}）> "
        )
    ).strip()
    api_key = key_answer or None
    if api_key:
        console.print("[dim]已记录 key；将写入用户配置，请注意文件保管。[/dim]")

    # 3) 模型名
    examples = ", ".join(entry.example_models)
    models_answer = (
        await repl.read_line(f"模型名，逗号分隔（示例: {examples}；回车跳过）> ")
    ).strip()
    models = [
        m.strip() for m in models_answer.replace("，", ",").split(",") if m.strip()
    ]

    pc = ProviderConfig(
        api_key_env=entry.api_key_env,
        base_url=base_url,
        lang=entry.lang,
        protocol=protocol,
        api_key=api_key,
    )
    return pc, models
