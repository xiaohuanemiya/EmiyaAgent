"""脱敏诊断报告：/doctor 与 `emiya doctor` 共用。

原则：
- 绝不输出 API key、token 或任何凭据值；凭据只报告"来源"（环境变量名等）。
- URL 去除 userinfo/query/fragment；hook/MCP 命令只保留可执行文件 basename。
- 只读：不创建目录、不连接网络、不启动子进程。
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from emiya_agent import __version__
from emiya_agent.config import (
    CONFIG_DIR_NAME,
    Config,
    project_config_path,
    user_config_path,
)

DIAGNOSTICS_SCHEMA_VERSION = 1

CheckStatus = str  # "ok" | "warn" | "fail"


def _sanitize_url(url: str) -> str:
    """只保留 HTTP(S) origin，不回显可承载凭据的 URL 部分。"""
    try:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme not in {"http", "https"}:
            return "(invalid URL)"
        host = parts.hostname
        port = parts.port  # 非法/越界端口会在此抛 ValueError。
        if not host or any(ord(char) < 32 for char in host):
            return "(invalid URL)"
        if ":" in host:  # urlsplit().hostname 会去掉 IPv6 方括号。
            safe_host = f"[{host}]"
        else:
            safe_host = host.encode("idna").decode("ascii")
    except (AttributeError, TypeError, UnicodeError, ValueError):
        return "(invalid URL)"

    authority = safe_host if port is None else f"{safe_host}:{port}"
    # path 同样常用于 bearer token/租户密钥，无法靠启发式可靠
    # 判断哪个 segment 安全；诊断中因此整体省略。
    return f"{scheme}://{authority}"


def _sanitize_load_warning(warning: Any) -> str:
    """把内部 warning 映射成有限、不回显原文的诊断类别。"""

    if isinstance(warning, str):
        if warning.startswith("配置文件解析失败"):
            return "配置文件解析失败；该配置层已忽略"
        if warning.startswith("未信任项目配置"):
            return "存在未信任的项目配置；该配置层已忽略"
        if warning.startswith("项目配置中的用户级安全策略已忽略"):
            return "项目配置尝试覆盖用户级安全策略；相关字段已忽略"
    return "配置加载警告（详细内容已脱敏）"


def _check(checks: list[dict[str, Any]], check_id: str, status: CheckStatus, detail: str) -> None:
    checks.append({"id": check_id, "status": status, "detail": detail})


def build_report(
    config: Config,
    *,
    mcp_status: dict[str, dict[str, Any]] | None = None,
    lsp_connections: int | None = None,
) -> dict[str, Any]:
    """构建一份可安全分享的诊断报告（dict，可直接 JSON 序列化）。

    ``mcp_status`` / ``lsp_connections`` 为 REPL 运行态可选信息；
    CLI 冷启动时为 None，报告仅含静态配置检查。
    """
    checks: list[dict[str, Any]] = []
    safe_load_warnings = [
        _sanitize_load_warning(warning) for warning in config.load_warnings
    ]

    user_path = user_config_path()
    project_path = project_config_path(config.cwd)

    if config.load_warnings:
        _check(
            checks,
            "config_load",
            "warn",
            f"{len(config.load_warnings)} 条配置加载警告（详见 config.load_warnings）",
        )
    else:
        _check(checks, "config_load", "ok", "配置加载无警告")

    if config.project_config_ignored:
        _check(
            checks,
            "project_trust",
            "warn",
            "存在未信任的项目配置，已整体忽略；确认仓库可信后运行 /trust",
        )
    else:
        _check(checks, "project_trust", "ok", "项目配置信任状态正常")

    providers: list[dict[str, Any]] = []
    for name, pc in config.providers.items():
        source = config.credential_source(name)
        providers.append(
            {
                "name": name,
                "protocol": pc.protocol,
                "lang": pc.lang,
                "base_url": _sanitize_url(pc.base_url) if pc.base_url else None,
                "credential_source": source,
                "available": source is not None,
            }
        )

    routes: dict[str, dict[str, Any]] = {}
    missing_provider_roles: list[str] = []
    unavailable_roles: list[str] = []
    for role, route in config.routes.items():
        provider_name = route.split("/", 1)[0]
        provider_defined = provider_name in config.providers
        credential = (
            provider_defined and config.credential_source(provider_name) is not None
        )
        info = config.model_info(route)
        has_pricing = bool(info.input_price or info.output_price)
        routes[role] = {
            "route": route,
            "provider_defined": provider_defined,
            "credential": credential,
            "pricing": has_pricing,
            "context_window": info.context_window,
        }
        if not provider_defined:
            missing_provider_roles.append(role)
        elif not credential:
            unavailable_roles.append(role)

    if missing_provider_roles:
        _check(
            checks,
            "route_providers",
            "fail",
            "以下角色路由指向未定义的 provider: " + ", ".join(missing_provider_roles),
        )
    elif unavailable_roles:
        severity = "fail" if "executor" in unavailable_roles else "warn"
        _check(
            checks,
            "route_providers",
            severity,
            "以下角色缺少可用凭据: " + ", ".join(unavailable_roles),
        )
    else:
        _check(checks, "route_providers", "ok", "所有角色路由的 provider 与凭据可用")

    executor_route = config.routes.get("executor")
    if executor_route is not None and not routes.get("executor", {}).get("pricing"):
        _check(
            checks,
            "executor_pricing",
            "warn",
            f"执行器路由 {executor_route} 未配置单价，--max-budget-usd 将不可用",
        )
    else:
        _check(checks, "executor_pricing", "ok", "执行器路由已配置单价")

    hooks = [
        {
            "id": hook.id,
            "event": hook.event.value,
            "on_error": hook.on_error,
            "timeout": hook.timeout,
            "executable": Path(hook.command[0]).name,
        }
        for hook in config.hooks
    ]

    mcp_servers: list[dict[str, Any]] = []
    for name, server in config.mcp_servers.items():
        if server.url:
            kind = "http"
            target = _sanitize_url(server.url)
        elif server.command:
            kind = "stdio"
            target = Path(server.command).name
        else:
            kind = "invalid"
            target = ""
        entry: dict[str, Any] = {"name": name, "kind": kind, "target": target}
        if mcp_status is not None and name in mcp_status:
            entry["connected"] = bool(mcp_status[name].get("connected"))
            entry["tools"] = mcp_status[name].get("tools", 0)
        mcp_servers.append(entry)
    invalid_mcp = [entry["name"] for entry in mcp_servers if entry["kind"] == "invalid"]
    if invalid_mcp:
        _check(
            checks,
            "mcp_config",
            "fail",
            "以下 MCP 服务器既无 command 也无 url: " + ", ".join(invalid_mcp),
        )
    elif mcp_servers:
        disconnected = [
            entry["name"]
            for entry in mcp_servers
            if entry.get("connected") is False
        ]
        if disconnected:
            _check(
                checks,
                "mcp_config",
                "warn",
                "以下 MCP 服务器未连接: " + ", ".join(disconnected),
            )
        else:
            _check(checks, "mcp_config", "ok", f"{len(mcp_servers)} 个 MCP 服务器配置有效")
    else:
        _check(checks, "mcp_config", "ok", "未配置 MCP 服务器")

    lsp_servers: list[dict[str, Any]] = []
    for name, server in sorted(
        config.lsp_servers.items(), key=lambda item: (-item[1].priority, item[0])
    ):
        found = shutil.which(server.command) is not None
        lsp_servers.append(
            {
                "name": name,
                "command": Path(server.command).name,
                "found_on_path": found,
                "disabled": server.disabled,
                "priority": server.priority,
            }
        )
    if config.lsp_enabled:
        usable = [
            entry
            for entry in lsp_servers
            if entry["found_on_path"] and not entry["disabled"]
        ]
        if usable:
            _check(
                checks,
                "lsp",
                "ok",
                f"LSP 已启用，{len(usable)} 个 server 可在 PATH 上找到",
            )
        else:
            _check(
                checks,
                "lsp",
                "warn",
                "LSP 已启用但 PATH 上找不到任何已配置 server"
                "（EmiyaAgent 绝不自动下载，请自行安装）",
            )
    else:
        _check(checks, "lsp", "ok", "LSP 已禁用（默认）")

    sessions_dir = config.cwd / CONFIG_DIR_NAME / "sessions"
    session_count: int | None = None
    sessions_error: str | None = None
    if sessions_dir.is_dir():
        try:
            session_count = sum(
                1 for p in sessions_dir.iterdir() if p.suffix == ".jsonl"
            )
        except OSError as exc:
            sessions_error = f"{type(exc).__name__}: {exc}"
    if sessions_error:
        _check(checks, "sessions", "warn", f"会话目录不可读: {sessions_error}")
    else:
        _check(
            checks,
            "sessions",
            "ok",
            "尚无会话目录"
            if session_count is None
            else f"{session_count} 个会话文件",
        )

    worst = "ok"
    if any(check["status"] == "warn" for check in checks):
        worst = "warn"
    if any(check["status"] == "fail" for check in checks):
        worst = "fail"

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": worst,
        "app": {
            "version": __version__,
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "config": {
            "user_config_present": user_path.is_file(),
            "project_config_present": project_path.is_file(),
            "project_trusted": config.project_trusted,
            "project_config_ignored": config.project_config_ignored,
            "load_warnings": safe_load_warnings,
            "tool_approval": config.tool_approval,
            "allow_external_paths": config.allow_external_paths,
            "max_tokens": config.max_tokens,
            "effort": config.effort,
            "shell_rules": len(config.shell_rules),
            "env_allowlists": {
                "shell": len(config.shell_inherit_env),
                "mcp": len(config.mcp_inherit_env),
                "hook": len(config.hook_inherit_env),
                "lsp": len(config.lsp_inherit_env),
            },
        },
        "providers": providers,
        "routes": routes,
        "hooks": hooks,
        "mcp_servers": mcp_servers,
        "lsp": {
            "enabled": config.lsp_enabled,
            "connections": lsp_connections,
            "servers": lsp_servers,
        },
        "sessions": {
            "dir": str(sessions_dir),
            "count": session_count,
            "error": sessions_error,
        },
        "checks": checks,
    }


_STATUS_MARKS = {"ok": "✓", "warn": "⚠", "fail": "✗"}


def render_report(report: dict[str, Any]) -> str:
    """把诊断报告渲染为无标记纯文本（REPL 与 CLI 共用）。"""
    lines: list[str] = []
    app = report["app"]
    lines.append(
        f"EmiyaAgent {app['version']} · Python {app['python']} · {app['platform']}"
        f" · 总体状态: {report['status']}"
    )
    cfg = report["config"]
    lines.append(
        f"配置: 用户级={'有' if cfg['user_config_present'] else '无'}"
        f" · 项目级={'有' if cfg['project_config_present'] else '无'}"
        f"（{'已信任' if cfg['project_trusted'] else '未信任'}）"
        f" · approval={cfg['tool_approval']}"
        f" · 外部路径={'允许' if cfg['allow_external_paths'] else '禁止'}"
    )
    for warning in cfg["load_warnings"]:
        lines.append(f"  ⚠ {warning}")
    lines.append("Provider:")
    for provider in report["providers"]:
        mark = "✓" if provider["available"] else "✗"
        source = provider["credential_source"] or "无可用凭据"
        base = provider["base_url"] or "(官方默认)"
        lines.append(
            f"  {mark} {provider['name']} [{provider['protocol']}/"
            f"{provider['lang']}] {base} · 凭据: {source}"
        )
    lines.append("角色路由:")
    for role, info in report["routes"].items():
        mark = "✓" if info["provider_defined"] and info["credential"] else "✗"
        extras: list[str] = [f"窗口 {info['context_window']}"]
        if not info["pricing"]:
            extras.append("无单价")
        lines.append(f"  {mark} {role} → {info['route']}（{'，'.join(extras)}）")
    if report["hooks"]:
        events = ", ".join(
            f"{hook['id']}@{hook['event']}" for hook in report["hooks"]
        )
        lines.append(f"Hook: {len(report['hooks'])} 个（{events}）")
    else:
        lines.append("Hook: 未配置")
    if report["mcp_servers"]:
        for entry in report["mcp_servers"]:
            state = ""
            if "connected" in entry:
                state = " · 已连接" if entry["connected"] else " · 未连接"
            lines.append(
                f"MCP: {entry['name']} [{entry['kind']}] {entry['target']}{state}"
            )
    else:
        lines.append("MCP: 未配置")
    lsp = report["lsp"]
    if lsp["enabled"]:
        found = [s["name"] for s in lsp["servers"] if s["found_on_path"]]
        lines.append(
            f"LSP: 已启用 · PATH 可用: {', '.join(found) if found else '（无）'}"
        )
    else:
        lines.append("LSP: 已禁用（默认）")
    sessions = report["sessions"]
    count = "无" if sessions["count"] is None else str(sessions["count"])
    lines.append(f"会话: {count}（{sessions['dir']}）")
    lines.append("检查:")
    for check in report["checks"]:
        mark = _STATUS_MARKS.get(check["status"], "?")
        lines.append(f"  {mark} {check['id']}: {check['detail']}")
    return "\n".join(lines)
