"""配置加载：内置默认值 ← ~/.emiyacode/config.toml ← 项目/.emiyacode/config.toml 逐层叠加。

API key 默认从环境变量读取；初始化向导也可选择写入用户配置。
"""

from __future__ import annotations

import os
import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from emiya_agent.errors import ConfigError
from emiya_agent.hooks.models import HookCommandConfig
from emiya_agent.safe_paths import is_linklike

CONFIG_DIR_NAME = ".emiyacode"
CONFIG_FILE_NAME = "config.toml"
DEFAULT_SHELL_ENV_ALLOWLIST = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "USER",
    "LOGNAME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES*",
    "LANG",
    "LC_*",
    "TERM",
    "COLORTERM",
    "NO_COLOR",
    "VIRTUAL_ENV",
)
DEFAULT_MCP_ENV_ALLOWLIST = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "USER",
    "LOGNAME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "LANG",
    "LC_*",
)
DEFAULT_HOOK_ENV_ALLOWLIST = DEFAULT_SHELL_ENV_ALLOWLIST
DEFAULT_LSP_ENV_ALLOWLIST = DEFAULT_SHELL_ENV_ALLOWLIST
USER_ONLY_CONFIG_KEYS = frozenset(
    {
        "trusted_projects",
        "allow_external_paths",
        "tool_approval",
        "shell_rules",
        "shell_inherit_env",
        "mcp_inherit_env",
        "mcp_server_inherit_env",
        "hooks",
        "hook_inherit_env",
        "lsp_enabled",
        "lsp_inherit_env",
        "cwd",
        "project_trusted",
        "project_config_ignored",
    }
)


def select_environment(
    patterns: list[str] | tuple[str, ...],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """按用户级白名单继承环境变量；Windows 下变量名不区分大小写。"""
    selected: dict[str, str] = {}
    normalized_patterns = [
        pattern.upper() if os.name == "nt" else pattern for pattern in patterns
    ]
    for name, value in os.environ.items():
        comparable = name.upper() if os.name == "nt" else name
        if any(fnmatchcase(comparable, pattern) for pattern in normalized_patterns):
            selected[name] = value
    if overrides:
        selected.update(overrides)
    return selected


def _ensure_workspace_directory(path: Path, *, workspace: Path, label: str) -> Path:
    """创建并验证项目运行时目录，拒绝链接、junction 与工作区逃逸。"""
    lexical_workspace = workspace.absolute()
    lexical_path = path.absolute()
    try:
        relative = lexical_path.relative_to(lexical_workspace)
    except ValueError:
        raise ConfigError(f"{label} 超出工作区: {lexical_path}") from None

    try:
        # mkdir 前后都逐层检查；既防已有链接，也防创建时穿过链接父目录。
        for _ in range(2):
            current = lexical_workspace
            for part in relative.parts:
                current /= part
                if is_linklike(current):
                    raise ConfigError(f"{label} 不能使用符号链接或 junction: {current}")
            lexical_path.mkdir(parents=True, exist_ok=True)

        if not lexical_path.is_dir():
            raise ConfigError(f"{label} 不是目录: {lexical_path}")
        resolved_workspace = lexical_workspace.resolve(strict=True)
        resolved_path = lexical_path.resolve(strict=True)
        resolved_path.relative_to(resolved_workspace)
    except ConfigError:
        raise
    except ValueError:
        raise ConfigError(f"{label} 解析后超出工作区: {lexical_path}") from None
    except OSError as exc:
        raise ConfigError(f"无法安全创建 {label} {lexical_path}: {exc}") from exc
    return lexical_path


class ProviderConfig(BaseModel):
    api_key_env: str
    base_url: str | None = None
    lang: Literal["zh", "en"] = "en"  # 该 provider 模型的默认主要语言
    protocol: Literal["openai", "anthropic"] = "openai"
    api_key: str | None = None  # 明文 key（可选；环境变量优先。注意配置文件权限）


class ModelInfo(BaseModel):
    context_window: int = 200_000
    input_price: float = 0.0  # USD / 1M tokens
    output_price: float = 0.0
    cache_read_price: float = 0.0
    lang: Literal["zh", "en"] | None = None  # 覆盖 provider 级主要语言


class MCPServerConfig(BaseModel):
    """stdio 服务器给 command/args/env；HTTP 服务器给 url。"""

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None


class LSPServerConfig(BaseModel):
    """一个由用户自行安装的 stdio language server。"""

    command: str
    args: list[str] = Field(default_factory=list)
    extensions: dict[str, str]
    root_markers: list[str] = Field(default_factory=lambda: [".git"])
    env: dict[str, str] = Field(default_factory=dict)
    initialization_options: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    startup_timeout: float = Field(default=15.0, gt=0, le=120)
    request_timeout: float = Field(default=10.0, gt=0, le=120)
    shutdown_timeout: float = Field(default=3.0, gt=0, le=30)
    priority: int = 0
    disabled: bool = False

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("LSP command 不能为空或包含 NUL")
        return value

    @field_validator("args")
    @classmethod
    def validate_args(cls, value: list[str]) -> list[str]:
        if any("\x00" in argument for argument in value):
            raise ValueError("LSP args 不能包含 NUL")
        return value

    @field_validator("extensions")
    @classmethod
    def validate_extensions(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("LSP extensions 不能为空")
        normalized: dict[str, str] = {}
        for extension, language in value.items():
            extension = extension.lower()
            if not extension.startswith(".") or any(
                char in extension for char in ("/", "\\", "\x00")
            ):
                raise ValueError(f"非法 LSP 扩展名: {extension!r}")
            if not language.strip():
                raise ValueError("LSP language id 不能为空")
            normalized[extension] = language.strip()
        return normalized

    @field_validator("root_markers")
    @classmethod
    def validate_root_markers(cls, value: list[str]) -> list[str]:
        for marker in value:
            path = Path(marker)
            if (
                not marker
                or "\x00" in marker
                or path.is_absolute()
                or ".." in path.parts
            ):
                raise ValueError(f"非法 LSP root marker: {marker!r}")
        return value


class ShellPermissionRule(BaseModel):
    """对完整 shell command 做 full-match 的用户级权限规则。"""

    action: Literal["allow", "ask", "deny"]
    pattern: str
    description: str = ""

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        if not value or len(value) > 4096:
            raise ValueError("shell rule pattern 必须为 1..4096 字符")
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"非法正则表达式: {exc}") from exc
        return value


def anthropic_fallback_credentials() -> str | None:
    """Anthropic SDK 凭据链中非 API key 的可用凭据（描述字符串），无则 None。

    解析顺序与官方 SDK 一致：ANTHROPIC_AUTH_TOKEN → ant auth login 的 OAuth profile。
    """
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "ANTHROPIC_AUTH_TOKEN"
    config_dir = os.environ.get("ANTHROPIC_CONFIG_DIR")
    if config_dir:
        base = Path(config_dir)
    elif sys.platform == "win32":
        base = (
            Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming")))
            / "Anthropic"
        )
    else:
        base = Path.home() / ".config" / "anthropic"
    cred_dir = base / "credentials"
    if cred_dir.is_dir() and any(cred_dir.glob("*.json")):
        return f"OAuth profile ({cred_dir})"
    return None


DEFAULTS: dict[str, Any] = {
    "max_tokens": 16000,
    "effort": "high",
    "tool_approval": "ask",  # ask: 写操作/shell/MCP 执行前须确认；auto: 直接执行
    # shell 权限规则只从用户级配置读取，按顺序对完整命令做 re.fullmatch。
    "shell_rules": [],
    # 子进程默认不继承 provider/cloud token；确有需要时由用户级配置显式追加。
    "shell_inherit_env": list(DEFAULT_SHELL_ENV_ALLOWLIST),
    "mcp_inherit_env": list(DEFAULT_MCP_ENV_ALLOWLIST),
    "mcp_server_inherit_env": {},
    # 命令式 Hook 只来自用户配置；项目脚本可变，路径信任不能替代定义审核。
    "hooks": [],
    "hook_inherit_env": list(DEFAULT_HOOK_ENV_ALLOWLIST),
    "lsp_enabled": False,
    "lsp_inherit_env": list(DEFAULT_LSP_ENV_ALLOWLIST),
    "lsp_servers": {
        "basedpyright": {
            "command": "basedpyright-langserver",
            "args": ["--stdio"],
            "extensions": {".py": "python", ".pyi": "python"},
            "root_markers": ["pyrightconfig.json", "pyproject.toml", ".git"],
            "priority": 100,
        },
        "pyright": {
            "command": "pyright-langserver",
            "args": ["--stdio"],
            "extensions": {".py": "python", ".pyi": "python"},
            "root_markers": ["pyrightconfig.json", "pyproject.toml", ".git"],
            "priority": 90,
        },
        "typescript": {
            "command": "typescript-language-server",
            "args": ["--stdio"],
            "extensions": {
                ".ts": "typescript",
                ".tsx": "typescriptreact",
                ".js": "javascript",
                ".jsx": "javascriptreact",
            },
            "root_markers": ["tsconfig.json", "jsconfig.json", "package.json", ".git"],
            "priority": 80,
        },
        "rust": {
            "command": "rust-analyzer",
            "extensions": {".rs": "rust"},
            "root_markers": ["Cargo.toml", ".git"],
            "priority": 80,
        },
        "go": {
            "command": "gopls",
            "extensions": {".go": "go"},
            "root_markers": ["go.work", "go.mod", ".git"],
            "priority": 80,
        },
        "clangd": {
            "command": "clangd",
            "extensions": {
                ".c": "c",
                ".h": "c",
                ".cc": "cpp",
                ".cpp": "cpp",
                ".cxx": "cpp",
                ".hpp": "cpp",
            },
            "root_markers": ["compile_commands.json", "compile_flags.txt", ".git"],
            "priority": 70,
        },
    },
    # 仅从用户级配置读取。项目配置在显式信任前不会生效，避免仓库一打开
    # 就通过 providers/base_url 或 stdio MCP 命令窃取凭据/执行代码。
    "trusted_projects": [],
    # 文件工具默认只能访问工作区；确需跨目录时只能在用户级配置中开启。
    "allow_external_paths": False,
    "routes": {
        "executor": "anthropic/claude-fable-5",
        "translator": "deepseek/deepseek-chat",
        "compactor": "deepseek/deepseek-chat",
        "subagent_explore": "deepseek/deepseek-chat",
        "subagent_plan": "anthropic/claude-opus-4-8",
        "subagent_general": "anthropic/claude-opus-4-8",
    },
    "providers": {
        "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "protocol": "anthropic"},
        "openai": {"api_key_env": "OPENAI_API_KEY"},
        "deepseek": {
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com",
            "lang": "zh",
        },
        "glm": {
            "api_key_env": "GLM_API_KEY",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "lang": "zh",
        },
    },
    "models": {
        "anthropic/claude-fable-5": {
            "context_window": 1_000_000,
            "input_price": 10.0,
            "output_price": 50.0,
            "cache_read_price": 1.0,
        },
        "anthropic/claude-opus-4-8": {
            "context_window": 1_000_000,
            "input_price": 5.0,
            "output_price": 25.0,
            "cache_read_price": 0.5,
        },
        "openai/gpt-5.6": {
            "context_window": 400_000,
        },
        "deepseek/deepseek-chat": {
            "context_window": 1_000_000,
        },
        "glm/glm-4.6": {
            "context_window": 200_000,
        },
    },
}


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 16000
    effort: str = "high"
    tool_approval: Literal["ask", "auto"] = "ask"
    shell_rules: list[ShellPermissionRule] = Field(default_factory=list)
    shell_inherit_env: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SHELL_ENV_ALLOWLIST)
    )
    mcp_inherit_env: list[str] = Field(
        default_factory=lambda: list(DEFAULT_MCP_ENV_ALLOWLIST)
    )
    mcp_server_inherit_env: dict[str, list[str]] = Field(default_factory=dict)
    hooks: list[HookCommandConfig] = Field(default_factory=list)
    hook_inherit_env: list[str] = Field(
        default_factory=lambda: list(DEFAULT_HOOK_ENV_ALLOWLIST)
    )
    lsp_enabled: bool = False
    lsp_inherit_env: list[str] = Field(
        default_factory=lambda: list(DEFAULT_LSP_ENV_ALLOWLIST)
    )
    trusted_projects: list[str] = Field(default_factory=list)
    allow_external_paths: bool = False
    project_trusted: bool = False
    project_config_ignored: bool = False
    load_warnings: list[str] = Field(default_factory=list)  # 配置文件解析警告
    routes: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelInfo] = Field(default_factory=dict)
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    lsp_servers: dict[str, LSPServerConfig] = Field(default_factory=dict)
    cwd: Path = Field(default_factory=Path.cwd)

    def api_key(self, provider: str) -> str | None:
        """key 解析：环境变量优先，其次配置文件明文 api_key。"""
        pc = self.providers.get(provider)
        if pc is None:
            return None
        return os.environ.get(pc.api_key_env) or pc.api_key or None

    def credential_source(self, provider: str) -> str | None:
        """该 provider 实际可用的凭据来源描述；不可用返回 None。"""
        pc = self.providers.get(provider)
        if pc is None:
            return None
        if os.environ.get(pc.api_key_env):
            return pc.api_key_env
        if pc.api_key:
            return "config.toml"
        if provider == "anthropic":
            return anthropic_fallback_credentials()
        return None

    def key_status(self) -> dict[str, bool]:
        return {
            name: self.credential_source(name) is not None for name in self.providers
        }

    def model_info(self, route: str) -> ModelInfo:
        return self.models.get(route, ModelInfo())

    def model_lang(self, route: str) -> str:
        """模型主要语言：model 级覆盖 → provider 级默认 → en。"""
        info = self.models.get(route)
        if info is not None and info.lang:
            return info.lang
        provider_name = route.split("/", 1)[0]
        pc = self.providers.get(provider_name)
        return pc.lang if pc is not None else "en"

    def route(self, role: str) -> str:
        try:
            return self.routes[role]
        except KeyError:
            raise ConfigError(f"未配置模型路由: routes.{role}") from None

    def shell_permission(self, command: str) -> Literal["allow", "ask", "deny"] | None:
        """返回首条匹配的命令规则；None 表示回退到全局 tool_approval。"""
        command = command.strip()
        for rule in self.shell_rules:
            if re.fullmatch(rule.pattern, command):
                return rule.action
        return None

    def project_dir(self) -> Path:
        """项目级 .emiyacode 目录（按需创建，且不得为链接/junction）。"""
        return _ensure_workspace_directory(
            self.cwd / CONFIG_DIR_NAME,
            workspace=self.cwd,
            label="项目运行时目录",
        )

    def project_runtime_dir(self, name: str) -> Path:
        """返回安全的运行时子目录（memory/sessions/logs 等）。"""
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ConfigError(f"非法运行时目录名: {name!r}")
        project_dir = self.project_dir()
        return _ensure_workspace_directory(
            project_dir / name,
            workspace=self.cwd,
            label=f"项目运行时目录 {name}",
        )

    @staticmethod
    def user_dir() -> Path:
        d = Path.home() / CONFIG_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """递归合并：overlay 中的字典与 base 同键字典合并，其余直接覆盖。"""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        # utf-8-sig: Windows 记事本/PowerShell 写的 UTF-8 常带 BOM
        return dict(tomlkit.parse(path.read_text(encoding="utf-8-sig")))
    except Exception as exc:  # tomlkit 解析错误
        raise ConfigError(f"配置文件解析失败 {path}: {exc}") from exc


def user_config_path() -> Path:
    return Path.home() / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def project_config_path(cwd: Path) -> Path:
    return cwd / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def config_files_exist(cwd: Path) -> bool:
    """用户级或项目级配置文件任一存在即视为已初始化。"""
    return user_config_path().is_file() or project_config_path(cwd).is_file()


def is_project_trusted(cwd: Path, trusted_projects: list[str]) -> bool:
    """用户级白名单是否信任 cwd（也信任白名单目录的子目录）。"""
    cwd = cwd.resolve()
    for raw in trusted_projects:
        try:
            trusted = Path(raw).expanduser().resolve()
            cwd.relative_to(trusted)
        except (OSError, RuntimeError, ValueError):
            continue
        return True
    return False


def persist_trusted_project(path: Path, cwd: Path) -> None:
    """把项目绝对路径写入用户配置的 trusted_projects 白名单。"""
    doc = _load_or_new_doc(path)
    current = doc.get("trusted_projects")
    try:
        values = [str(v) for v in current] if current is not None else []
    except TypeError:
        values = []
    resolved = str(cwd.resolve())
    if resolved not in values:
        values.append(resolved)
    doc["trusted_projects"] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8", newline="\n")
    _restrict_secret_file(path)


def _restrict_secret_file(path: Path) -> None:
    """尽力把包含明文凭据的配置限制为仅当前 POSIX 用户可读写。"""
    if os.name == "posix":
        try:
            path.chmod(0o600)
        except OSError:
            pass


def persist_provider(path: Path, name: str, provider: ProviderConfig) -> None:
    """把 provider 定义写回配置文件。

    tomlkit 往返式读写：已有内容、注释、排版全部保留，只增改目标条目。
    文件不存在时创建带说明注释的新文件。
    """
    if path.is_file():
        doc = tomlkit.parse(path.read_text(encoding="utf-8-sig"))
    else:
        doc = tomlkit.document()
        doc.add(tomlkit.comment("EmiyaAgent 配置（由 /provider 命令生成，可手工编辑）"))
        doc.add(tomlkit.nl())
    providers = doc.get("providers")
    if providers is None:
        providers = tomlkit.table(True)  # super table：直接渲染 [providers.<name>]
        doc["providers"] = providers
    entry = tomlkit.table()
    entry["api_key_env"] = provider.api_key_env
    if provider.base_url:
        entry["base_url"] = provider.base_url
    if provider.lang != "en":
        entry["lang"] = provider.lang
    if provider.protocol != "openai":
        entry["protocol"] = provider.protocol
    if provider.api_key:
        entry["api_key"] = provider.api_key
    providers[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8", newline="\n")
    if provider.api_key:
        _restrict_secret_file(path)


def _load_or_new_doc(path: Path) -> "tomlkit.TOMLDocument":
    if path.is_file():
        return tomlkit.parse(path.read_text(encoding="utf-8-sig"))
    doc = tomlkit.document()
    doc.add(tomlkit.comment("EmiyaAgent 配置（可手工编辑；/init 可重新运行向导）"))
    doc.add(tomlkit.nl())
    return doc


def persist_routes(path: Path, routes: dict[str, str]) -> None:
    """写回角色路由（只覆盖给出的角色，其余保留）。"""
    doc = _load_or_new_doc(path)
    tbl = doc.get("routes")
    if tbl is None:
        tbl = tomlkit.table()
        doc["routes"] = tbl
    for role, route in routes.items():
        tbl[role] = route
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8", newline="\n")


def persist_models(path: Path, routes: list[str]) -> None:
    """登记模型路由（空元数据表，窗口/单价用默认值，可事后补）。"""
    doc = _load_or_new_doc(path)
    models = doc.get("models")
    if models is None:
        models = tomlkit.table(True)
        doc["models"] = models
    for route in routes:
        if route not in models:
            models[route] = tomlkit.table()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8", newline="\n")


def write_config_template(path: Path) -> None:
    """写入带注释的空白配置（用户跳过向导时，避免每次启动重复提醒）。"""
    if path.is_file():
        return
    template = """\
# EmiyaAgent 配置。运行 /init 可随时启动初始化向导。
#
# [providers.某家]                 # 供应商定义
# api_key_env = "XXX_API_KEY"      # key 的环境变量名
# base_url = "https://.../v1"      # OpenAI 兼容端点；anthropic 协议端点见 protocol
# protocol = "openai"              # openai | anthropic
# lang = "zh"                      # 模型主要语言：zh | en
# api_key = "sk-..."               # 可选：明文 key（环境变量优先）
#
# [routes]                         # 各智能体角色使用的模型
# executor = "deepseek/deepseek-v4-pro"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8", newline="\n")


def remove_provider_from_file(path: Path, name: str) -> bool:
    """从配置文件删除 provider 条目；删除了返回 True。"""
    if not path.is_file():
        return False
    doc = tomlkit.parse(path.read_text(encoding="utf-8-sig"))
    providers = doc.get("providers")
    if providers is None or name not in providers:
        return False
    del providers[name]
    path.write_text(tomlkit.dumps(doc), encoding="utf-8", newline="\n")
    return True


def load_config(cwd: Path | None = None) -> Config:
    """叠加加载，并对项目级可执行配置实施显式信任门控。

    用户配置始终生效。项目配置可声明 provider 端点和 stdio MCP 命令，
    因此只有 cwd 被用户配置的 trusted_projects 覆盖时才整体加载。
    """
    cwd = (cwd or Path.cwd()).resolve()
    merged = DEFAULTS
    warnings: list[str] = []

    user_path = user_config_path()
    try:
        user_overlay = _read_toml(user_path)
    except ConfigError as exc:
        warnings.append(f"{exc}（该文件已被忽略）")
        user_overlay = {}
    if user_overlay:
        merged = deep_merge(merged, user_overlay)

    raw_trusted = merged.get("trusted_projects", [])
    try:
        trusted_projects = [str(v) for v in raw_trusted]
    except TypeError:
        trusted_projects = []
    project_trusted = is_project_trusted(cwd, trusted_projects)
    project_ignored = False
    project_path = project_config_path(cwd)
    try:
        project_overlay = _read_toml(project_path)
    except ConfigError as exc:
        warnings.append(f"{exc}（该文件已被忽略）")
        project_overlay = {}
    if project_overlay and project_trusted:
        # 信任允许项目提供模型端点与 MCP，但权限边界始终由用户层掌控。
        ignored_policy_keys = USER_ONLY_CONFIG_KEYS.intersection(project_overlay)
        project_overlay = {
            key: value
            for key, value in project_overlay.items()
            if key not in USER_ONLY_CONFIG_KEYS
        }
        if ignored_policy_keys:
            warnings.append(
                "项目配置中的用户级安全策略已忽略: "
                + ", ".join(sorted(ignored_policy_keys))
            )
        merged = deep_merge(merged, project_overlay)
    elif project_overlay:
        project_ignored = True
        warnings.append(
            f"未信任项目配置 {project_path}，已全部忽略；"
            "确认仓库可信后运行 /trust 并重启"
        )

    # 运行时字段不能由任何配置文件伪造，也避免与显式关键字重复。
    merged.pop("cwd", None)
    merged.pop("project_trusted", None)
    merged.pop("project_config_ignored", None)
    try:
        config = Config(
            cwd=cwd,
            project_trusted=project_trusted,
            project_config_ignored=project_ignored,
            **merged,
        )
    except ValidationError as exc:
        raise ConfigError(f"配置值校验失败: {exc}") from exc
    config.load_warnings = warnings
    return config
