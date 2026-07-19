"""配置加载：内置默认值 ← ~/.emiyacode/config.toml ← 项目/.emiyacode/config.toml 逐层叠加。

API key 一律从环境变量读取，配置里只存环境变量名。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import tomlkit
from pydantic import BaseModel, Field

from emiya_agent.errors import ConfigError

CONFIG_DIR_NAME = ".emiyacode"
CONFIG_FILE_NAME = "config.toml"


class ProviderConfig(BaseModel):
    api_key_env: str
    base_url: str | None = None
    lang: str = "en"  # 语言基座：zh（中文原生）/ en。决定翻译管线是否介入
    protocol: str = "openai"  # 调用协议：openai 兼容 / anthropic 兼容
    api_key: str | None = None  # 明文 key（可选；环境变量优先。注意配置文件权限）


class ModelInfo(BaseModel):
    context_window: int = 200_000
    input_price: float = 0.0  # USD / 1M tokens
    output_price: float = 0.0
    cache_read_price: float = 0.0
    lang: str | None = None  # 覆盖 provider 级语言基座（聚合网关下有用）


class MCPServerConfig(BaseModel):
    """stdio 服务器给 command/args/env；HTTP 服务器给 url。"""

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None


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
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming"))) / "Anthropic"
    else:
        base = Path.home() / ".config" / "anthropic"
    cred_dir = base / "credentials"
    if cred_dir.is_dir() and any(cred_dir.glob("*.json")):
        return f"OAuth profile ({cred_dir})"
    return None


DEFAULTS: dict[str, Any] = {
    "language": "zh",  # zh: 启用翻译管线；en: 直通
    "max_tokens": 16000,
    "effort": "high",
    "tool_approval": "ask",  # ask: 写操作/shell/MCP 执行前须确认；auto: 直接执行
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
    language: str = "zh"
    max_tokens: int = 16000
    effort: str = "high"
    tool_approval: str = "ask"  # ask | auto
    load_warnings: list[str] = Field(default_factory=list)  # 配置文件解析警告
    routes: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelInfo] = Field(default_factory=dict)
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
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
        """路由的语言基座：model 级覆盖 → provider 级默认 → en。"""
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

    def project_dir(self) -> Path:
        """项目级 .emiyacode 目录（按需创建）。"""
        d = self.cwd / CONFIG_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d

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
# lang = "zh"                      # 语言基座：zh 模型作执行器时免翻译
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
    """叠加加载。某层配置文件损坏时跳过该层并记录警告，不阻断启动。"""
    cwd = (cwd or Path.cwd()).resolve()
    merged = DEFAULTS
    warnings: list[str] = []
    for path in (
        Path.home() / CONFIG_DIR_NAME / CONFIG_FILE_NAME,
        cwd / CONFIG_DIR_NAME / CONFIG_FILE_NAME,
    ):
        try:
            overlay = _read_toml(path)
        except ConfigError as exc:
            warnings.append(f"{exc}（该文件已被忽略）")
            continue
        if overlay:
            merged = deep_merge(merged, overlay)
    config = Config(cwd=cwd, **merged)
    config.load_warnings = warnings
    return config
