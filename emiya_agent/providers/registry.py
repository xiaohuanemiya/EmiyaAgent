"""路由字符串 "provider/model" → 已配置的 Provider 实例（按 provider 缓存）。"""

from __future__ import annotations

from emiya_agent.config import Config, anthropic_fallback_credentials
from emiya_agent.errors import ConfigError
from emiya_agent.providers.anthropic_provider import AnthropicProvider
from emiya_agent.providers.base import Provider
from emiya_agent.providers.openai_compat import OpenAICompatProvider


def parse_route(route: str) -> tuple[str, str]:
    if "/" not in route:
        raise ConfigError(f'路由格式应为 "provider/model"，收到: {route!r}')
    provider, model = route.split("/", 1)
    return provider, model


class ProviderRegistry:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._cache: dict[str, Provider] = {}

    def get(self, route: str) -> tuple[Provider, str]:
        """返回 (Provider 实例, 裸模型 id)。key 缺失时给出可操作的报错。"""
        provider_name, model = parse_route(route)
        if provider_name not in self._cache:
            self._cache[provider_name] = self._build(provider_name)
        return self._cache[provider_name], model

    def invalidate(self, provider_name: str) -> None:
        """provider 定义变更（如 /provider add 覆盖 base_url）后丢弃缓存实例。"""
        self._cache.pop(provider_name, None)

    def _build(self, provider_name: str) -> Provider:
        pc = self._config.providers.get(provider_name)
        if pc is None:
            raise ConfigError(f"未知 provider: {provider_name!r}（config.toml [providers] 未定义）")
        api_key = self._config.api_key(provider_name)

        if pc.protocol == "anthropic":
            # 仅官方 anthropic 支持 SDK OAuth 凭据链回退；
            # 第三方 anthropic 协议端点（如 deepseek 的 /anthropic）必须给 key
            if provider_name == "anthropic":
                if not api_key and anthropic_fallback_credentials() is None:
                    raise ConfigError(
                        f"缺少 Anthropic 凭据：设置环境变量 {pc.api_key_env}，"
                        "或运行 `ant auth login` 用账号 OAuth 登录"
                        "（注意：两种方式均为 API 按量计费）"
                    )
            elif not api_key:
                raise ConfigError(
                    f"缺少 API key：请设置环境变量 {pc.api_key_env}，"
                    "或在配置 [providers.{name}] 中给 api_key".format(
                        name=provider_name
                    )
                )
            try:
                return AnthropicProvider(
                    api_key=api_key,
                    base_url=pc.base_url,
                    name=provider_name,
                )
            except Exception as exc:
                raise ConfigError(f"Anthropic 协议客户端初始化失败: {exc}") from exc

        if not api_key:
            raise ConfigError(
                f"缺少 API key：请设置环境变量 {pc.api_key_env}"
                f'（PowerShell 当前会话: $env:{pc.api_key_env}="..."；'
                f"持久化: setx {pc.api_key_env} \"...\"），"
                f"或在配置 [providers.{provider_name}] 中给 api_key"
            )
        return OpenAICompatProvider(
            name=provider_name, api_key=api_key, base_url=pc.base_url
        )
