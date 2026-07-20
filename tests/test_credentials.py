"""凭据解析：Anthropic OAuth 凭据链回退 + 四家 provider 的路由构造。"""

from pathlib import Path

import pytest

from emiya_agent.config import load_config, persist_trusted_project, user_config_path
from emiya_agent.errors import ConfigError
from emiya_agent.providers.openai_compat import OpenAICompatProvider
from emiya_agent.providers.registry import ProviderRegistry


def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """隔离真实机器上的 key/profile。"""
    for env in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GLM_API_KEY",
    ):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "empty_profile"))
    return load_config(tmp_path)


# ------------------------------------------------------------------ OAuth 回退


def test_anthropic_falls_back_to_auth_token(tmp_path, monkeypatch):
    config = _isolated_config(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oauth-token-xyz")
    provider, model = ProviderRegistry(config).get("anthropic/claude-fable-5")
    assert provider.name == "anthropic"
    assert model == "claude-fable-5"
    assert config.credential_source("anthropic") == "ANTHROPIC_AUTH_TOKEN"


def test_anthropic_falls_back_to_oauth_profile(tmp_path, monkeypatch):
    config = _isolated_config(tmp_path, monkeypatch)
    cred_dir = tmp_path / "empty_profile" / "credentials"
    cred_dir.mkdir(parents=True)
    (cred_dir / "default.json").write_text("{}", encoding="utf-8")
    source = config.credential_source("anthropic")
    assert source is not None and "OAuth profile" in source


def test_anthropic_no_credentials_raises(tmp_path, monkeypatch):
    config = _isolated_config(tmp_path, monkeypatch)
    with pytest.raises(ConfigError, match="ant auth login"):
        ProviderRegistry(config).get("anthropic/claude-fable-5")


def test_openai_still_requires_key(tmp_path, monkeypatch):
    config = _isolated_config(tmp_path, monkeypatch)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        ProviderRegistry(config).get("openai/gpt-5.6")


# ------------------------------------------------------------------ 四家 provider


def test_glm_provider_in_defaults(tmp_path, monkeypatch):
    config = _isolated_config(tmp_path, monkeypatch)
    assert config.providers["glm"].base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert config.providers["glm"].api_key_env == "GLM_API_KEY"
    assert "glm/glm-4.6" in config.models


def test_model_lang_resolution(tmp_path, monkeypatch):
    """模型主要语言：provider 默认 → model 覆盖 → 未知回退 en。"""
    persist_trusted_project(user_config_path(), tmp_path)
    d = tmp_path / ".emiyacode"
    d.mkdir()
    (d / "config.toml").write_text(
        '[models."openrouter/deepseek-chat"]\nlang = "zh"\n'
        "[providers.openrouter]\n"
        'api_key_env = "OPENROUTER_API_KEY"\n'
        'base_url = "https://openrouter.ai/api/v1"\n',
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    # provider 级默认
    assert config.model_lang("deepseek/deepseek-chat") == "zh"
    assert config.model_lang("glm/glm-4.6") == "zh"
    assert config.model_lang("anthropic/claude-fable-5") == "en"
    assert config.model_lang("openai/gpt-5.6") == "en"
    # model 级覆盖（聚合网关下按模型标注）
    assert config.model_lang("openrouter/deepseek-chat") == "zh"
    # 未知 provider 回退 en
    assert config.model_lang("unknown/model-x") == "en"


def test_all_four_providers_buildable(tmp_path, monkeypatch):
    """四家 API 的路由都能构造出 Provider（任意角色可用）。"""
    config = _isolated_config(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-d")
    monkeypatch.setenv("GLM_API_KEY", "sk-g")
    registry = ProviderRegistry(config)

    for route, expected_name in [
        ("anthropic/claude-fable-5", "anthropic"),
        ("openai/gpt-5.6", "openai"),
        ("deepseek/deepseek-chat", "deepseek"),
        ("glm/glm-4.6", "glm"),
    ]:
        provider, model = registry.get(route)
        assert provider.name == expected_name
        assert "/" not in model  # 裸模型 id


def test_glm_uses_openai_compat_with_base_url(tmp_path, monkeypatch):
    config = _isolated_config(tmp_path, monkeypatch)
    monkeypatch.setenv("GLM_API_KEY", "sk-g")
    provider, _ = ProviderRegistry(config).get("glm/glm-4.6")
    assert isinstance(provider, OpenAICompatProvider)
    assert "bigmodel.cn" in str(provider._client.base_url)


def test_any_role_can_route_to_any_provider(tmp_path, monkeypatch):
    """配置层验证：所有角色路由均可指向任意一家。"""
    persist_trusted_project(user_config_path(), tmp_path)
    d = tmp_path / ".emiyacode"
    d.mkdir()
    (d / "config.toml").write_text(
        "[routes]\n"
        'executor = "glm/glm-4.6"\n'
        'translator = "openai/gpt-5.6"\n'
        'compactor = "anthropic/claude-opus-4-8"\n'
        'subagent_explore = "glm/glm-4.6"\n'
        'subagent_plan = "deepseek/deepseek-chat"\n'
        'subagent_general = "openai/gpt-5.6"\n',
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.route("executor") == "glm/glm-4.6"
    assert config.route("compactor") == "anthropic/claude-opus-4-8"
    assert config.route("subagent_plan") == "deepseek/deepseek-chat"
