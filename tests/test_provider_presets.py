"""供应商预设与配置写回：往返保留注释、加载生效、删除。"""

from pathlib import Path

import pytest

from emiya_agent.config import (
    ProviderConfig,
    load_config,
    persist_provider,
    persist_trusted_project,
    remove_provider_from_file,
    user_config_path,
)
from emiya_agent.providers.openai_compat import OpenAICompatProvider
from emiya_agent.providers.presets import PRESETS
from emiya_agent.providers.registry import ProviderRegistry


def _trust_project(path: Path) -> None:
    persist_trusted_project(user_config_path(), path)


def test_presets_well_formed():
    assert "minimax" in PRESETS
    for name, preset in PRESETS.items():
        assert preset.name == name
        assert preset.base_url.startswith(("http://", "https://"))
        assert preset.api_key_env.isupper()
        assert preset.example_model
        assert preset.lang in ("zh", "en")
    # provider 预设声明其模型默认主要语言，聚合网关可在 model 级覆盖。
    assert PRESETS["minimax"].lang == "zh"
    assert PRESETS["kimi"].lang == "zh"
    assert PRESETS["openrouter"].lang == "en"


def test_persist_provider_lang_roundtrip(tmp_path: Path):
    _trust_project(tmp_path)
    path = tmp_path / ".emiyacode" / "config.toml"
    persist_provider(
        path,
        "minimax",
        ProviderConfig(
            api_key_env="MINIMAX_API_KEY", base_url="https://x/v1", lang="zh"
        ),
    )
    cfg = load_config(tmp_path)
    assert cfg.providers["minimax"].lang == "zh"
    assert cfg.model_lang("minimax/MiniMax-M2") == "zh"


def test_persist_provider_creates_file_and_loads(tmp_path: Path):
    _trust_project(tmp_path)
    path = tmp_path / ".emiyacode" / "config.toml"
    preset = PRESETS["minimax"]
    persist_provider(
        path,
        "minimax",
        ProviderConfig(api_key_env=preset.api_key_env, base_url=preset.base_url),
    )
    assert path.is_file()

    cfg = load_config(tmp_path)  # 项目级配置被读到
    assert "minimax" in cfg.providers
    assert cfg.providers["minimax"].base_url == "https://api.minimaxi.com/v1"
    assert cfg.providers["minimax"].api_key_env == "MINIMAX_API_KEY"
    # 内置四家不受影响
    assert "anthropic" in cfg.providers and "glm" in cfg.providers


def test_persist_provider_preserves_existing_content_and_comments(tmp_path: Path):
    _trust_project(tmp_path)
    path = tmp_path / ".emiyacode" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# 我的备注\n"
        "max_tokens = 12345\n"
        "\n"
        "[routes]\n"
        'executor = "openai/gpt-5.6"  # 行内注释\n',
        encoding="utf-8",
    )
    persist_provider(
        path,
        "minimax",
        ProviderConfig(api_key_env="MINIMAX_API_KEY", base_url="https://x/v1"),
    )
    text = path.read_text(encoding="utf-8")
    assert "# 我的备注" in text
    assert "# 行内注释" in text
    assert "[providers.minimax]" in text

    cfg = load_config(tmp_path)
    assert cfg.max_tokens == 12345  # 原有配置保留
    assert cfg.routes["executor"] == "openai/gpt-5.6"
    assert cfg.providers["minimax"].base_url == "https://x/v1"


def test_persist_provider_overwrites_same_name(tmp_path: Path):
    _trust_project(tmp_path)
    path = tmp_path / ".emiyacode" / "config.toml"
    persist_provider(
        path, "foo", ProviderConfig(api_key_env="A", base_url="https://a/v1")
    )
    persist_provider(
        path, "foo", ProviderConfig(api_key_env="B", base_url="https://b/v1")
    )
    cfg = load_config(tmp_path)
    assert cfg.providers["foo"].base_url == "https://b/v1"
    assert cfg.providers["foo"].api_key_env == "B"


def test_remove_provider_from_file(tmp_path: Path):
    _trust_project(tmp_path)
    path = tmp_path / ".emiyacode" / "config.toml"
    persist_provider(
        path, "foo", ProviderConfig(api_key_env="A", base_url="https://a/v1")
    )
    assert remove_provider_from_file(path, "foo") is True
    assert remove_provider_from_file(path, "foo") is False  # 已不存在
    assert "foo" not in load_config(tmp_path).providers


def test_remove_from_missing_file(tmp_path: Path):
    assert remove_provider_from_file(tmp_path / "nope.toml", "x") is False


def test_added_provider_routable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """新增供应商后任意角色即可路由到它。"""
    _trust_project(tmp_path)
    path = tmp_path / ".emiyacode" / "config.toml"
    preset = PRESETS["minimax"]
    persist_provider(
        path,
        "minimax",
        ProviderConfig(api_key_env=preset.api_key_env, base_url=preset.base_url),
    )
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-mm")
    cfg = load_config(tmp_path)
    provider, model = ProviderRegistry(cfg).get("minimax/MiniMax-M2")
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.name == "minimax"
    assert model == "MiniMax-M2"
