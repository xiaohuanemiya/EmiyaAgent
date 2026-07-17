"""配置叠加合并 / 路由解析 / key 检测。"""

from pathlib import Path

import pytest

from emiya_agent.config import DEFAULTS, deep_merge, load_config
from emiya_agent.errors import ConfigError
from emiya_agent.providers.registry import parse_route


def test_deep_merge_nested_override():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    overlay = {"a": {"y": 20, "z": 30}, "c": 4}
    merged = deep_merge(base, overlay)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}
    # 原字典不被修改
    assert base["a"] == {"x": 1, "y": 2}


def test_deep_merge_dict_replaces_scalar():
    assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}


def test_load_config_defaults(tmp_path: Path):
    cfg = load_config(tmp_path)
    assert cfg.routes["executor"] == "anthropic/claude-fable-5"
    assert cfg.providers["deepseek"].base_url == "https://api.deepseek.com"
    assert cfg.language == "zh"


def test_load_config_project_overlay(tmp_path: Path):
    d = tmp_path / ".emiyacode"
    d.mkdir()
    (d / "config.toml").write_text(
        '[routes]\nexecutor = "openai/gpt-5.6"\n\nlanguage = "en"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.routes["executor"] == "openai/gpt-5.6"
    # 其余路由保留默认
    assert cfg.routes["translator"] == "deepseek/deepseek-chat"


def test_load_config_tolerates_utf8_bom(tmp_path: Path):
    """Windows 记事本/PowerShell 写的 UTF-8 带 BOM，必须能解析。"""
    d = tmp_path / ".emiyacode"
    d.mkdir()
    content = 'language = "en"\n'.encode("utf-8")
    (d / "config.toml").write_bytes(b"\xef\xbb\xbf" + content)
    cfg = load_config(tmp_path)
    assert cfg.language == "en"


def test_api_key_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(tmp_path)
    status = cfg.key_status()
    assert status["deepseek"] is True
    assert status["anthropic"] is False


def test_parse_route():
    assert parse_route("anthropic/claude-fable-5") == ("anthropic", "claude-fable-5")
    with pytest.raises(ConfigError):
        parse_route("no-slash")


def test_route_missing_role(tmp_path: Path):
    cfg = load_config(tmp_path)
    with pytest.raises(ConfigError):
        cfg.route("nonexistent_role")


def test_defaults_models_have_context_windows():
    for route, info in DEFAULTS["models"].items():
        assert info["context_window"] > 0, route
