"""脱敏诊断：报告绝不含凭据值；检查项状态语义；`emiya doctor` CLI。"""

from __future__ import annotations

import json

import pytest

from emiya_agent import cli
from emiya_agent.config import (
    MCPServerConfig,
    ProviderConfig,
    load_config,
    user_config_path,
)
from emiya_agent.diagnostics import build_report, render_report
from emiya_agent.hooks import HookCommandConfig, HookEvent

_SECRET_ENVS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GLM_API_KEY",
)


def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for env in _SECRET_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "empty_profile"))


def test_report_never_contains_secret_values(tmp_path, monkeypatch) -> None:
    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("EMIYA_DIAG_KEY", "env-secret-value")
    config = load_config(tmp_path)
    config.providers["custom"] = ProviderConfig(
        api_key_env="EMIYA_DIAG_KEY",
        base_url=(
            "https://user:p4ssw0rd@example.com:8443/v1/path-token-secret"
            "?api_key=query-secret#fragment-secret"
        ),
        api_key="plaintext-secret",
    )
    config.hooks = [
        HookCommandConfig(
            id="guard",
            event=HookEvent.PRE_TOOL_USE,
            command=["C:/tools/guard.exe", "--token", "argv-secret"],
        )
    ]
    config.mcp_servers["remote"] = MCPServerConfig(
        url=(
            "https://mcp.example.com/mcp/mcp-path-secret"
            "?key=mcp-url-secret#mcp-fragment-secret"
        )
    )
    config.mcp_servers["local"] = MCPServerConfig(
        command="C:/tools/mcp-server.exe",
        args=["--secret", "mcp-arg-secret"],
        env={"TOKEN": "mcp-env-secret"},
    )

    report = build_report(config)
    dump = json.dumps(report, ensure_ascii=False) + render_report(report)

    for secret in (
        "env-secret-value",
        "plaintext-secret",
        "p4ssw0rd",
        "path-token-secret",
        "query-secret",
        "fragment-secret",
        "argv-secret",
        "mcp-path-secret",
        "mcp-url-secret",
        "mcp-fragment-secret",
        "mcp-arg-secret",
        "mcp-env-secret",
    ):
        assert secret not in dump

    custom = next(p for p in report["providers"] if p["name"] == "custom")
    assert custom["credential_source"] == "EMIYA_DIAG_KEY"  # 只报来源名
    assert custom["base_url"] == "https://example.com:8443"  # 只保留 origin
    assert report["hooks"][0]["executable"] == "guard.exe"  # 只留 basename
    targets = {entry["name"]: entry["target"] for entry in report["mcp_servers"]}
    assert targets["remote"] == "https://mcp.example.com"
    assert targets["local"] == "mcp-server.exe"


def test_report_sanitizes_injected_load_warnings(tmp_path, monkeypatch) -> None:
    _clean_env(monkeypatch, tmp_path)
    secret = "WARNING-CANARY-a80e"
    config = load_config(tmp_path)
    config.load_warnings = [
        secret,
        f"配置文件解析失败 C:/config.toml: {secret}",
        f"未信任项目配置 {secret}",
    ]

    report = build_report(config)
    dump = json.dumps(report, ensure_ascii=False) + render_report(report)

    assert secret not in dump
    assert report["config"]["load_warnings"] == [
        "配置加载警告（详细内容已脱敏）",
        "配置文件解析失败；该配置层已忽略",
        "存在未信任的项目配置；该配置层已忽略",
    ]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://user:password@example.com/a/path-token?key=query#fragment",
            "https://example.com",
        ),
        ("http://[2001:db8::1]:8080/token", "http://[2001:db8::1]:8080"),
        ("https://example.com:not-a-port/token", "(invalid URL)"),
        ("https://example.com:70000/token", "(invalid URL)"),
        ("not a URL/path-token", "(invalid URL)"),
    ],
)
def test_report_url_sanitizer_is_total_and_omits_secret_parts(
    tmp_path, monkeypatch, url: str, expected: str
) -> None:
    _clean_env(monkeypatch, tmp_path)
    config = load_config(tmp_path)
    config.providers["custom"] = ProviderConfig(
        api_key_env="CUSTOM_KEY",
        base_url=url,
    )

    report = build_report(config)
    custom = next(p for p in report["providers"] if p["name"] == "custom")

    assert custom["base_url"] == expected
    assert "password" not in custom["base_url"]
    assert "path-token" not in custom["base_url"]
    assert "query" not in custom["base_url"]
    assert "fragment" not in custom["base_url"]


def test_missing_credentials_fail_and_recover(tmp_path, monkeypatch) -> None:
    _clean_env(monkeypatch, tmp_path)
    report = build_report(load_config(tmp_path))
    assert report["status"] == "fail"
    checks = {check["id"]: check for check in report["checks"]}
    assert checks["route_providers"]["status"] == "fail"  # executor 无凭据

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    recovered = build_report(load_config(tmp_path))
    assert recovered["status"] == "ok"
    dump = json.dumps(recovered, ensure_ascii=False)
    assert "sk-test-anthropic" not in dump
    assert "sk-test-deepseek" not in dump


def test_doctor_cli_json_reports_and_exit_codes(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clean_env(monkeypatch, tmp_path)

    exit_code = cli.run(["doctor", "--json", "--cwd", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3  # fail 级检查项 → CONFIG_ERROR
    assert payload["schema_version"] == 1
    assert payload["status"] == "fail"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    exit_code = cli.run(["doctor", "--json", "--cwd", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["app"]["version"]
    assert "sk-test-anthropic" not in json.dumps(payload)


def test_doctor_cli_config_error_does_not_echo_validation_input(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clean_env(monkeypatch, tmp_path)
    secret = "DOCTOR-VALIDATION-CANARY-f52b"
    user_path = user_config_path()
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_text(
        "[providers.custom]\n"
        f'api_key_env = {{ secret = "{secret}" }}\n',
        encoding="utf-8",
    )

    exit_code = cli.run(["doctor", "--json", "--cwd", str(tmp_path)])
    output = capsys.readouterr().out

    assert exit_code == 3
    payload = json.loads(output)
    assert payload["status"] == "fail"
    assert "配置值校验失败" in payload["error"]
    assert secret not in output
    assert "input_value" not in output


def test_doctor_cli_human_output(tmp_path, monkeypatch, capsys) -> None:
    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")

    exit_code = cli.run(["doctor", "--cwd", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "EmiyaAgent" in out
    assert "角色路由" in out
    assert "sk-test-anthropic" not in out
