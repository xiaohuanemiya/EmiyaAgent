"""脱敏诊断：报告绝不含凭据值；检查项状态语义；`emiya doctor` CLI。"""

from __future__ import annotations

import json

import pytest

from emiya_agent import cli
from emiya_agent.config import (
    MCPServerConfig,
    ProviderConfig,
    load_config,
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
        base_url="https://user:p4ssw0rd@example.com/v1?api_key=query-secret",
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
        url="https://mcp.example.com/mcp?key=mcp-url-secret"
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
        "query-secret",
        "argv-secret",
        "mcp-url-secret",
        "mcp-arg-secret",
        "mcp-env-secret",
    ):
        assert secret not in dump

    custom = next(p for p in report["providers"] if p["name"] == "custom")
    assert custom["credential_source"] == "EMIYA_DIAG_KEY"  # 只报来源名
    assert custom["base_url"] == "https://example.com/v1"  # 去 userinfo/query
    assert report["hooks"][0]["executable"] == "guard.exe"  # 只留 basename
    targets = {entry["name"]: entry["target"] for entry in report["mcp_servers"]}
    assert targets["remote"] == "https://mcp.example.com/mcp"
    assert targets["local"] == "mcp-server.exe"


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
