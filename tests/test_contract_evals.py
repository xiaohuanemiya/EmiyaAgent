from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.run_contracts import DEFAULT_CASES_PATH, main, run_contracts


@pytest.mark.asyncio
async def test_default_contract_suite_exercises_safety_boundaries() -> None:
    report = await run_contracts()

    assert report["ok"] is True
    assert report["summary"] == {"total": 9, "passed": 9, "failed": 0}
    cases = {case["id"]: case for case in report["cases"]}
    assert set(cases) == {
        "normal_tool_execution",
        "max_tokens_blocks_side_effect",
        "budget_blocks_side_effect",
        "duplicate_tool_call_id_not_executed",
        "parallel_read_only_probes_execute_without_approval",
        "unknown_tool_reports_error_and_continues",
        "cancel_mid_batch_pairs_remaining_calls",
        "approval_grant_executes_side_effect",
        "approval_denial_blocks_side_effect",
    }
    assert cases["normal_tool_execution"]["observed"]["side_effects"] == [
        {"value": "safe"}
    ]
    for case_id in (
        "max_tokens_blocks_side_effect",
        "budget_blocks_side_effect",
        "duplicate_tool_call_id_not_executed",
        "unknown_tool_reports_error_and_continues",
        "approval_denial_blocks_side_effect",
    ):
        assert cases[case_id]["observed"]["side_effects"] == []
    assert cases["cancel_mid_batch_pairs_remaining_calls"]["observed"][
        "side_effects"
    ] == [{"value": "first", "cancel": True}]


@pytest.mark.asyncio
async def test_contract_report_is_deterministic() -> None:
    assert await run_contracts() == await run_contracts()


def test_json_cli_emits_machine_readable_success(capsys) -> None:
    exit_code = main(["--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["summary"]["failed"] == 0


def test_contract_failure_returns_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = json.loads(DEFAULT_CASES_PATH.read_text(encoding="utf-8"))
    suite["cases"][0]["expect"]["side_effects"] = []
    failing_path = tmp_path / "failing-contracts.json"
    failing_path.write_text(json.dumps(suite), encoding="utf-8")

    exit_code = main(["--json", "--cases", str(failing_path)])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["summary"] == {"total": 9, "passed": 8, "failed": 1}
    failure = payload["cases"][0]["failures"][0]
    assert failure["field"] == "side_effects"
    assert failure["expected"] == []
    assert failure["actual"] == [{"value": "safe"}]
