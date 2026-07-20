"""Run deterministic, network-free contracts against the real agent loop."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import tempfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Never

from emiya_agent.agent.control import RunController, RunLimits
from emiya_agent.agent.loop import AgentLoop
from emiya_agent.config import Config, ModelInfo
from emiya_agent.providers.base import (
    ChatRequest,
    MessageDone,
    Provider,
    StreamEvent,
)
from emiya_agent.tools.base import Tool, ToolContext, ToolResult
from emiya_agent.tools.registry import ToolRegistry
from emiya_agent.types import Message, ToolResultBlock, Usage, user_message

SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
DEFAULT_CASES_PATH = Path(__file__).with_name("cases.json")
MODEL = "contract-model"
ROUTE = "eval/contract-model"

_OBSERVED_FIELDS = frozenset(
    {
        "stop_reason",
        "final_text",
        "side_effects",
        "approval_requests",
        "provider_requests",
        "history_roles",
        "tool_results",
    }
)


def _reject_constant(value: str) -> Never:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _build_turn(raw: dict[str, Any], label: str) -> MessageDone:
    stop_reason = raw.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError(f"{label}.stop_reason must be a non-empty string")
    content = _require_list(raw.get("content"), f"{label}.content")
    usage_raw = _require_mapping(raw.get("usage", {}), f"{label}.usage")
    try:
        message = Message(role="assistant", content=content)
        usage = Usage.model_validate(usage_raw)
    except Exception as exc:
        raise ValueError(f"{label} is invalid: {exc}") from exc
    return MessageDone(message=message, usage=usage, stop_reason=stop_reason)


def _validate_case(raw: Any, index: int, seen_ids: set[str]) -> dict[str, Any]:
    label = f"cases[{index}]"
    case = _require_mapping(raw, label)
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{label}.id must be a non-empty string")
    if case_id in seen_ids:
        raise ValueError(f"duplicate contract case id: {case_id!r}")
    seen_ids.add(case_id)

    description = case.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{label}.description must be a non-empty string")
    if case.get("approval") not in {"ask", "auto"}:
        raise ValueError(f"{label}.approval must be 'ask' or 'auto'")

    decisions = _require_list(
        case.get("approval_decisions", []), f"{label}.approval_decisions"
    )
    if any(type(decision) is not bool for decision in decisions):
        raise ValueError(f"{label}.approval_decisions must contain only booleans")

    turns = _require_list(case.get("turns"), f"{label}.turns")
    if not turns:
        raise ValueError(f"{label}.turns must not be empty")
    for turn_index, turn in enumerate(turns):
        turn_mapping = _require_mapping(turn, f"{label}.turns[{turn_index}]")
        _build_turn(turn_mapping, f"{label}.turns[{turn_index}]")

    expected = _require_mapping(case.get("expect"), f"{label}.expect")
    unknown = sorted(set(expected) - _OBSERVED_FIELDS)
    if unknown:
        raise ValueError(f"{label}.expect has unknown fields: {', '.join(unknown)}")
    if not expected:
        raise ValueError(f"{label}.expect must not be empty")

    limits = _require_mapping(case.get("limits", {}), f"{label}.limits")
    pricing = _require_mapping(case.get("pricing", {}), f"{label}.pricing")
    try:
        RunLimits.model_validate(limits)
        ModelInfo.model_validate(pricing)
    except Exception as exc:
        raise ValueError(f"{label} has invalid limits or pricing: {exc}") from exc
    return case


def load_cases(path: Path | str = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    """Load and validate a strict JSON contract suite."""

    source = Path(path)
    data = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    root = _require_mapping(data, "contract suite")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "unsupported contract schema_version: "
            f"{root.get('schema_version')!r} (expected {SCHEMA_VERSION})"
        )
    raw_cases = _require_list(root.get("cases"), "contract suite.cases")
    if not raw_cases:
        raise ValueError("contract suite.cases must not be empty")
    seen_ids: set[str] = set()
    return [
        _validate_case(raw, index, seen_ids)
        for index, raw in enumerate(raw_cases)
    ]


class _ScriptedProvider(Provider):
    name = "contract-script"

    def __init__(self, turns: Sequence[MessageDone]) -> None:
        self._turns = list(turns)
        self.requests: list[ChatRequest] = []

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(req)
        if not self._turns:
            raise RuntimeError("scripted provider turns exhausted")
        yield self._turns.pop(0)


class _RecordingTool(Tool):
    name: ClassVar[str] = "record"
    description: ClassVar[str] = "Record a deterministic side effect."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": True,
    }
    read_only: ClassVar[bool] = False

    def __init__(self) -> None:
        self.side_effects: list[dict[str, Any]] = []

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.side_effects.append(copy.deepcopy(args))
        return ToolResult(content="recorded")


class _ScriptedUI:
    def __init__(self, approvals: Sequence[bool]) -> None:
        self._approvals = list(approvals)
        self.approval_requests: list[str] = []
        self.notifications: list[str] = []

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        raise RuntimeError("contract eval does not permit interactive questions")

    def notify(self, text: str) -> None:
        self.notifications.append(text)

    async def confirm_tool(self, name: str, args_brief: str) -> bool:
        self.approval_requests.append(name)
        if not self._approvals:
            raise RuntimeError("contract approval decisions exhausted")
        return self._approvals.pop(0)


async def _render(stream: AsyncIterator[StreamEvent]) -> MessageDone:
    terminal: MessageDone | None = None
    async for event in stream:
        if isinstance(event, MessageDone):
            terminal = event
    if terminal is None:
        raise RuntimeError("provider stream ended without MessageDone")
    return terminal


def _turns_for(case: dict[str, Any]) -> list[MessageDone]:
    return [
        _build_turn(turn, f"case {case['id']!r} turn {index}")
        for index, turn in enumerate(case["turns"])
    ]


async def _observe_case(case: dict[str, Any]) -> dict[str, Any]:
    provider = _ScriptedProvider(_turns_for(case))
    tool = _RecordingTool()
    registry = ToolRegistry()
    registry.register(tool)
    ui = _ScriptedUI(case.get("approval_decisions", []))

    with tempfile.TemporaryDirectory(prefix="emiya-contract-") as temp_dir:
        config = Config(cwd=Path(temp_dir), tool_approval=case["approval"])
        config.models[ROUTE] = ModelInfo.model_validate(case.get("pricing", {}))
        limits = RunLimits.model_validate(case.get("limits", {}))
        controller = RunController(config, limits) if case.get("limits") else None
        ctx = ToolContext(cwd=config.cwd, config=config, ui=ui)
        messages = [user_message("exercise the declared contract")]
        loop = AgentLoop(
            provider,
            MODEL,
            ROUTE,
            registry,
            _render,
            max_iterations=max(2, len(case["turns"]) + 1),
            controller=controller,
        )
        result = await loop.run(messages, system="contract-eval", ctx=ctx)

    tool_results = [
        {
            "tool_use_id": block.tool_use_id,
            "is_error": block.is_error,
        }
        for message in messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    return {
        "stop_reason": result.stop_reason,
        "final_text": result.final_text,
        "side_effects": tool.side_effects,
        "approval_requests": ui.approval_requests,
        "provider_requests": len(provider.requests),
        "history_roles": [message.role for message in messages],
        "tool_results": tool_results,
    }


async def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    try:
        observed = await _observe_case(case)
    except Exception as exc:
        observed = {
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        }
        failures = [
            {
                "field": "execution",
                "expected": "case completes",
                "actual": observed["exception"],
            }
        ]
    else:
        failures = []
        for field, expected in case["expect"].items():
            actual = observed[field]
            if actual != expected:
                failures.append(
                    {
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return {
        "id": case["id"],
        "description": case["description"],
        "ok": not failures,
        "failures": failures,
        "observed": observed,
    }


async def run_contracts(
    cases_path: Path | str = DEFAULT_CASES_PATH,
) -> dict[str, Any]:
    """Run every declared case and return a stable machine-readable report."""

    cases = load_cases(cases_path)
    results = [await _evaluate_case(case) for case in cases]
    passed = sum(result["ok"] for result in results)
    total = len(results)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ok": passed == total,
        "summary": {"total": total, "passed": passed, "failed": total - passed},
        "cases": results,
    }


def _error_report(exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ok": False,
        "summary": {"total": 0, "passed": 0, "failed": 0},
        "cases": [],
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _print_human(report: dict[str, Any]) -> None:
    if "error" in report:
        error = report["error"]
        print(f"ERROR {error['type']}: {error['message']}")
        return
    for case in report["cases"]:
        status = "PASS" if case["ok"] else "FAIL"
        print(f"{status} {case['id']}: {case['description']}")
        for failure in case["failures"]:
            print(
                f"  {failure['field']}: expected {failure['expected']!r}, "
                f"observed {failure['actual']!r}"
            )
    summary = report["summary"]
    print(f"{summary['passed']}/{summary['total']} contracts passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="path to a contract suite JSON file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one machine-readable JSON report",
    )
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(run_contracts(args.cases))
    except (OSError, ValueError) as exc:
        report = _error_report(exc)
        exit_code = 2
    else:
        exit_code = 0 if report["ok"] else 1

    if args.json:
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        _print_human(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
