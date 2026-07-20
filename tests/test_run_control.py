from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from pydantic import ValidationError

import emiya_agent.agent.control as control_module
from emiya_agent.agent.control import (
    ExitCode,
    RunController,
    RunLimitExceeded,
    RunLimits,
    exit_code_for_reason,
)
from emiya_agent.config import Config, ModelInfo
from emiya_agent.types import Usage


def priced_config() -> Config:
    return Config(
        models={
            "model/a": ModelInfo(
                input_price=2.0,
                output_price=8.0,
                cache_read_price=0.5,
            ),
            "model/b": ModelInfo(input_price=1.0, output_price=4.0),
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_turns", 0),
        ("max_turns", -1),
        ("max_turns", 1.5),
        ("timeout_seconds", 0),
        ("timeout_seconds", -0.1),
        ("timeout_seconds", float("inf")),
        ("max_budget_usd", Decimal("0")),
        ("max_budget_usd", Decimal("-1")),
        ("max_budget_usd", Decimal("Infinity")),
    ],
)
def test_limits_require_positive_finite_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RunLimits(**{field: value})


def test_limits_are_frozen() -> None:
    limits = RunLimits(max_turns=1)
    with pytest.raises(ValidationError):
        limits.max_turns = 2


async def test_turns_are_global_across_routes() -> None:
    controller = RunController(priced_config(), RunLimits(max_turns=2))

    await controller.before_request("model/a")
    await controller.before_request("model/b")

    with pytest.raises(RunLimitExceeded, match="max_turns") as exc_info:
        await controller.before_request("model/a")
    assert exc_info.value.reason == "max_turns"
    assert controller.turns == 2
    assert controller.stop_reason == "max_turns"


async def test_concurrent_turn_admission_is_atomic() -> None:
    controller = RunController(priced_config(), RunLimits(max_turns=7))

    results = await asyncio.gather(
        *(controller.before_request("model/a") for _ in range(50)),
        return_exceptions=True,
    )

    assert sum(result is None for result in results) == 7
    failures = [result for result in results if isinstance(result, RunLimitExceeded)]
    assert len(failures) == 43
    assert {failure.reason for failure in failures} == {"max_turns"}
    assert controller.turns == 7


async def test_budget_cost_uses_all_usage_categories_and_stops_at_limit() -> None:
    controller = RunController(
        priced_config(), RunLimits(max_budget_usd=Decimal("0.016"))
    )
    await controller.before_request("model/a")

    controller.record_usage(
        "model/a",
        Usage(
            input_tokens=1_000,
            output_tokens=1_000,
            cache_read_input_tokens=1_000,
            cache_creation_input_tokens=1_000,
        ),
    )

    # 0.002 + 0.008 + 0.0005 + (0.002 * 1.25)
    assert controller.cost == Decimal("0.0130")
    assert controller.usage.total_prompt_tokens == 3_000
    assert controller.stop_reason is None

    await controller.before_request("model/b")
    controller.record_usage("model/b", Usage(output_tokens=750))
    assert controller.cost == Decimal("0.0160")
    assert controller.stop_reason == "budget_exceeded"

    with pytest.raises(RunLimitExceeded) as exc_info:
        await controller.before_request("model/a")
    assert exc_info.value.reason == "budget_exceeded"


async def test_budget_fails_closed_when_route_prices_are_missing() -> None:
    controller = RunController(Config(), RunLimits(max_budget_usd=Decimal("1")))

    with pytest.raises(RunLimitExceeded) as exc_info:
        await controller.before_request("unpriced/model")

    assert exc_info.value.reason == "budget_unavailable"
    assert controller.turns == 0
    assert controller.stop_reason == "budget_unavailable"


async def test_deadline_uses_monotonic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr(control_module, "monotonic", lambda: now)
    controller = RunController(priced_config(), RunLimits(timeout_seconds=5))
    assert controller.elapsed_seconds == 0
    assert controller.remaining_timeout == 5

    now = 105.0
    assert controller.remaining_timeout == 0
    with pytest.raises(RunLimitExceeded) as exc_info:
        await controller.before_request("model/a")
    assert exc_info.value.reason == "timeout"
    assert controller.turns == 0


def test_exit_code_mapping_is_stable() -> None:
    assert exit_code_for_reason(None) is ExitCode.SUCCESS
    assert exit_code_for_reason("max_turns") is ExitCode.MAX_TURNS
    assert exit_code_for_reason("budget_exceeded") is ExitCode.BUDGET_EXCEEDED
    assert exit_code_for_reason("budget_unavailable") is ExitCode.CONFIG_ERROR
    assert exit_code_for_reason("timeout") is ExitCode.TIMEOUT
    assert int(ExitCode.INTERRUPTED) == 130
