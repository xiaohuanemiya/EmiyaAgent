"""Process-wide run limits shared by every model request in one invocation."""

from __future__ import annotations

import threading
from decimal import Decimal
from enum import IntEnum
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from emiya_agent.config import Config, ModelInfo
from emiya_agent.types import Usage

StopReason = Literal[
    "max_turns",
    "budget_exceeded",
    "budget_unavailable",
    "timeout",
]

_MILLION = Decimal(1_000_000)
_CACHE_WRITE_MULTIPLIER = Decimal("1.25")


class ExitCode(IntEnum):
    """Stable process exit codes for interactive and automation clients."""

    SUCCESS = 0
    EXECUTION_ERROR = 1
    USAGE = 2
    CONFIG_ERROR = 3
    PROVIDER_ERROR = 4
    REFUSAL = 5
    MAX_TURNS = 6
    BUDGET_EXCEEDED = 7
    MAX_TOKENS = 8
    PROTOCOL_ERROR = 9
    TIMEOUT = 124
    INTERRUPTED = 130


class RunLimits(BaseModel):
    """Immutable optional limits for a complete top-level invocation."""

    model_config = ConfigDict(frozen=True)

    max_turns: int | None = Field(default=None, gt=0, strict=True)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_budget_usd: Decimal | None = Field(default=None, gt=0)

    @field_validator("timeout_seconds")
    @classmethod
    def timeout_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and value != float("inf"):
            return value
        if value is not None:
            raise ValueError("timeout_seconds must be finite")
        return None

    @field_validator("max_budget_usd")
    @classmethod
    def budget_must_be_finite(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError("max_budget_usd must be finite")
        return value


class RunLimitExceeded(RuntimeError):
    """Raised before a request when a stable invocation limit is reached."""

    def __init__(self, reason: StopReason) -> None:
        self.reason = reason
        super().__init__(reason)


def exit_code_for_reason(reason: StopReason | None) -> ExitCode:
    """Map a controller stop reason to its stable process exit code."""

    if reason is None:
        return ExitCode.SUCCESS
    return {
        "max_turns": ExitCode.MAX_TURNS,
        "budget_exceeded": ExitCode.BUDGET_EXCEEDED,
        "budget_unavailable": ExitCode.CONFIG_ERROR,
        "timeout": ExitCode.TIMEOUT,
    }[reason]


class RunController:
    """Thread-safe accounting and admission control for all model requests.

    One instance is intended to be shared by the executor, translators,
    compactors, and subagents belonging to a single top-level invocation.
    """

    def __init__(self, config: Config, limits: RunLimits) -> None:
        self._config = config
        self._limits = limits
        self._lock = threading.Lock()
        self._started_at = monotonic()
        self._deadline = (
            self._started_at + limits.timeout_seconds
            if limits.timeout_seconds is not None
            else None
        )
        self._turns = 0
        self._cost = Decimal(0)
        self._usage = Usage()
        self._stop_reason: StopReason | None = None

    async def before_request(self, route: str) -> None:
        """Atomically admit one request or raise :class:`RunLimitExceeded`."""

        with self._lock:
            if self._deadline is not None and monotonic() >= self._deadline:
                self._stop_reason = self._stop_reason or "timeout"
                raise RunLimitExceeded(self._stop_reason)

            if self._stop_reason is not None:
                raise RunLimitExceeded(self._stop_reason)

            if self._limits.max_budget_usd is not None:
                if self._cost >= self._limits.max_budget_usd:
                    self._stop_reason = "budget_exceeded"
                    raise RunLimitExceeded(self._stop_reason)
                if not _has_request_pricing(self._config.model_info(route)):
                    self._stop_reason = "budget_unavailable"
                    raise RunLimitExceeded(self._stop_reason)

            if (
                self._limits.max_turns is not None
                and self._turns >= self._limits.max_turns
            ):
                self._stop_reason = "max_turns"
                raise RunLimitExceeded(self._stop_reason)

            self._turns += 1

    def record_usage(self, route: str, usage: Usage) -> None:
        """Accumulate global usage and exact USD cost for one completed request."""

        with self._lock:
            self._usage = self._usage + usage
            info = self._config.model_info(route)
            if (
                self._limits.max_budget_usd is not None
                and not _has_request_pricing(info)
            ):
                self._stop_reason = self._stop_reason or "budget_unavailable"
                return

            self._cost += _usage_cost(info, usage)
            if (
                self._limits.max_budget_usd is not None
                and self._cost >= self._limits.max_budget_usd
            ):
                self._stop_reason = self._stop_reason or "budget_exceeded"

    @property
    def turns(self) -> int:
        with self._lock:
            return self._turns

    @property
    def cost(self) -> Decimal:
        with self._lock:
            return self._cost

    @property
    def usage(self) -> Usage:
        with self._lock:
            return self._usage.model_copy(deep=True)

    @property
    def stop_reason(self) -> StopReason | None:
        with self._lock:
            return self._stop_reason

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, monotonic() - self._started_at)

    @property
    def remaining_timeout(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - monotonic())


def _decimal_price(value: float) -> Decimal:
    return Decimal(str(value))


def _has_request_pricing(info: ModelInfo) -> bool:
    return _decimal_price(info.input_price) != 0 or _decimal_price(
        info.output_price
    ) != 0


def _usage_cost(info: ModelInfo, usage: Usage) -> Decimal:
    input_price = _decimal_price(info.input_price)
    return (
        Decimal(usage.input_tokens) * input_price
        + Decimal(usage.output_tokens) * _decimal_price(info.output_price)
        + Decimal(usage.cache_read_input_tokens)
        * _decimal_price(info.cache_read_price)
        + Decimal(usage.cache_creation_input_tokens)
        * input_price
        * _CACHE_WRITE_MULTIPLIER
    ) / _MILLION
