"""Deterministic, bounded command hooks for EmiyaAgent lifecycle events."""

from typing import TYPE_CHECKING, Any

from emiya_agent.hooks.models import (
    DECISION_EVENTS,
    HookCommandConfig,
    HookDecision,
    HookDiagnostic,
    HookEvent,
    HookRunResult,
)

if TYPE_CHECKING:
    from emiya_agent.hooks.runner import HookRunner


def __getattr__(name: str) -> Any:
    if name == "HookRunner":
        from emiya_agent.hooks.runner import HookRunner

        return HookRunner
    raise AttributeError(name)

__all__ = [
    "DECISION_EVENTS",
    "HookCommandConfig",
    "HookDecision",
    "HookDiagnostic",
    "HookEvent",
    "HookRunResult",
    "HookRunner",
]
