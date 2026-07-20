"""Machine-readable output primitives for non-interactive CLI modes."""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel

from emiya_agent.providers.base import (
    BlockStop,
    MessageDone,
    StreamEvent,
    StreamStart,
    TextDelta,
    ThinkingDelta,
    ToolInputDelta,
)

SCHEMA_VERSION = "1"


def _jsonable(value: Any) -> Any:
    """Return a value accepted by the standard-library JSON encoder."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


class JsonEventSink:
    """Write versioned NDJSON events to a text stream."""

    def __init__(self, stream: TextIO, enabled: bool = True) -> None:
        self.stream = stream
        self.enabled = enabled
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, event_type: str, **payload: Any) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._seq += 1
            event = {
                **_jsonable(payload),
                "schema_version": SCHEMA_VERSION,
                "seq": self._seq,
                "timestamp": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "type": event_type,
            }
            self.stream.write(
                json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self.stream.flush()


class JsonStreamRenderer:
    """Consume provider events and mirror every event as stable NDJSON."""

    def __init__(self, sink: JsonEventSink) -> None:
        self.sink = sink

    async def render(self, events: AsyncIterator[StreamEvent]) -> MessageDone:
        done: MessageDone | None = None

        async for event in events:
            if isinstance(event, StreamStart):
                self.sink.emit(
                    "stream_start",
                    index=event.index,
                    block_type=event.block_type,
                    tool_use_id=event.tool_use_id,
                    tool_name=event.tool_name,
                )
            elif isinstance(event, TextDelta):
                self.sink.emit("text_delta", index=event.index, text=event.text)
            elif isinstance(event, ThinkingDelta):
                self.sink.emit(
                    "thinking_delta", index=event.index, thinking=event.thinking
                )
            elif isinstance(event, ToolInputDelta):
                self.sink.emit(
                    "tool_input_delta",
                    index=event.index,
                    partial_json=event.partial_json,
                )
            elif isinstance(event, BlockStop):
                self.sink.emit("block_stop", index=event.index)
            elif isinstance(event, MessageDone):
                done = event
                self.sink.emit(
                    "message_done",
                    stop_reason=event.stop_reason,
                    served_by=event.served_by,
                    refusal=event.refusal,
                    message=event.message,
                    usage=event.usage,
                )

        assert done is not None, "event stream did not end with MessageDone"
        return done


def final_payload(
    *,
    status: str,
    stop_reason: str | None = None,
    exit_code: int = 0,
    output: Any = "",
    route: str | None = None,
    model: str | None = None,
    served_by: str | None = None,
    refusal: str | None = None,
    usage: Any = None,
    cost_usd: Any = None,
    turns: int = 0,
    elapsed_seconds: float = 0.0,
    session_id: str | None = None,
    error: Any = None,
) -> dict[str, Any]:
    """Build the stable terminal object shared by JSON and NDJSON modes."""
    return _jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "type": "final",
            "status": status,
            "stop_reason": stop_reason,
            "exit_code": exit_code,
            "output": output,
            "route": route,
            "model": model,
            "served_by": served_by,
            "refusal": refusal,
            "usage": usage,
            "cost_usd": cost_usd,
            "turns": turns,
            "elapsed_seconds": elapsed_seconds,
            "session_id": session_id,
            "error": error,
        }
    )
