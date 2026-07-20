from __future__ import annotations

import io
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from emiya_agent.providers.base import (
    BlockStop,
    MessageDone,
    StreamStart,
    TextDelta,
    ThinkingDelta,
    ToolInputDelta,
)
from emiya_agent.types import Message, TextBlock, Usage
from emiya_agent.ui.json_output import (
    SCHEMA_VERSION,
    JsonEventSink,
    JsonStreamRenderer,
    final_payload,
)


class FlushTrackingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_event_sink_orders_unicode_events_and_flushes() -> None:
    stream = FlushTrackingStream()
    sink = JsonEventSink(stream)

    sink.emit("first", text="你好，Emiya")
    sink.emit("second", schema_version="bad", seq=-1, type="bad")

    events = _lines(stream)
    assert [event["seq"] for event in events] == [1, 2]
    assert [event["type"] for event in events] == ["first", "second"]
    assert all(event["schema_version"] == SCHEMA_VERSION for event in events)
    assert all(event["timestamp"].endswith("Z") for event in events)
    assert events[0]["text"] == "你好，Emiya"
    assert "你好" in stream.getvalue()
    assert stream.flush_count == 2


def test_disabled_event_sink_is_silent() -> None:
    stream = FlushTrackingStream()
    JsonEventSink(stream, enabled=False).emit("ignored", value=1)
    assert stream.getvalue() == ""
    assert stream.flush_count == 0


@pytest.mark.asyncio
async def test_stream_renderer_maps_every_event() -> None:
    message = Message(role="assistant", content=[TextBlock(text="完成")])
    usage = Usage(input_tokens=5, output_tokens=2)
    expected_done = MessageDone(
        message=message,
        usage=usage,
        stop_reason="end_turn",
        served_by="fallback-model",
        refusal=None,
    )

    async def events():
        yield StreamStart(index=0, block_type="thinking")
        yield ThinkingDelta(index=0, thinking="分析")
        yield BlockStop(index=0)
        yield StreamStart(
            index=1,
            block_type="tool_use",
            tool_use_id="tool-1",
            tool_name="shell",
        )
        yield ToolInputDelta(index=1, partial_json='{"命令":')
        yield BlockStop(index=1)
        yield StreamStart(index=2, block_type="text")
        yield TextDelta(index=2, text="完成")
        yield BlockStop(index=2)
        yield expected_done

    stream = io.StringIO()
    done = await JsonStreamRenderer(JsonEventSink(stream)).render(events())
    emitted = _lines(stream)

    assert done is expected_done
    assert [event["type"] for event in emitted] == [
        "stream_start",
        "thinking_delta",
        "block_stop",
        "stream_start",
        "tool_input_delta",
        "block_stop",
        "stream_start",
        "text_delta",
        "block_stop",
        "message_done",
    ]
    assert emitted[3]["tool_use_id"] == "tool-1"
    assert emitted[4]["partial_json"] == '{"命令":'
    assert emitted[-1]["message"] == message.model_dump(mode="json")
    assert emitted[-1]["usage"] == usage.model_dump(mode="json")
    assert emitted[-1]["served_by"] == "fallback-model"


@pytest.mark.asyncio
async def test_stream_renderer_requires_message_done() -> None:
    async def events():
        yield TextDelta(index=0, text="incomplete")

    with pytest.raises(AssertionError, match="MessageDone"):
        await JsonStreamRenderer(JsonEventSink(io.StringIO())).render(events())


def test_final_payload_is_json_serializable() -> None:
    payload = final_payload(
        status="ok",
        stop_reason="end_turn",
        exit_code=0,
        output={"artifact": Path("产物/result.txt")},
        route="main",
        model="claude-example",
        served_by=None,
        refusal=None,
        usage=Usage(input_tokens=10, output_tokens=4),
        cost_usd=Decimal("0.0123"),
        turns=2,
        elapsed_seconds=1.25,
        session_id="session-1",
        error=None,
    )

    encoded = json.dumps(payload, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == SCHEMA_VERSION
    assert decoded["type"] == "final"
    assert decoded["output"]["artifact"] == str(Path("产物/result.txt"))
    assert decoded["usage"]["input_tokens"] == 10
    assert decoded["cost_usd"] == 0.0123
