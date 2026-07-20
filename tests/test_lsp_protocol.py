from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from emiya_agent.lsp import (
    MAX_FRAME_BYTES,
    MAX_HEADER_BYTES,
    LSPClient,
    LSPConnectionClosed,
    LSPProtocolError,
    LSPRequestTimeout,
    encode_message,
    read_message,
)
from emiya_agent.lsp import client as client_module
from emiya_agent.lsp.client import (
    MAX_CONFIGURATION_COMPONENTS,
    MAX_CONFIGURATION_ITEMS,
    MAX_CONFIGURATION_RESPONSE_BYTES,
    MAX_NOTIFICATION_BYTES,
    MAX_PENDING_RESPONSE_BYTES,
    MAX_SERVER_REQUEST_BYTES,
    MAX_SERVER_REQUESTS,
    MAX_STDERR_BYTES,
    LSPClientError,
)

_FAKE_SERVER = r'''
import json
import sys
import time


def read_message():
    header = bytearray()
    while not header.endswith(b"\r\n\r\n"):
        chunk = sys.stdin.buffer.read(1)
        if not chunk:
            return None
        header.extend(chunk)
    length = None
    for line in bytes(header[:-4]).decode("ascii").split("\r\n"):
        name, value = line.split(":", 1)
        if name.lower() == "content-length":
            length = int(value.strip())
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def send(message, fragmented=False):
    body = json.dumps(
        message, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    if fragmented:
        for end in (1, 7, 19, len(frame)):
            if not frame:
                break
            sys.stdout.buffer.write(frame[:end])
            sys.stdout.buffer.flush()
            frame = frame[end:]
            time.sleep(0.005)
    else:
        sys.stdout.buffer.write(frame)
        sys.stdout.buffer.flush()


mode = sys.argv[1]
sys.stderr.buffer.write(b"e" * 100000)
sys.stderr.buffer.flush()
initialize = read_message()
if mode == "invalid":
    sys.stdout.buffer.write(b"Content-Length: 2\r\n\r\n[]")
    sys.stdout.buffer.flush()
    time.sleep(10)
    raise SystemExit
if mode == "oversized":
    sys.stdout.buffer.write(b"Content-Length: 8388609\r\n\r\n")
    sys.stdout.buffer.flush()
    time.sleep(10)
    raise SystemExit

send(
    {
        "jsonrpc": "2.0",
        "id": initialize["id"],
        "result": {"capabilities": {"hoverProvider": True}, "name": "语言服务"},
    },
    fragmented=True,
)
initialized = read_message()
if mode == "blocked":
    time.sleep(10)
    raise SystemExit
first = None
while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    if method == "state":
        send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "initialized": initialized.get("method") == "initialized",
                    "rootUri": initialize["params"]["rootUri"],
                },
            }
        )
    elif method == "first":
        first = message
    elif method == "second":
        send({"jsonrpc": "2.0", "id": message["id"], "result": "two"})
        send({"jsonrpc": "2.0", "id": first["id"], "result": "one"})
    elif method == "triggerServer":
        send(
            {
                "jsonrpc": "2.0",
                "id": "apply",
                "method": "workspace/applyEdit",
                "params": {"edit": {"changes": {"file:///secret": []}}},
            }
        )
        send(
            {
                "jsonrpc": "2.0",
                "id": "config",
                "method": "workspace/configuration",
                "params": {
                    "items": [
                        {"section": "python.analysis"},
                        {"section": "flat"},
                        {"section": "missing"},
                        {},
                    ]
                },
            }
        )
        send(
            {
                "jsonrpc": "2.0",
                "id": "unknown",
                "method": "window/showMessageRequest",
                "params": {"message": "approve?"},
            }
        )
        responses = [read_message(), read_message(), read_message()]
        send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": responses,
            }
        )
    elif method == "slow":
        cancellation = read_message()
        send(
            {
                "jsonrpc": "2.0",
                "method": "test/cancelled",
                "params": cancellation.get("params"),
            }
        )
    elif method == "die":
        sys.stdout.buffer.close()
        raise SystemExit
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": message["id"], "result": None})
    elif method == "exit":
        break
'''


def _reader(data: bytes, *, chunk_size: int | None = None) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    if chunk_size is None:
        reader.feed_data(data)
    else:
        for offset in range(0, len(data), chunk_size):
            reader.feed_data(data[offset : offset + chunk_size])
    reader.feed_eof()
    return reader


def _server_script(tmp_path: Path) -> Path:
    path = tmp_path / "fake_lsp.py"
    path.write_text(_FAKE_SERVER, encoding="utf-8")
    return path


def _client(tmp_path: Path, mode: str = "normal") -> tuple[LSPClient, str]:
    secret = "environment-secret-must-not-leak"
    env = dict(os.environ)
    env["EMIYA_LSP_TEST_SECRET"] = secret
    client = LSPClient(
        (sys.executable, "-u", str(_server_script(tmp_path)), mode),
        cwd=tmp_path,
        env=env,
        settings={"python": {"analysis": {"enabled": True}}, "flat": 7},
        request_timeout=3,
        shutdown_timeout=1,
    )
    return client, secret


@pytest.mark.asyncio
async def test_protocol_uses_utf8_byte_length_and_accepts_fragmented_frames():
    message = {"jsonrpc": "2.0", "method": "测试", "params": {"text": "é"}}
    frame = encode_message(message)
    header, body = frame.split(b"\r\n\r\n", 1)

    assert header == f"Content-Length: {len(body)}".encode()
    assert await read_message(_reader(frame, chunk_size=1)) == message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame",
    [
        b"Content-Length: nope\r\n\r\n{}",
        b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
        b"Content-Length: 2\r\n\r\n[]",
        b"Content-Length: 13\r\n\r\n{\"x\":1,\"x\":2}",
    ],
)
async def test_protocol_rejects_invalid_frames(frame: bytes):
    with pytest.raises(LSPProtocolError):
        await read_message(_reader(frame))


@pytest.mark.asyncio
async def test_protocol_rejects_oversized_headers_and_frames_before_body():
    oversized_header = b"X-A: " + b"x" * MAX_HEADER_BYTES + b"\r\n\r\n"
    with pytest.raises(LSPProtocolError, match="header"):
        await read_message(_reader(oversized_header))

    declared = f"Content-Length: {MAX_FRAME_BYTES + 1}\r\n\r\n".encode()
    with pytest.raises(LSPProtocolError, match="frame"):
        await read_message(_reader(declared))


@pytest.mark.asyncio
async def test_protocol_reports_partial_frame_eof():
    with pytest.raises(LSPConnectionClosed):
        await read_message(_reader(b"Content-Length: 20\r\n\r\n{}"))


@pytest.mark.asyncio
async def test_client_initializes_and_matches_out_of_order_responses(tmp_path: Path):
    client, secret = _client(tmp_path)
    try:
        result = await client.start()
        assert result["name"] == "语言服务"
        assert client.server_capabilities == {"hoverProvider": True}
        assert secret not in repr(client)

        state = await client.request("state")
        assert state == {"initialized": True, "rootUri": tmp_path.as_uri()}
        first = asyncio.create_task(client.request("first"))
        second = asyncio.create_task(client.request("second"))
        assert await asyncio.gather(first, second) == ["one", "two"]
        assert client.pending_count == 0
    finally:
        await client.close()

    assert client.process is not None and client.process.returncode == 0
    assert len(client.stderr_text.encode()) < MAX_STDERR_BYTES + 100
    assert "stderr truncated" in client.stderr_text
    assert secret not in client.stderr_text


@pytest.mark.asyncio
async def test_client_refuses_server_side_effect_requests(tmp_path: Path):
    client, _ = _client(tmp_path)
    try:
        await client.start()
        responses = await client.request("triggerServer")
    finally:
        await client.close()

    by_id = {response["id"]: response for response in responses}
    assert by_id["apply"]["result"]["applied"] is False
    assert by_id["config"]["result"] == [
        {"enabled": True},
        7,
        None,
        {"python": {"analysis": {"enabled": True}}, "flat": 7},
    ]
    assert by_id["unknown"]["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_request_timeout_sends_cancel_notification(tmp_path: Path):
    client, _ = _client(tmp_path)
    try:
        await client.start()
        with pytest.raises(LSPRequestTimeout, match="slow"):
            await client.request("slow", timeout=0.05)
        notification = await client.next_notification(timeout=1)
        assert notification["method"] == "test/cancelled"
        assert notification["params"] == {"id": 2}
        assert client.pending_count == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_eof_fails_pending_request(tmp_path: Path):
    client, _ = _client(tmp_path)
    await client.start()
    with pytest.raises(LSPConnectionClosed):
        await client.request("die")
    await client.close(graceful=False)
    assert client.pending_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invalid", "oversized"])
async def test_client_rejects_bad_server_frames_and_reaps_process(
    tmp_path: Path, mode: str
):
    client, secret = _client(tmp_path, mode)
    with pytest.raises(LSPProtocolError) as raised:
        await client.start()
    assert secret not in str(raised.value)
    assert client.process is not None and client.process.returncode is not None


@pytest.mark.asyncio
async def test_protocol_normalizes_excessively_long_decimal_length():
    frame = b"Content-Length: " + b"9" * 5000 + b"\r\n\r\n"
    with pytest.raises(LSPProtocolError, match="Content-Length"):
        await read_message(_reader(frame))


@pytest.mark.asyncio
async def test_notifications_are_bounded_by_total_wire_bytes(tmp_path: Path):
    client, _ = _client(tmp_path)
    frame_bytes = MAX_NOTIFICATION_BYTES // 2 + 1
    for index in range(3):
        client._dispatch_message(  # noqa: SLF001 - protocol boundary regression
            {
                "jsonrpc": "2.0",
                "method": "test/event",
                "params": {"index": index},
            },
            frame_bytes=frame_bytes,
        )

    assert client._queued_notification_bytes <= MAX_NOTIFICATION_BYTES  # noqa: SLF001
    assert client._notifications.qsize() == 1  # noqa: SLF001
    assert (await client.next_notification(timeout=1))["params"] == {"index": 2}
    await client.close(graceful=False)


@pytest.mark.asyncio
async def test_server_requests_have_count_and_byte_backpressure(tmp_path: Path):
    client, _ = _client(tmp_path)
    for request_id in range(MAX_SERVER_REQUESTS):
        client._dispatch_message(  # noqa: SLF001 - protocol boundary regression
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "unknown/request",
            },
            frame_bytes=1,
        )

    with pytest.raises(LSPProtocolError, match="too many concurrent"):
        client._dispatch_message(  # noqa: SLF001 - protocol boundary regression
            {
                "jsonrpc": "2.0",
                "id": MAX_SERVER_REQUESTS,
                "method": "unknown/request",
            },
            frame_bytes=1,
        )
    await client.close(graceful=False)

    byte_client, _ = _client(tmp_path)
    byte_client._dispatch_message(  # noqa: SLF001
        {"jsonrpc": "2.0", "id": 1, "method": "unknown/request"},
        frame_bytes=MAX_SERVER_REQUEST_BYTES,
    )
    with pytest.raises(LSPProtocolError, match="too many concurrent"):
        byte_client._dispatch_message(  # noqa: SLF001
            {"jsonrpc": "2.0", "id": 2, "method": "unknown/request"},
            frame_bytes=1,
        )
    await byte_client.close(graceful=False)


@pytest.mark.asyncio
async def test_unconsumed_responses_have_a_total_byte_budget(tmp_path: Path):
    client, _ = _client(tmp_path)
    loop = asyncio.get_running_loop()
    first = loop.create_future()
    second = loop.create_future()
    third = loop.create_future()
    client._pending[1] = first  # noqa: SLF001
    client._pending[2] = second  # noqa: SLF001
    client._pending[3] = third  # noqa: SLF001
    frame_bytes = MAX_FRAME_BYTES
    client._dispatch_message(  # noqa: SLF001
        {"jsonrpc": "2.0", "id": 1, "result": "one"},
        frame_bytes=frame_bytes,
    )
    client._dispatch_message(  # noqa: SLF001
        {"jsonrpc": "2.0", "id": 2, "result": "two"},
        frame_bytes=frame_bytes,
    )
    assert client._pending_response_bytes == MAX_PENDING_RESPONSE_BYTES  # noqa: SLF001
    with pytest.raises(LSPProtocolError, match="pending byte budget"):
        client._dispatch_message(  # noqa: SLF001
            {"jsonrpc": "2.0", "id": 3, "result": "three"},
            frame_bytes=frame_bytes,
        )
    client._pending_response_sizes.clear()  # noqa: SLF001
    client._pending_response_bytes = 0  # noqa: SLF001
    await client.close(graceful=False)
    assert first.result() == "one"
    assert second.result() == "two"
    assert isinstance(third.exception(), LSPConnectionClosed)


def test_workspace_configuration_requests_are_bounded(tmp_path: Path):
    client, _ = _client(tmp_path)
    with pytest.raises(ValueError, match="too many"):
        client._configuration_result(  # noqa: SLF001
            {"items": [{}] * (MAX_CONFIGURATION_ITEMS + 1)}
        )

    section = ".".join("x" for _ in range(MAX_CONFIGURATION_COMPONENTS + 1))
    with pytest.raises(ValueError, match="too many components"):
        client._configuration_result(  # noqa: SLF001
            {"items": [{"section": section}]}
        )

    large_client = LSPClient(
        (sys.executable,),
        cwd=tmp_path,
        env={},
        settings={"blob": "x" * (MAX_CONFIGURATION_RESPONSE_BYTES // 2)},
    )
    with pytest.raises(ValueError, match="too large"):
        large_client._configuration_result(  # noqa: SLF001
            {"items": [{}, {}, {}]}
        )


@pytest.mark.asyncio
async def test_concurrent_start_is_single_use(tmp_path: Path):
    client, _ = _client(tmp_path)
    first = asyncio.create_task(client.start())
    second = asyncio.create_task(client.start())
    results = await asyncio.gather(first, second, return_exceptions=True)
    try:
        assert sum(isinstance(result, dict) for result in results) == 1
        assert sum(isinstance(result, LSPClientError) for result in results) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_eof_wakes_notification_waiters(tmp_path: Path):
    client, _ = _client(tmp_path)
    await client.start()
    waiter = asyncio.create_task(client.next_notification())
    try:
        with pytest.raises(LSPConnectionClosed):
            await client.request("die")
        with pytest.raises(LSPConnectionClosed):
            await asyncio.wait_for(waiter, timeout=1)
    finally:
        await client.close(graceful=False)


@pytest.mark.asyncio
async def test_close_aborts_a_blocked_stdin_pipe(tmp_path: Path):
    client, _ = _client(tmp_path, "blocked")
    await client.start()
    payload = {"blob": "x" * (MAX_FRAME_BYTES - 1024)}
    with pytest.raises(LSPRequestTimeout):
        await client.request("large", payload, timeout=0.05)
    await asyncio.wait_for(client.close(graceful=False), timeout=6)
    assert client.process is not None and client.process.returncode is not None


@pytest.mark.asyncio
async def test_terminated_posix_leader_still_cleans_process_group(monkeypatch):
    calls: list[tuple[int, int]] = []

    def killpg(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    class ExitedProcess:
        pid = 4321
        returncode = 0

        async def wait(self) -> int:
            return 0

    monkeypatch.setattr(client_module.sys, "platform", "linux")
    monkeypatch.setattr(client_module.os, "killpg", killpg, raising=False)
    await client_module._terminate_process_tree(  # noqa: SLF001
        ExitedProcess(), None  # type: ignore[arg-type]
    )
    assert calls[0] == (4321, client_module.signal.SIGTERM)


@pytest.mark.asyncio
async def test_cancelled_internal_close_task_does_not_spin(
    tmp_path: Path, monkeypatch
):
    client, _ = _client(tmp_path)

    async def cancelled_close(*, graceful: bool) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(client, "_close_impl", cancelled_close)
    await asyncio.wait_for(client.close(graceful=False), timeout=1)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 601])
def test_client_rejects_effectively_unbounded_timeouts(
    tmp_path: Path, timeout: float
):
    with pytest.raises(ValueError, match="range|finite"):
        LSPClient(
            (sys.executable,),
            cwd=tmp_path,
            env={},
            request_timeout=timeout,
        )
