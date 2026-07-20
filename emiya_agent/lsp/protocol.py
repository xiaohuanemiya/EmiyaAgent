"""Bounded JSON-RPC framing for the Language Server Protocol."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any

MAX_HEADER_BYTES = 16 * 1024
MAX_FRAME_BYTES = 8 * 1024 * 1024
_HEADER_SEPARATOR = b"\r\n\r\n"
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")


class LSPProtocolError(RuntimeError):
    """Raised when an LSP peer sends an invalid or oversized frame."""


class LSPConnectionClosed(LSPProtocolError):
    """Raised when the peer closes its output before a complete frame arrives."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def encode_message(message: Mapping[str, Any]) -> bytes:
    """Encode one bounded LSP message using its UTF-8 byte length."""

    if not isinstance(message, Mapping):
        raise LSPProtocolError("LSP message must be a JSON object")
    if any(not isinstance(key, str) for key in message):
        raise LSPProtocolError("LSP object keys must be strings")
    body = _encode_json_bytes(dict(message), max_bytes=MAX_FRAME_BYTES)
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _encode_json_bytes(value: Any, *, max_bytes: int) -> bytes:
    """Encode JSON incrementally so an oversized value is never accumulated."""

    body = bytearray()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    try:
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode("utf-8")
            if len(body) + len(encoded) > max_bytes:
                raise LSPProtocolError(
                    f"LSP frame exceeds the {max_bytes}-byte limit"
                )
            body.extend(encoded)
    except LSPProtocolError:
        raise
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise LSPProtocolError("LSP message is not valid JSON") from exc
    return bytes(body)


def encoded_json_size(value: Any, *, max_bytes: int) -> int:
    """Return a bounded UTF-8 JSON size without retaining the encoded value."""

    size = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    try:
        for chunk in encoder.iterencode(value):
            encoded_size = len(chunk.encode("utf-8"))
            if size + encoded_size > max_bytes:
                raise LSPProtocolError(
                    f"LSP frame exceeds the {max_bytes}-byte limit"
                )
            size += encoded_size
    except LSPProtocolError:
        raise
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise LSPProtocolError("LSP message is not valid JSON") from exc
    return size


async def read_message_with_size(
    reader: asyncio.StreamReader,
) -> tuple[dict[str, Any], int]:
    """Read one message and return it with its declared body byte length."""

    try:
        raw_header = await reader.readuntil(_HEADER_SEPARATOR)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            raise LSPConnectionClosed("LSP peer closed the connection") from exc
        raise LSPConnectionClosed(
            "LSP peer closed the connection during a header"
        ) from exc
    except asyncio.LimitOverrunError as exc:
        raise LSPProtocolError("LSP header exceeds the stream limit") from exc

    if len(raw_header) > MAX_HEADER_BYTES:
        raise LSPProtocolError(
            f"LSP header exceeds the {MAX_HEADER_BYTES}-byte limit"
        )
    try:
        header_text = raw_header[: -len(_HEADER_SEPARATOR)].decode("ascii")
    except UnicodeDecodeError as exc:
        raise LSPProtocolError("LSP header must contain ASCII only") from exc

    content_length: int | None = None
    for line in header_text.split("\r\n"):
        if not line or ":" not in line:
            raise LSPProtocolError("malformed LSP header")
        name, value = line.split(":", 1)
        if not _HEADER_NAME.fullmatch(name):
            raise LSPProtocolError("malformed LSP header name")
        if any((ord(character) < 32 and character != "\t") or ord(character) == 127 for character in value):
            raise LSPProtocolError("malformed LSP header value")
        if name.lower() != "content-length":
            continue
        if content_length is not None:
            raise LSPProtocolError("duplicate Content-Length header")
        value = value.strip()
        if not value.isascii() or not value.isdecimal():
            raise LSPProtocolError("invalid Content-Length header")
        try:
            content_length = int(value)
        except ValueError as exc:
            # Python bounds decimal-to-int conversion.  Normalize that failure
            # into the protocol error callers expect for every invalid header.
            raise LSPProtocolError("invalid Content-Length header") from exc

    if content_length is None:
        raise LSPProtocolError("missing Content-Length header")
    if content_length > MAX_FRAME_BYTES:
        raise LSPProtocolError(
            f"LSP frame exceeds the {MAX_FRAME_BYTES}-byte limit"
        )

    try:
        body = await reader.readexactly(content_length)
    except asyncio.IncompleteReadError as exc:
        raise LSPConnectionClosed(
            "LSP peer closed the connection during a frame"
        ) from exc
    try:
        message = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (
        RecursionError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise LSPProtocolError("LSP frame does not contain valid UTF-8 JSON") from exc
    if not isinstance(message, dict):
        raise LSPProtocolError("LSP message must be a JSON object")
    return message, content_length


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read and validate one Content-Length framed JSON object."""

    message, _ = await read_message_with_size(reader)
    return message


async def write_message(
    writer: asyncio.StreamWriter, message: Mapping[str, Any]
) -> None:
    """Write and drain one complete LSP frame."""

    writer.write(encode_message(message))
    try:
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError) as exc:
        raise LSPConnectionClosed("LSP peer closed the connection") from exc
