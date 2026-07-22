"""Anthropic Provider：流式、adaptive thinking、prompt caching、
fable-5 服务端 fallback、refusal 处理。

模型行为规则（当前 API 文档，勿凭旧记忆改动）：
- claude-fable-5 / mythos: 不传 thinking 参数（永远在线，显式传会 400）；
  默认启用服务端 fallback → claude-opus-4-8（beta: server-side-fallback-2026-06-01）。
- 其余 4.6+ 模型: 显式 thinking={"type": "adaptive", "display": "summarized"}。
- 所有现役模型: 不传 temperature/top_p/top_k。
- effort 走 output_config.effort。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import anthropic

from emiya_agent.errors import ProviderError
from emiya_agent.providers.base import (
    BlockStop,
    ChatRequest,
    MessageDone,
    Provider,
    StreamEvent,
    StreamStart,
    TextDelta,
    ThinkingDelta,
    ToolInputDelta,
    ToolSpec,
)
from emiya_agent.types import (
    ContentBlock,
    FallbackBlock,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

FALLBACK_BETA = "server-side-fallback-2026-06-01"
FALLBACK_MODEL = "claude-opus-4-8"
OFFICIAL_ANTHROPIC_SOURCE = "anthropic:official"

_STOP_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
    "stop_sequence": "end_turn",
}


def _is_fable(model: str) -> bool:
    return "fable" in model or "mythos" in model


def anthropic_endpoint_identity(provider_name: str, base_url: str | None) -> str:
    """Return a non-secret identity used to confine opaque provider blocks."""

    if base_url is None:
        return OFFICIAL_ANTHROPIC_SOURCE
    normalized = base_url.rstrip("/")
    try:
        parts = urlsplit(normalized)
        official = (
            parts.scheme.lower() == "https"
            and parts.hostname == "api.anthropic.com"
            and parts.port in (None, 443)
            and parts.username is None
            and parts.password is None
            and parts.path in ("", "/")
            and not parts.query
            and not parts.fragment
        )
    except ValueError:
        official = False
    if official:
        return OFFICIAL_ANTHROPIC_SOURCE
    endpoint_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"anthropic-compatible:{provider_name}:{endpoint_digest}"


def _source_matches(source: str | None, endpoint_identity: str) -> bool:
    # Missing provenance is intentionally fail-closed.  The block remains in
    # session history but no endpoint receives provider-private material whose
    # origin cannot be proven.
    return source is not None and source == endpoint_identity


def _json_value(value: Any) -> Any:
    """Convert an SDK content object into JSON values without interpreting it."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json", by_alias=True))
    try:
        attributes = vars(value)
    except TypeError:
        return value
    return {
        ("from" if key == "from_" else key): _json_value(item)
        for key, item in attributes.items()
        if not key.startswith("_")
    }


def _block_value(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _fallback_data(block: Any) -> dict[str, Any]:
    value = _json_value(block)
    if not isinstance(value, dict):
        return {}
    # ``type`` belongs to the internal discriminator and ``source`` is local
    # provenance.  All other fields are echoed without interpretation.
    return {
        key: item for key, item in value.items() if key not in {"type", "source"}
    }


def _contains_fallback(messages: list[dict[str, Any]]) -> bool:
    return any(
        block.get("type") == "fallback"
        for message in messages
        for block in message["content"]
    )


def to_anthropic_messages(
    messages: list[Message],
    *,
    cache: bool = False,
    endpoint_identity: str = OFFICIAL_ANTHROPIC_SOURCE,
) -> list[dict[str, Any]]:
    """内部消息 → Anthropic 消息 dict。

    - thinking 块仅在带 signature 时回传（同模型续写规则）；无签名的丢弃。
    - cache=True 时在最后一条消息的最后一个 text/tool_result 块上加 cache_control 断点。
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        blocks: list[dict[str, Any]] = []
        for b in m.content:
            if isinstance(b, TextBlock):
                blocks.append({"type": "text", "text": b.text})
            elif isinstance(b, ThinkingBlock):
                if b.signature and _source_matches(b.source, endpoint_identity):
                    blocks.append(
                        {
                            "type": "thinking",
                            "thinking": b.thinking,
                            "signature": b.signature,
                        }
                    )
            elif isinstance(b, RedactedThinkingBlock):
                # redacted_thinking is an official Anthropic opaque format.
                # DeepSeek and other Anthropic-compatible endpoints must never
                # receive it, even if a malformed session claims their source.
                if endpoint_identity == OFFICIAL_ANTHROPIC_SOURCE and _source_matches(
                    b.source, endpoint_identity
                ):
                    blocks.append({"type": "redacted_thinking", "data": b.data})
            elif isinstance(b, FallbackBlock):
                # Server-side fallback boundaries are likewise official-only
                # and must be echoed at their exact position when supported.
                if endpoint_identity == OFFICIAL_ANTHROPIC_SOURCE and _source_matches(
                    b.source, endpoint_identity
                ):
                    blocks.append({**b.data, "type": "fallback"})
            elif isinstance(b, ToolUseBlock):
                blocks.append(
                    {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                )
            elif isinstance(b, ToolResultBlock):
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": b.tool_use_id,
                    "content": b.content,
                }
                if b.is_error:
                    block["is_error"] = True
                blocks.append(block)
        if blocks:
            out.append({"role": m.role, "content": blocks})

    if cache and out:
        for block in reversed(out[-1]["content"]):
            if block["type"] in ("text", "tool_result"):
                block["cache_control"] = {"type": "ephemeral"}
                break
    return out


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """按名称排序保证缓存前缀稳定。"""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in sorted(tools, key=lambda t: t.name)
    ]


def from_anthropic_content(
    content: list[Any], *, source: str = OFFICIAL_ANTHROPIC_SOURCE
) -> list[ContentBlock]:
    """Anthropic 响应块 → 内部块，并规范化最终 fallback 边界。

    Anthropic 要求续轮时保留 fallback 块原位置，但丢弃最终边界之前的
    thinking/redacted_thinking/client tool_use；边界前的 text 仍然保留。
    """

    blocks: list[ContentBlock] = []
    last_fallback = max(
        (
            index
            for index, block in enumerate(content)
            if _block_value(block, "type") == "fallback"
        ),
        default=-1,
    )
    for index, b in enumerate(content):
        btype = _block_value(b, "type")
        if index < last_fallback and btype in {
            "thinking",
            "redacted_thinking",
            "tool_use",
        }:
            continue
        if btype == "text":
            blocks.append(TextBlock(text=_block_value(b, "text")))
        elif btype == "thinking":
            blocks.append(
                ThinkingBlock(
                    thinking=_block_value(b, "thinking") or "",
                    signature=_block_value(b, "signature"),
                    source=source,
                )
            )
        elif btype == "redacted_thinking":
            blocks.append(
                RedactedThinkingBlock(data=_block_value(b, "data"), source=source)
            )
        elif btype == "fallback":
            blocks.append(FallbackBlock(data=_fallback_data(b), source=source))
        elif btype == "tool_use":
            raw = _block_value(b, "input")
            if isinstance(raw, str):  # 防御：极少数情况下未解析
                raw = json.loads(raw) if raw.strip() else {}
            blocks.append(
                ToolUseBlock(
                    id=_block_value(b, "id"),
                    name=_block_value(b, "name"),
                    input=raw or {},
                )
            )
        # server_tool_use 等未知块跳过
    return blocks


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        name: str = "anthropic",
    ) -> None:
        # api_key=None → SDK 凭据链解析：
        # ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN → ant auth login 的 OAuth profile
        # base_url → 第三方 anthropic 兼容端点（如 https://api.deepseek.com/anthropic）
        self.name = name
        self._client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
        # The SDK may resolve ANTHROPIC_BASE_URL or an authenticated profile
        # when no explicit URL was passed.  Provenance must follow the actual
        # transport endpoint, not merely the constructor argument.
        effective_base_url = str(self._client.base_url)
        self._endpoint_identity = anthropic_endpoint_identity(
            name, effective_base_url
        )
        self._official_endpoint = (
            self._endpoint_identity == OFFICIAL_ANTHROPIC_SOURCE
        )

    def _build_kwargs(self, req: ChatRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": to_anthropic_messages(
                req.messages,
                cache=req.cache,
                endpoint_identity=self._endpoint_identity,
            ),
        }
        if req.system:
            block: dict[str, Any] = {"type": "text", "text": req.system}
            if req.cache:
                block["cache_control"] = {"type": "ephemeral"}
            kwargs["system"] = [block]
        if req.tools:
            kwargs["tools"] = to_anthropic_tools(req.tools)
        # effort / adaptive thinking 是 Claude 特有参数；
        # 第三方 anthropic 兼容端点（deepseek 等非 claude 模型）不发，避免 400
        is_claude = req.model.startswith("claude")
        if req.effort and is_claude:
            kwargs["output_config"] = {"effort": req.effort}
        if is_claude and not _is_fable(req.model):
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        return kwargs

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        kwargs = self._build_kwargs(req)
        replaying_fallback = self._official_endpoint and _contains_fallback(
            kwargs["messages"]
        )
        if self._official_endpoint and _is_fable(req.model):
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = [{"model": FALLBACK_MODEL}]
            manager = self._client.beta.messages.stream(**kwargs)
        elif replaying_fallback:
            kwargs["betas"] = [FALLBACK_BETA]
            manager = self._client.beta.messages.stream(**kwargs)
        else:
            manager = self._client.messages.stream(**kwargs)

        try:
            async with manager as stream:
                async for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "content_block_start":
                        cb = event.content_block
                        yield StreamStart(
                            index=event.index,
                            block_type=getattr(cb, "type", "unknown"),
                            tool_use_id=getattr(cb, "id", None),
                            tool_name=getattr(cb, "name", None),
                        )
                    elif etype == "content_block_delta":
                        delta = event.delta
                        dtype = getattr(delta, "type", "")
                        if dtype == "text_delta":
                            yield TextDelta(index=event.index, text=delta.text)
                        elif dtype == "thinking_delta":
                            yield ThinkingDelta(
                                index=event.index, thinking=delta.thinking
                            )
                        elif dtype == "input_json_delta":
                            yield ToolInputDelta(
                                index=event.index, partial_json=delta.partial_json
                            )
                    elif etype == "content_block_stop":
                        yield BlockStop(index=event.index)
                final = await stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API 错误 (HTTP {exc.status_code}): {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"Anthropic 连接失败: {exc}") from exc

        usage = Usage(
            input_tokens=final.usage.input_tokens or 0,
            output_tokens=final.usage.output_tokens or 0,
            cache_read_input_tokens=getattr(final.usage, "cache_read_input_tokens", 0)
            or 0,
            cache_creation_input_tokens=getattr(
                final.usage, "cache_creation_input_tokens", 0
            )
            or 0,
        )
        raw_stop = final.stop_reason or "end_turn"
        stop = _STOP_MAP.get(raw_stop, raw_stop)

        served_by = final.model if final.model != req.model else None
        refusal = None
        if stop == "refusal":
            details = getattr(final, "stop_details", None)
            if details is not None:
                refusal = f"{getattr(details, 'category', None) or 'refused'}: " + (
                    getattr(details, "explanation", None) or ""
                )
            else:
                refusal = "请求被安全分类器拒绝"

        yield MessageDone(
            message=Message(
                role="assistant",
                content=from_anthropic_content(
                    final.content, source=self._endpoint_identity
                ),
            ),
            usage=usage,
            stop_reason=stop,
            served_by=served_by,
            refusal=refusal,
        )

    async def count_tokens(self, req: ChatRequest) -> int | None:
        try:
            kwargs: dict[str, Any] = {
                "model": req.model,
                "messages": to_anthropic_messages(
                    req.messages, endpoint_identity=self._endpoint_identity
                ),
            }
            if req.system:
                kwargs["system"] = req.system
            if req.tools:
                kwargs["tools"] = to_anthropic_tools(req.tools)
            if self._official_endpoint and _contains_fallback(kwargs["messages"]):
                kwargs["betas"] = [FALLBACK_BETA]
                resp = await self._client.beta.messages.count_tokens(**kwargs)
            else:
                resp = await self._client.messages.count_tokens(**kwargs)
            return resp.input_tokens
        except Exception:
            return None
