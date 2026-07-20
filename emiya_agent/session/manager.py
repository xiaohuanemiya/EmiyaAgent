"""会话管理：原子编号、列表、恢复与带 lineage 的分叉。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from emiya_agent.safe_paths import UnsafePathError, confined_path, is_linklike
from emiya_agent.session.store import (
    SessionLineage,
    SessionStore,
    validate_checkpoint_name,
)
from emiya_agent.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

_SESSION_ID_PATTERN = re.compile(r"^\d{8}-\d{6}(?:-\d+)?$")
_MAX_ID_ATTEMPTS = 10_000


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    path: Path
    mtime: float
    preview: str
    parent_session_id: str | None = None
    checkpoint_name: str | None = None


def repaired_message_copy(
    messages: Sequence[Message],
    note: str,
) -> tuple[list[Message], bool]:
    """深拷贝并修复整段工具协议，而不改变原列表。

    损坏 JSONL 可能让中间的 tool_result 消失，因此不能只检查最后一条消息。
    有效结果会保留；缺失结果按调用顺序补合成错误，孤立/重复结果会被移除。
    """
    source = [
        Message.model_validate(message.model_dump(mode="json"))
        for message in messages
    ]
    repaired: list[Message] = []
    changed = False
    index = 0
    while index < len(source):
        message = source[index]
        if message.role == "user":
            # 没有紧跟对应 assistant(tool_use) 的结果对 provider 来说是非法的。
            content = [block for block in message.content if isinstance(block, TextBlock)]
            if len(content) != len(message.content):
                changed = True
            if content:
                repaired.append(Message(role="user", content=content))
            else:
                changed = True
            index += 1
            continue

        assistant_content = [
            block
            for block in message.content
            if isinstance(block, (TextBlock, ThinkingBlock, ToolUseBlock))
        ]
        if len(assistant_content) != len(message.content):
            changed = True
        calls = [
            block for block in assistant_content if isinstance(block, ToolUseBlock)
        ]
        ids = [call.id for call in calls]
        invalid_calls = bool(calls) and (
            any(not call.id.strip() or not call.name.strip() for call in calls)
            or len(ids) != len(set(ids))
            or any("_raw" in call.input for call in calls)
        )
        if invalid_calls:
            assistant_content = [
                block
                for block in assistant_content
                if not isinstance(block, ToolUseBlock)
            ]
            calls = []
            changed = True
        if assistant_content:
            repaired.append(Message(role="assistant", content=assistant_content))
        else:
            changed = True

        if not calls:
            index += 1
            continue

        next_message = source[index + 1] if index + 1 < len(source) else None
        existing: dict[str, ToolResultBlock] = {}
        user_text: list[TextBlock] = []
        if next_message is not None and next_message.role == "user":
            for block in next_message.content:
                if isinstance(block, ToolResultBlock):
                    if block.tool_use_id in ids and block.tool_use_id not in existing:
                        existing[block.tool_use_id] = block
                    else:
                        changed = True
                elif isinstance(block, TextBlock):
                    user_text.append(block)
                else:
                    changed = True
            index += 1  # 配对 user 消息由下面的规范化版本替代。

        result_blocks: list[ToolResultBlock] = []
        for call in calls:
            result = existing.get(call.id)
            if result is None:
                changed = True
                result = ToolResultBlock(
                    tool_use_id=call.id,
                    content=note,
                    is_error=True,
                )
            result_blocks.append(result)
        normalized_user = Message(
            role="user", content=[*result_blocks, *user_text]
        )
        if next_message is None or next_message.role != "user":
            changed = True
        elif normalized_user != next_message:
            changed = True
        repaired.append(normalized_user)
        index += 1

    return repaired, changed


class SessionManager:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir.absolute()
        self.dir = confined_path(
            self.project_dir, "sessions", label="会话目录"
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        # mkdir 前后各经过一次约束解析，拒绝已有或竞态替换的链接目录。
        self.dir = confined_path(
            self.project_dir, "sessions", label="会话目录"
        )

    def _safe_dir(self) -> Path:
        return confined_path(
            self.project_dir, "sessions", label="会话目录"
        )

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(
            session_id
        ):
            raise ValueError(f"无效的会话 ID: {session_id!r}")
        return session_id

    def _candidate(self, base_id: str, number: int) -> tuple[str, Path]:
        session_id = base_id if number == 0 else f"{base_id}-{number}"
        return session_id, self._safe_dir() / f"{session_id}.jsonl"

    def new_store(
        self, *, acquire_writer: bool = False
    ) -> tuple[str, SessionStore]:
        """用独占创建保留 ID，多个进程在同一秒也不会串写。"""
        base_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        for number in range(_MAX_ID_ATTEMPTS):
            session_id, path = self._candidate(base_id, number)
            if path.exists() or is_linklike(path):
                continue
            try:
                store = SessionStore(path)
                store.create()
            except FileExistsError:
                continue
            except UnsafePathError:
                if is_linklike(path):
                    continue
                raise
            if acquire_writer:
                store.acquire_writer()
            return session_id, store
        raise RuntimeError("无法分配唯一会话 ID")

    def _store_for_id(self, session_id: str) -> SessionStore:
        valid_id = self._validate_session_id(session_id)
        path = confined_path(
            self._safe_dir(), f"{valid_id}.jsonl", label="会话文件"
        )
        if not path.is_file():
            raise KeyError(f"会话 {valid_id!r} 不存在")
        return SessionStore(path)

    def fork(
        self,
        parent_session_id: str,
        messages: Sequence[Message] | None = None,
        *,
        checkpoint_name: str | None = None,
        acquire_writer: bool = False,
    ) -> tuple[str, SessionStore, list[Message]]:
        """从活跃上下文或父会话的命名检查点创建独立子会话。

        ``checkpoint_name`` 存在时由管理器直接读取不可变检查点；否则必须显式
        传入当前内存中的活跃上下文，避免误从完整审计日志分叉。
        """
        parent_id = self._validate_session_id(parent_session_id)
        parent_store = self._store_for_id(parent_id)
        normalized_checkpoint: str | None = None
        if checkpoint_name is not None:
            if messages is not None:
                raise ValueError("从命名检查点分叉时不能同时传入消息")
            normalized_checkpoint = validate_checkpoint_name(checkpoint_name)
            source_messages = parent_store.load_checkpoint(normalized_checkpoint)
            if source_messages is None:
                raise KeyError(f"检查点 {normalized_checkpoint!r} 不存在")
        else:
            if messages is None:
                raise ValueError("从当前上下文分叉时必须传入消息")
            source_messages = messages

        child_messages, _repaired = repaired_message_copy(
            source_messages,
            "Tool call not executed before the session was forked",
        )
        base_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        for number in range(_MAX_ID_ATTEMPTS):
            child_id, path = self._candidate(base_id, number)
            if path.exists() or is_linklike(path):
                continue
            try:
                child_store = SessionStore(path)
                child_store.create_fork(
                    child_messages,
                    parent_session_id=parent_id,
                    checkpoint_name=normalized_checkpoint,
                )
            except FileExistsError:
                continue
            except UnsafePathError:
                if is_linklike(path):
                    continue
                raise
            if acquire_writer:
                child_store.acquire_writer()
            return child_id, child_store, child_messages
        raise RuntimeError("无法分配唯一分叉会话 ID")

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        if limit <= 0:
            return []
        infos: list[SessionInfo] = []
        try:
            paths = list(self._safe_dir().glob("*.jsonl"))
        except (OSError, UnsafePathError):
            return infos
        for path in paths:
            if not _SESSION_ID_PATTERN.fullmatch(path.stem):
                continue
            try:
                safe_path = confined_path(
                    self._safe_dir(), path.name, label="会话文件"
                )
                stat = safe_path.stat()
            except (OSError, UnsafePathError):
                continue
            preview = ""
            try:
                messages = SessionStore.load_messages(safe_path)
                lineage: SessionLineage | None = SessionStore.load_lineage(
                    safe_path
                )
            except (OSError, UnsafePathError):
                continue
            for message in messages:
                if message.role == "user" and message.text().strip():
                    preview = message.text().strip().splitlines()[0][:60]
                    break
            if lineage is not None and not _SESSION_ID_PATTERN.fullmatch(
                lineage.parent_session_id
            ):
                lineage = None
            infos.append(
                SessionInfo(
                    session_id=path.stem,
                    path=safe_path,
                    mtime=stat.st_mtime,
                    preview=preview,
                    parent_session_id=(
                        lineage.parent_session_id if lineage else None
                    ),
                    checkpoint_name=(lineage.checkpoint_name if lineage else None),
                )
            )
        infos.sort(key=lambda info: (info.mtime, info.session_id), reverse=True)
        return infos[:limit]

    def resume(
        self,
        session_id: str | None = None,
        *,
        acquire_writer: bool = False,
    ) -> tuple[str, SessionStore, list[Message]] | None:
        """恢复指定会话（缺省最近一个），返回 ID、续写 store 与活跃消息。"""
        if session_id is not None:
            try:
                store = self._store_for_id(session_id)
            except (KeyError, ValueError, UnsafePathError):
                return None
            if acquire_writer:
                store.acquire_writer()
            try:
                messages = SessionStore.load_messages(store.path)
            except BaseException:
                store.release_writer()
                raise
            return session_id, store, messages

        infos = self.list_sessions(limit=1000)
        if not infos:
            return None
        info = infos[0]
        store = SessionStore(info.path)
        if acquire_writer:
            store.acquire_writer()
        try:
            messages = SessionStore.load_messages(info.path)
        except BaseException:
            store.release_writer()
            raise
        return info.session_id, store, messages
