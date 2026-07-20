"""JSONL 会话记录：追加写入、命名检查点与可恢复的分叉基线。"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from emiya_agent.safe_paths import (
    UnsafePathError,
    confined_path,
    is_linklike,
    open_confined,
)
from emiya_agent.types import Message

SESSION_RECORD_VERSION = 1
MAX_SESSION_RECORD_BYTES = 64 * 1024 * 1024
_PATH_LOCKS = tuple(threading.RLock() for _ in range(64))
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.05
_LEASE_GUARD = threading.Lock()
_LEASED_PATHS: set[str] = set()

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class CheckpointExistsError(ValueError):
    """同一会话内的命名检查点不可覆盖。"""


class SessionRecordTooLargeError(ValueError):
    """单条会话记录超过安全上限。"""


class SessionBusyError(RuntimeError):
    """会话已被另一个写入者占用。"""


@dataclass(frozen=True)
class CheckpointInfo:
    name: str
    created_at: str
    message_count: int
    preview: str


@dataclass(frozen=True)
class SessionLineage:
    parent_session_id: str
    checkpoint_name: str | None
    created_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def validate_checkpoint_name(name: str) -> str:
    """规范化便携的检查点名称；名称不会被用作文件路径。"""
    if not isinstance(name, str):
        raise ValueError("检查点名称必须是字符串")
    if any(unicodedata.category(char).startswith("C") for char in name):
        raise ValueError("检查点名称不能包含控制字符")
    normalized = unicodedata.normalize("NFC", name.strip())
    if not normalized:
        raise ValueError("检查点名称不能为空")
    if len(normalized) > 64 or len(normalized.encode("utf-8")) > 128:
        raise ValueError("检查点名称最多 64 个字符且不超过 128 字节")
    if not normalized[0].isalnum() or any(
        not (char.isalnum() or char in "._-") for char in normalized[1:]
    ):
        raise ValueError(
            "检查点名称必须以字母或数字开头，且只能包含字母、数字、点、下划线和连字符"
        )
    return normalized


def _path_lock(path: Path) -> threading.RLock:
    key = _path_key(path)
    return _PATH_LOCKS[hash(key) % len(_PATH_LOCKS)]


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.absolute()))


def _verify_single_link(file: Any, *, label: str) -> None:
    links = getattr(os.fstat(file.fileno()), "st_nlink", 1)
    if links != 1:
        raise UnsafePathError(f"{label} 不能是硬链接（link count={links}）")


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    """用相邻 lock 文件串行化跨进程读取/追加，等待时间有界。"""
    lock_name = f"{path.name}.lock"
    with open_confined(
        path.parent,
        lock_name,
        "a+b",
        label="会话锁文件",
        buffering=0,
    ) as lock_file:
        _verify_single_link(lock_file, label="会话锁文件")
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        if os.name == "nt":
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
            while True:
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("等待会话文件锁超时") from None
                    time.sleep(_LOCK_POLL_SECONDS)
            try:
                yield
            finally:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            while True:
                try:
                    fcntl.flock(
                        lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("等待会话文件锁超时") from None
                    time.sleep(_LOCK_POLL_SECONDS)
            try:
                yield
            finally:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def _process_owns_lease(path: Path) -> bool:
    with _LEASE_GUARD:
        return _path_key(path) in _LEASED_PATHS


@contextmanager
def _read_process_lock(path: Path) -> Iterator[None]:
    if _process_owns_lease(path):
        yield
    else:
        with _process_lock(path):
            yield


def _message_preview(messages: Sequence[Message]) -> str:
    for message in messages:
        if message.role == "user" and message.text().strip():
            return message.text().strip().splitlines()[0][:60]
    return ""


def _validated_messages(data: Any) -> list[Message] | None:
    if not isinstance(data, list):
        return None
    try:
        return [Message.model_validate(item) for item in data]
    except (TypeError, ValueError, RecursionError):
        return None


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path.absolute()
        self._writer_context: Any | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        confined_path(self.path.parent, self.path.name, label="会话文件")

    @property
    def has_writer_lease(self) -> bool:
        return self._writer_context is not None

    def acquire_writer(self) -> None:
        """持有会话级单写者租约，防止旧 snapshot 覆盖并发追加。"""
        if self._writer_context is not None:
            return
        key = _path_key(self.path)
        with _LEASE_GUARD:
            if key in _LEASED_PATHS:
                raise SessionBusyError("会话已在当前进程的另一处打开")
            _LEASED_PATHS.add(key)
        context = _process_lock(self.path)
        try:
            context.__enter__()
        except BaseException as exc:
            with _LEASE_GUARD:
                _LEASED_PATHS.discard(key)
            if isinstance(exc, TimeoutError):
                raise SessionBusyError("会话正由另一个进程写入") from exc
            raise
        self._writer_context = context

    def release_writer(self) -> None:
        context = self._writer_context
        if context is None:
            return
        self._writer_context = None
        key = _path_key(self.path)
        try:
            context.__exit__(None, None, None)
        finally:
            with _LEASE_GUARD:
                _LEASED_PATHS.discard(key)

    @contextmanager
    def _operation_lock(self, *, read_only: bool = False) -> Iterator[None]:
        if self._writer_context is not None or (
            read_only and _process_owns_lease(self.path)
        ):
            yield
        else:
            with _process_lock(self.path):
                yield

    def create(self) -> None:
        """以独占方式原子占用一个新会话文件。"""
        self._write_record(
            {
                "type": "meta",
                "kind": "session_created",
                "created_at": _utc_now(),
                "data": {},
            },
            mode="xb",
        )

    def create_fork(
        self,
        messages: Sequence[Message],
        *,
        parent_session_id: str,
        checkpoint_name: str | None = None,
    ) -> None:
        """以单条独占记录创建分叉，避免 lineage 与上下文只写入一半。"""
        normalized_checkpoint = (
            validate_checkpoint_name(checkpoint_name)
            if checkpoint_name is not None
            else None
        )
        self._write_record(
            {
                "type": "session_fork",
                "version": SESSION_RECORD_VERSION,
                "created_at": _utc_now(),
                "parent_session_id": parent_session_id,
                "checkpoint": normalized_checkpoint,
                "data": [message.model_dump(mode="json") for message in messages],
            },
            mode="xb",
        )

    def append_message(self, message: Message) -> None:
        self._append({"type": "message", "data": message.model_dump(mode="json")})

    def append_meta(self, kind: str, data: dict[str, Any]) -> None:
        """非消息事件：spec、报告、压缩记录等。"""
        self._append({"type": "meta", "kind": kind, "data": data})

    def append_snapshot(self, messages: Sequence[Message]) -> None:
        """记录当前活跃上下文，作为后续恢复的回放基线。

        snapshot 之前的 JSONL 记录仍保留用于审计；``load_messages`` 遇到
        snapshot 后会以其中的消息替换此前回放结果，再继续追加后续 message。
        """
        self._append(
            {
                "type": "active_context",
                "version": SESSION_RECORD_VERSION,
                "data": [message.model_dump(mode="json") for message in messages],
            }
        )

    def append_checkpoint(
        self, name: str, messages: Sequence[Message]
    ) -> CheckpointInfo:
        """保存不可变的命名检查点；同名检查点不会被覆盖。"""
        normalized = validate_checkpoint_name(name)
        lock = _path_lock(self.path)
        with lock:
            with self._operation_lock():
                existing = self._list_checkpoints_unlocked()
                if any(info.name == normalized for info in existing):
                    raise CheckpointExistsError(
                        f"检查点 {normalized!r} 已存在；检查点不可覆盖"
                    )
                created_at = _utc_now()
                copied = [
                    Message.model_validate(message.model_dump(mode="json"))
                    for message in messages
                ]
                self._write_record_unlocked(
                    {
                        "type": "checkpoint",
                        "version": SESSION_RECORD_VERSION,
                        "name": normalized,
                        "created_at": created_at,
                        "data": [
                            message.model_dump(mode="json") for message in copied
                        ],
                    },
                    mode="ab",
                )
            return CheckpointInfo(
                name=normalized,
                created_at=created_at,
                message_count=len(copied),
                preview=_message_preview(copied),
            )

    def list_checkpoints(self) -> list[CheckpointInfo]:
        with _path_lock(self.path):
            with self._operation_lock(read_only=True):
                return self._list_checkpoints_unlocked()

    def _list_checkpoints_unlocked(self) -> list[CheckpointInfo]:
        infos: list[CheckpointInfo] = []
        seen: set[str] = set()
        for record in self._iter_records(self.path):
            parsed = self._parse_checkpoint(record)
            if parsed is None:
                continue
            info, _messages = parsed
            if info.name in seen:
                continue  # 人工追加重复记录时，第一个有效检查点胜出。
            seen.add(info.name)
            infos.append(info)
        return infos

    def load_checkpoint(self, name: str) -> list[Message] | None:
        normalized = validate_checkpoint_name(name)
        with _path_lock(self.path):
            with self._operation_lock(read_only=True):
                seen: set[str] = set()
                for record in self._iter_records(self.path):
                    parsed = self._parse_checkpoint(record)
                    if parsed is None:
                        continue
                    info, messages = parsed
                    if info.name in seen:
                        continue
                    seen.add(info.name)
                    if info.name == normalized:
                        return messages
        return None

    @staticmethod
    def _parse_checkpoint(
        record: dict[str, Any],
    ) -> tuple[CheckpointInfo, list[Message]] | None:
        if (
            record.get("type") != "checkpoint"
            or record.get("version") != SESSION_RECORD_VERSION
        ):
            return None
        try:
            name = validate_checkpoint_name(record["name"])
        except (KeyError, TypeError, ValueError):
            return None
        created_at = record.get("created_at")
        if not isinstance(created_at, str) or not created_at or len(created_at) > 64:
            return None
        messages = _validated_messages(record.get("data"))
        if messages is None:
            return None
        return (
            CheckpointInfo(
                name=name,
                created_at=created_at,
                message_count=len(messages),
                preview=_message_preview(messages),
            ),
            messages,
        )

    def _append(self, record: dict[str, Any]) -> None:
        self._write_record(record, mode="ab")

    def _write_record(self, record: dict[str, Any], *, mode: str) -> None:
        with _path_lock(self.path):
            with self._operation_lock():
                if mode == "xb":
                    self._create_record_atomic(record)
                else:
                    self._write_record_unlocked(record, mode=mode)

    @staticmethod
    def _encode_record(record: dict[str, Any]) -> bytes:
        payload = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if len(payload) > MAX_SESSION_RECORD_BYTES:
            raise SessionRecordTooLargeError(
                "单条会话记录超过 64 MiB，已拒绝写入"
            )
        return payload

    def _create_record_atomic(self, record: dict[str, Any]) -> None:
        """完整写入隐藏临时文件，再以不覆盖的硬链接原子发布。"""
        payload = self._encode_record(record)
        temp_name = (
            f".{self.path.name}.{os.getpid()}-{threading.get_ident()}-"
            f"{secrets.token_hex(8)}.tmp"
        )
        temp_path = confined_path(
            self.path.parent, temp_name, label="会话临时文件"
        )
        temp_stat: os.stat_result | None = None

        def matching_path(
            path: Path, expected: os.stat_result
        ) -> tuple[Path, os.stat_result] | None:
            try:
                safe = confined_path(
                    self.path.parent, path.name, label="会话事务文件"
                )
                if is_linklike(safe):
                    return None
                current = os.stat(safe, follow_symlinks=False)
            except (FileNotFoundError, OSError, UnsafePathError):
                return None
            if not os.path.samestat(current, expected):
                return None
            return safe, current

        try:
            with open_confined(
                self.path.parent,
                temp_name,
                "xb",
                label="会话临时文件",
                buffering=0,
            ) as file:
                _verify_single_link(file, label="会话临时文件")
                temp_stat = os.fstat(file.fileno())
                view = memoryview(payload)
                while view:
                    written = file.write(view)
                    if written is None or written <= 0:
                        raise OSError("会话记录写入不完整")
                    view = view[written:]
                os.fsync(file.fileno())
            with open_confined(
                self.path.parent,
                temp_name,
                "rb",
                label="会话临时文件",
            ) as file:
                _verify_single_link(file, label="会话临时文件")
                if file.read(MAX_SESSION_RECORD_BYTES + 1) != payload:
                    raise OSError("会话临时文件回读校验失败")
            # hard-link 创建在目标已存在时原子失败，不会像 replace 覆盖竞态赢家。
            os.link(temp_path, self.path, follow_symlinks=False)
        except BaseException:
            try:
                if temp_stat is not None:
                    matched = matching_path(temp_path, temp_stat)
                    if matched is not None:
                        matched[0].unlink()
            except (OSError, UnsafePathError):
                pass
            raise

        try:
            safe_temp = confined_path(
                self.path.parent, temp_name, label="会话临时文件"
            )
            safe_temp.unlink()
        except (OSError, UnsafePathError) as exc:
            rollback_error: OSError | UnsafePathError | None = None
            assert temp_stat is not None
            matched_temp = matching_path(temp_path, temp_stat)
            matched_target = matching_path(self.path, temp_stat)
            if matched_temp is None:
                # 另一个清理器已经移除临时目录项；目标降回单链接即为成功发布。
                if matched_target is not None and matched_target[1].st_nlink == 1:
                    return
                rollback_error = OSError("事务临时文件身份已改变或消失")
            elif matched_target is None:
                rollback_error = OSError("最终会话文件身份已改变或消失")
            else:
                try:
                    matched_target[0].unlink()
                except (OSError, UnsafePathError) as rollback_exc:
                    rollback_error = rollback_exc
            # 删除临时链接再尝试一次；即使成功也必须报告本次发布失败。
            try:
                matched_temp = matching_path(temp_path, temp_stat)
                if matched_temp is not None:
                    matched_temp[0].unlink()
            except (OSError, UnsafePathError):
                pass
            detail = (
                f"；最终名称回滚也失败: {rollback_error}"
                if rollback_error is not None
                else "；最终名称已回滚"
            )
            raise OSError(f"会话原子发布后清理临时链接失败{detail}") from exc

    def _write_record_unlocked(
        self, record: dict[str, Any], *, mode: str
    ) -> None:
        payload = self._encode_record(record)
        open_mode = "a+b" if mode == "ab" else mode
        with open_confined(
            self.path.parent,
            self.path.name,
            open_mode,
            label="会话文件",
            buffering=0,
        ) as file:
            _verify_single_link(file, label="会话文件")
            if mode == "ab":
                file.seek(0, os.SEEK_END)
                if file.tell() > 0:
                    file.seek(-1, os.SEEK_END)
                    if file.read(1) != b"\n":
                        # 崩溃留下的半行必须先封口，否则下一条合法记录会与其
                        # 拼成同一物理行并在恢复时一起被跳过。
                        payload = b"\n" + payload
            view = memoryview(payload)
            while view:
                written = file.write(view)
                if written is None or written <= 0:
                    raise OSError("会话记录写入不完整")
                view = view[written:]

    @staticmethod
    def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
        """逐行读取合法对象；坏 UTF-8、超大行和损坏 JSON 仅跳过该行。"""
        try:
            safe_path = confined_path(path.parent, path.name, label="会话文件")
        except UnsafePathError:
            return
        if not safe_path.is_file():
            return
        with open_confined(
            path.parent,
            path.name,
            "rb",
            label="会话文件",
        ) as file:
            _verify_single_link(file, label="会话文件")
            while True:
                raw = file.readline(MAX_SESSION_RECORD_BYTES + 1)
                if not raw:
                    break
                if len(raw) > MAX_SESSION_RECORD_BYTES:
                    while raw and not raw.endswith(b"\n"):
                        raw = file.readline(MAX_SESSION_RECORD_BYTES + 1)
                    continue
                try:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                    continue
                if isinstance(record, dict):
                    yield record

    @staticmethod
    def load_messages(path: Path) -> list[Message]:
        """回放活跃上下文，跳过元数据和任意损坏记录。"""
        messages: list[Message] = []
        with _path_lock(path):
            with _read_process_lock(path):
                for record in SessionStore._iter_records(path):
                    record_type = record.get("type")
                    if record_type == "message":
                        try:
                            messages.append(Message.model_validate(record["data"]))
                        except (KeyError, TypeError, ValueError, RecursionError):
                            continue
                    elif record_type in {"active_context", "session_fork"}:
                        if record.get("version") != SESSION_RECORD_VERSION:
                            continue
                        snapshot = _validated_messages(record.get("data"))
                        if snapshot is not None:
                            messages = snapshot
        return messages

    @staticmethod
    def load_lineage(path: Path) -> SessionLineage | None:
        """读取子会话的首个有效 lineage；后续伪造记录不能覆盖它。"""
        with _path_lock(path):
            with _read_process_lock(path):
                for record in SessionStore._iter_records(path):
                    if (
                        record.get("type") != "session_fork"
                        or record.get("version") != SESSION_RECORD_VERSION
                    ):
                        continue
                    parent = record.get("parent_session_id")
                    created_at = record.get("created_at")
                    checkpoint = record.get("checkpoint")
                    if not isinstance(parent, str) or not parent:
                        continue
                    if not isinstance(created_at, str) or not created_at:
                        continue
                    if checkpoint is not None:
                        try:
                            checkpoint = validate_checkpoint_name(checkpoint)
                        except (TypeError, ValueError):
                            continue
                    if _validated_messages(record.get("data")) is None:
                        continue
                    return SessionLineage(
                        parent_session_id=parent,
                        checkpoint_name=checkpoint,
                        created_at=created_at,
                    )
        return None
