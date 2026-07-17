"""JSONL 会话转录：逐条追加，崩溃后可完整回放。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emiya_agent.types import Message


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_message(self, message: Message) -> None:
        self._append({"type": "message", "data": message.model_dump(mode="json")})

    def append_meta(self, kind: str, data: dict[str, Any]) -> None:
        """非消息事件：spec、报告、压缩记录等。"""
        self._append({"type": "meta", "kind": kind, "data": data})

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def load_messages(path: Path) -> list[Message]:
        """回放消息（跳过 meta 与损坏行）。"""
        messages: list[Message] = []
        if not path.is_file():
            return messages
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("type") == "message":
                    messages.append(Message.model_validate(record["data"]))
            except (json.JSONDecodeError, ValueError):
                continue  # 尾部损坏行（如崩溃时写了一半）
        return messages
