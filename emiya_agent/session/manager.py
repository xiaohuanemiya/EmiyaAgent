"""会话管理：编号、列表、恢复。存于项目 .emiyacode/sessions/*.jsonl。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from emiya_agent.session.store import SessionStore
from emiya_agent.types import Message


@dataclass
class SessionInfo:
    session_id: str
    path: Path
    mtime: float
    preview: str  # 首条用户消息片段


class SessionManager:
    def __init__(self, project_dir: Path) -> None:
        self.dir = project_dir / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)

    def new_store(self) -> tuple[str, SessionStore]:
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        # 同秒冲突时追加序号
        path = self.dir / f"{session_id}.jsonl"
        n = 1
        while path.exists():
            path = self.dir / f"{session_id}-{n}.jsonl"
            n += 1
        return path.stem, SessionStore(path)

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        infos: list[SessionInfo] = []
        for path in self.dir.glob("*.jsonl"):
            preview = ""
            for msg in SessionStore.load_messages(path):
                if msg.role == "user" and msg.text().strip():
                    preview = msg.text().strip().splitlines()[0][:60]
                    break
            infos.append(
                SessionInfo(
                    session_id=path.stem,
                    path=path,
                    mtime=path.stat().st_mtime,
                    preview=preview,
                )
            )
        infos.sort(key=lambda i: i.mtime, reverse=True)
        return infos[:limit]

    def resume(self, session_id: str | None = None) -> tuple[str, SessionStore, list[Message]] | None:
        """恢复指定会话（缺省最近一个）。返回 (id, 续写 store, 历史消息)。"""
        infos = self.list_sessions(limit=1000)
        if not infos:
            return None
        if session_id is None:
            info = infos[0]
        else:
            matched = [i for i in infos if i.session_id == session_id]
            if not matched:
                return None
            info = matched[0]
        messages = SessionStore.load_messages(info.path)
        return info.session_id, SessionStore(info.path), messages
