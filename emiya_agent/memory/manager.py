"""项目记忆：.emiyacode/memory/ 下 MEMORY.md 索引 + 一事一文件。

索引在会话开始时注入系统提示词（静态快照，保证缓存稳定）；
会话中的读写通过 memory 工具，新内容下个会话生效。
"""

from __future__ import annotations

from pathlib import Path

INDEX_FILE = "MEMORY.md"
MAX_FILE_CHARS = 50_000


class MemoryManager:
    def __init__(self, memory_dir: Path) -> None:
        self.dir = memory_dir

    def _ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, name: str) -> Path:
        """限制在 memory 目录内，拒绝路径穿越。"""
        if not name or not name.strip():
            raise ValueError("记忆文件名不能为空")
        path = (self.dir / name).resolve()
        if not path.is_relative_to(self.dir.resolve()):
            raise ValueError(f"非法记忆路径: {name}")
        return path

    def index_text(self) -> str:
        path = self.dir / INDEX_FILE
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                return ""
        return ""

    def list_files(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(
            str(p.relative_to(self.dir))
            for p in self.dir.rglob("*")
            if p.is_file()
        )

    def read(self, name: str) -> str:
        path = self._resolve(name)
        if not path.is_file():
            raise FileNotFoundError(f"记忆文件不存在: {name}")
        return path.read_text(encoding="utf-8")

    def write(self, name: str, content: str) -> None:
        if len(content) > MAX_FILE_CHARS:
            raise ValueError(f"记忆文件过大（>{MAX_FILE_CHARS} 字符），请精简")
        self._ensure_dir()
        path = self._resolve(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

    def delete(self, name: str) -> None:
        path = self._resolve(name)
        if path.is_file():
            path.unlink()
