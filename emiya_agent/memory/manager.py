"""项目记忆：.emiyacode/memory/ 下 MEMORY.md 索引 + 一事一文件。

索引在会话开始时注入系统提示词（静态快照，保证缓存稳定）；
会话中的读写通过 memory 工具，新内容下个会话生效。
"""

from __future__ import annotations

from pathlib import Path

from emiya_agent.safe_paths import UnsafePathError, confined_path, open_confined

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
        try:
            return confined_path(self.dir, name, label="记忆文件")
        except UnsafePathError as exc:
            raise ValueError(f"非法记忆路径: {name}（{exc}）") from None

    def index_text(self) -> str:
        try:
            path = self._resolve(INDEX_FILE)
            if not path.is_file():
                return ""
            with open_confined(
                self.dir,
                INDEX_FILE,
                "r",
                label="记忆索引",
                encoding="utf-8",
            ) as file:
                return file.read().strip()
        except (OSError, UnicodeDecodeError, ValueError):
            return ""

    def list_files(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        files: list[str] = []
        for path in self.dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.dir)
            try:
                self._resolve(str(relative))
            except ValueError:
                continue
            files.append(str(relative))
        return sorted(files)

    def read(self, name: str) -> str:
        path = self._resolve(name)
        if not path.is_file():
            raise FileNotFoundError(f"记忆文件不存在: {name}")
        with open_confined(
            self.dir, name, "r", label="记忆文件", encoding="utf-8"
        ) as file:
            return file.read()

    def write(self, name: str, content: str) -> None:
        if len(content) > MAX_FILE_CHARS:
            raise ValueError(f"记忆文件过大（>{MAX_FILE_CHARS} 字符），请精简")
        self._ensure_dir()
        path = self._resolve(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # mkdir 后再校验，防止嵌套目录本身是仓库预置的链接。
        self._resolve(name)
        with open_confined(
            self.dir,
            name,
            "w",
            label="记忆文件",
            encoding="utf-8",
            newline="\n",
        ) as file:
            file.write(content)

    def delete(self, name: str) -> None:
        path = self._resolve(name)
        if path.is_file():
            path.unlink()
