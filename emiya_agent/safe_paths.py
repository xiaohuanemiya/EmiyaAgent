"""运行时文件路径防护：工作目录约束、链接拒绝与 POSIX no-follow 打开。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Any


class UnsafePathError(ValueError):
    """路径越过声明边界，或任一现有组件是链接/reparse point。"""


def is_linklike(path: Path) -> bool:
    """识别 POSIX symlink 与 Windows junction/其他 reparse point。"""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction and is_junction():
        return True
    try:
        attrs = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & 0x400)  # Windows FILE_ATTRIBUTE_REPARSE_POINT


def confined_path(root: Path, relative: str | Path, *, label: str) -> Path:
    """解析 root 下的路径；拒绝逃逸及现有链接组件。"""
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise UnsafePathError(f"{label} 不能是绝对路径: {relative_path}")

    root_path = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(root_path / relative_path))
    try:
        parts = candidate.relative_to(root_path).parts
    except ValueError:
        raise UnsafePathError(f"{label} 超出目录边界: {relative_path}") from None

    if is_linklike(root_path):
        raise UnsafePathError(f"{label} 根目录不能是链接: {root_path}")
    current = root_path
    for part in parts:
        current /= part
        if is_linklike(current):
            raise UnsafePathError(f"{label} 不能使用链接: {current}")

    try:
        candidate.resolve().relative_to(root_path.resolve())
    except (OSError, ValueError):
        raise UnsafePathError(f"{label} 解析后超出目录边界: {relative_path}") from None
    return candidate


def open_confined(
    root: Path,
    relative: str | Path,
    mode: str,
    *,
    label: str,
    **kwargs: Any,
) -> IO[Any]:
    """安全打开 root 内文件；POSIX 同时用 O_NOFOLLOW 收窄竞态窗口。"""
    path = confined_path(root, relative, label=label)

    def opener(raw_path: str, flags: int) -> int:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        return os.open(raw_path, flags | nofollow, 0o666)

    return open(path, mode, opener=opener, **kwargs)
