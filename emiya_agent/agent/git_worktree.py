"""Safe asynchronous helpers for isolated Git worktrees."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from emiya_agent.config import DEFAULT_SHELL_ENV_ALLOWLIST, select_environment
from emiya_agent.safe_paths import is_linklike
from emiya_agent.tools.shell import (
    _WindowsJob,
    _read_limited,
    _terminate_process_tree,
)

GIT_TIMEOUT_SECONDS = 120
GIT_ROLLBACK_TIMEOUT_SECONDS = 5
GIT_ROLLBACK_TOTAL_SECONDS = 15


class GitWorktreeError(RuntimeError):
    """Raised when a worktree operation cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class WorktreeInfo:
    """Metadata for a worktree created for a delegated task."""

    repo_root: Path
    path: Path
    task_id: str
    head: str
    branch: str | None
    detached: bool
    managed_base: Path | None = None
    git_dir: Path | None = None
    source_git_dir: Path | None = None


def _git_executable() -> Path:
    """Resolve Git from explicit PATH entries, without subprocess cwd lookup."""

    names = ("git.exe",) if os.name == "nt" else ("git",)
    for raw_entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = os.path.expandvars(raw_entry.strip().strip('"'))
        if not entry:
            continue
        for name in names:
            candidate = Path(entry).expanduser() / name
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if resolved.is_file() and (os.name == "nt" or os.access(resolved, os.X_OK)):
                return resolved
    raise GitWorktreeError("git executable was not found in PATH")


def _git_environment() -> dict[str, str]:
    """Use a minimal host environment and disable prompts/download smudges."""

    return select_environment(
        DEFAULT_SHELL_ENV_ALLOWLIST,
        overrides={
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_LFS_SKIP_SMUDGE": "1",
        },
    )


async def _run_bounded_process(
    argv: Sequence[str], *, env: dict[str, str], timeout: float
) -> tuple[int, bytes, int]:
    spawn_options: dict[str, Any] = {}
    if sys.platform == "win32":
        spawn_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        spawn_options["start_new_session"] = True
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        **spawn_options,
    )
    job = _WindowsJob(process.pid) if sys.platform == "win32" else None

    try:
        stdout, omitted = await asyncio.wait_for(
            _read_limited(process), timeout=timeout
        )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        await _terminate_process_tree(process, job)
        raise GitWorktreeError(
            f"process timed out after {timeout:g} seconds"
        ) from exc
    except asyncio.CancelledError:
        await _terminate_process_tree(process, job)
        raise
    finally:
        if process.returncode is None:
            await _terminate_process_tree(process, job)
        elif job is not None:
            job.close()

    assert process.returncode is not None
    return process.returncode, stdout, omitted


async def _run_git(
    args: Sequence[str | Path], *, timeout: float = GIT_TIMEOUT_SECONDS
) -> bytes:
    argv = tuple(str(arg) for arg in args)
    git = _git_executable()
    null_hooks = "NUL" if os.name == "nt" else "/dev/null"
    command = (
        str(git),
        "-c",
        f"core.hooksPath={null_hooks}",
        "-c",
        "core.fsmonitor=false",
        *argv,
    )
    returncode, stdout, omitted = await _run_bounded_process(
        command, env=_git_environment(), timeout=timeout
    )
    if returncode != 0:
        detail = stdout.decode("utf-8", errors="replace").strip()
        if omitted:
            detail += f"\n[output truncated: {omitted} bytes omitted]"
        display = "git " + " ".join(argv)
        message = f"{display} failed with exit code {returncode}"
        if detail:
            message += f": {detail}"
        raise GitWorktreeError(message)
    return stdout


def _decode_path(value: bytes) -> Path:
    text = value.rstrip(b"\r\n").decode("utf-8", errors="surrogateescape")
    if not text:
        raise GitWorktreeError("git returned an empty repository path")
    return Path(text).resolve()


def _branch_name(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", task_id).strip("-_")
    short = safe[:8].rstrip("-_")
    if not short:
        short = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
    return f"emiya/task-{short}"


def _target_path(base_dir: Path, task_id: str) -> Path:
    if not task_id or task_id in {".", ".."} or "\x00" in task_id:
        raise GitWorktreeError("task_id must be a non-empty path component")
    if "/" in task_id or "\\" in task_id:
        raise GitWorktreeError("task_id must not contain path separators")
    if is_linklike(base_dir):
        raise GitWorktreeError("worktree base must not be a symlink or junction")
    return base_dir.resolve() / task_id


async def _branch_ref(
    repo_root: Path,
    branch: str,
    *,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> str:
    output = await _run_git(
        (
            "-C",
            repo_root,
            "for-each-ref",
            "--format=%(objectname)",
            f"refs/heads/{branch}",
        ),
        timeout=timeout,
    )
    return output.decode("ascii", errors="strict").strip()


def _validate_rollback_target(
    managed_base: Path, target: Path, task_id: str
) -> None:
    expected = _target_path(managed_base, task_id)
    if (
        expected != target
        or target.parent != managed_base
        or managed_base.resolve() != managed_base
        or is_linklike(managed_base)
    ):
        raise GitWorktreeError("unsafe worktree rollback target")


async def _remove_tree_bounded(target: Path) -> None:
    python = Path(sys.executable).resolve(strict=True)
    script = "import shutil,sys; shutil.rmtree(sys.argv[1])"
    returncode, output, omitted = await _run_bounded_process(
        (str(python), "-I", "-c", script, str(target)),
        env=select_environment(DEFAULT_SHELL_ENV_ALLOWLIST),
        timeout=GIT_ROLLBACK_TIMEOUT_SECONDS,
    )
    if returncode != 0:
        detail = output.decode("utf-8", errors="replace").strip()
        if omitted:
            detail += f"\n[output truncated: {omitted} bytes omitted]"
        raise GitWorktreeError(
            f"bounded rollback deletion failed with exit code {returncode}: {detail}"
        )


async def _rollback_partial_worktree(
    repo_root: Path,
    managed_base: Path,
    target: Path,
    task_id: str,
    branch: str | None,
    head: str,
) -> None:
    """Best-effort rollback of an add that may have stopped mid-checkout."""

    _validate_rollback_target(managed_base, target, task_id)
    try:
        await _run_git(
            ("-C", repo_root, "worktree", "remove", "--force", target),
            timeout=GIT_ROLLBACK_TIMEOUT_SECONDS,
        )
    except GitWorktreeError:
        # Git may have been killed before registering the freshly-created path.
        pass

    if target.exists() or is_linklike(target):
        _validate_rollback_target(managed_base, target, task_id)
        if is_linklike(target) or target.resolve() != target:
            raise GitWorktreeError("unsafe redirected path left by worktree add")
        await _remove_tree_bounded(target)

    await _run_git(
        ("-C", repo_root, "worktree", "prune", "--expire", "now"),
        timeout=GIT_ROLLBACK_TIMEOUT_SECONDS,
    )
    if (
        branch is not None
        and await _branch_ref(
            repo_root,
            branch,
            timeout=GIT_ROLLBACK_TIMEOUT_SECONDS,
        )
        == head
    ):
        try:
            await _run_git(
                ("-C", repo_root, "branch", "-D", branch),
                timeout=GIT_ROLLBACK_TIMEOUT_SECONDS,
            )
        except GitWorktreeError:
            pass

    if target.exists() or is_linklike(target):
        raise GitWorktreeError("worktree rollback left its target directory behind")
    if branch is not None and await _branch_ref(
        repo_root,
        branch,
        timeout=GIT_ROLLBACK_TIMEOUT_SECONDS,
    ):
        raise GitWorktreeError("worktree rollback left its task branch behind")


async def _wait_for_rollback(task: asyncio.Task[None]) -> None:
    """Finish a bounded rollback even if the parent receives repeated cancels."""

    current = asyncio.current_task()
    while True:
        try:
            await asyncio.shield(task)
            return
        except asyncio.CancelledError:
            if current is not None:
                current.uncancel()


async def create_worktree(
    repo_path: Path,
    base_dir: Path,
    task_id: str,
    *,
    detached: bool = False,
) -> WorktreeInfo:
    """Create a clean, isolated worktree for ``task_id``.

    General tasks receive a new ``emiya/task-<short>`` branch. Passing
    ``detached=True`` checks out the same commit without creating a branch.
    The source worktree must be clean so that no local edits are silently
    omitted from the delegated task.
    """

    target = _target_path(base_dir, task_id)
    repo_root = _decode_path(
        await _run_git(("-C", repo_path, "rev-parse", "--show-toplevel"))
    )
    source_git_dir = _decode_path(
        await _run_git(("-C", repo_root, "rev-parse", "--absolute-git-dir"))
    )
    head_bytes = await _run_git(
        ("-C", repo_root, "rev-parse", "--verify", "HEAD^{commit}")
    )
    head = head_bytes.decode("ascii", errors="strict").strip()
    if not head:
        raise GitWorktreeError("git returned an empty HEAD commit")

    status = await _run_git(
        (
            "-C",
            repo_root,
            "status",
            "--porcelain",
            "-z",
            "--untracked-files=all",
        )
    )
    if status:
        raise GitWorktreeError(
            "refusing to create a worktree while the source worktree is dirty"
        )
    if target.exists():
        raise GitWorktreeError(f"worktree target already exists: {target}")

    managed_base = base_dir.resolve()
    managed_base.mkdir(parents=True, exist_ok=True)
    target = _target_path(managed_base, task_id)
    branch = None if detached else _branch_name(task_id)
    if branch is not None and await _branch_ref(repo_root, branch):
        raise GitWorktreeError(f"task branch already exists: {branch}")
    if detached:
        add_args: tuple[str | Path, ...] = (
            "-C",
            repo_root,
            "worktree",
            "add",
            "--detach",
            target,
            head,
        )
    else:
        add_args = (
            "-C",
            repo_root,
            "worktree",
            "add",
            "-b",
            branch,
            target,
            head,
        )
    try:
        await _run_git(add_args)
        git_dir = _decode_path(
            await _run_git(("-C", target, "rev-parse", "--absolute-git-dir"))
        )
    except BaseException as failure:
        rollback = asyncio.create_task(
            asyncio.wait_for(
                _rollback_partial_worktree(
                    repo_root,
                    managed_base,
                    target,
                    task_id,
                    branch,
                    head,
                ),
                timeout=GIT_ROLLBACK_TOTAL_SECONDS,
            )
        )
        try:
            await _wait_for_rollback(rollback)
        except Exception as rollback_error:
            raise GitWorktreeError(
                f"worktree creation failed and rollback was incomplete: "
                f"{rollback_error}"
            ) from rollback_error
        raise failure

    return WorktreeInfo(
        repo_root=repo_root,
        path=target.resolve(),
        task_id=task_id,
        head=head,
        branch=branch,
        detached=detached,
        managed_base=managed_base,
        git_dir=git_dir,
        source_git_dir=source_git_dir,
    )


async def _validate_managed_worktree(info: WorktreeInfo) -> None:
    if (
        info.managed_base is None
        or info.git_dir is None
        or info.source_git_dir is None
    ):
        raise GitWorktreeError("worktree is missing managed-path metadata")
    try:
        expected = _target_path(info.managed_base, info.task_id)
    except GitWorktreeError as exc:
        raise GitWorktreeError("invalid managed worktree metadata") from exc
    if expected != info.path or info.path.parent != info.managed_base:
        raise GitWorktreeError("worktree path does not match its managed task ID")
    for label, path in (
        ("base", info.managed_base),
        ("worktree", info.path),
        ("git directory", info.git_dir),
        ("source repository", info.repo_root),
        ("source Git directory", info.source_git_dir),
    ):
        if is_linklike(path) or path.resolve() != path:
            raise GitWorktreeError(
                f"refusing to clean up a replaced or redirected {label} path"
            )
    dot_git = info.path / ".git"
    if not dot_git.is_file() or is_linklike(dot_git):
        raise GitWorktreeError("managed worktree .git metadata is missing or redirected")
    current_root = _decode_path(
        await _run_git(("-C", info.path, "rev-parse", "--show-toplevel"))
    )
    current_git_dir = _decode_path(
        await _run_git(("-C", info.path, "rev-parse", "--absolute-git-dir"))
    )
    current_source_git_dir = _decode_path(
        await _run_git(
            ("-C", info.repo_root, "rev-parse", "--absolute-git-dir")
        )
    )
    if (
        current_root != info.path
        or current_git_dir != info.git_dir
        or current_source_git_dir != info.source_git_dir
    ):
        raise GitWorktreeError("managed worktree Git identity has changed")


async def cleanup_worktree(info: WorktreeInfo, *, force: bool = False) -> None:
    """Remove a managed worktree through Git, preserving dirty work by default."""

    await _validate_managed_worktree(info)
    status = await _run_git(
        (
            "-C",
            info.path,
            "status",
            "--porcelain",
            "-z",
            "--untracked-files=all",
        )
    )
    if status and not force:
        raise GitWorktreeError(
            "refusing to remove a dirty worktree without force=True"
        )

    remove_args: list[str | Path] = [
        "-C",
        info.repo_root,
        "worktree",
        "remove",
    ]
    if force:
        remove_args.append("--force")
    remove_args.append(info.path)
    await _run_git(remove_args)
