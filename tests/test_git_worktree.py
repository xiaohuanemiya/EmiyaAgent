"""End-to-end tests for isolated Git worktrees."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from emiya_agent.agent import git_worktree as worktree_module
from emiya_agent.agent.git_worktree import (
    GitWorktreeError,
    cleanup_worktree,
    create_worktree,
)


def _git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")

    repo = tmp_path / "源代码 repo with spaces"
    repo.mkdir()
    subprocess.run(
        ["git", "init", str(repo)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    _git(repo, "config", "user.name", "EmiyaAgent Tests")
    _git(repo, "config", "user.email", "tests@example.invalid")
    (repo / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


@pytest.mark.asyncio
async def test_create_general_worktree_and_cleanup(
    git_repo: Path, tmp_path: Path
) -> None:
    base = tmp_path / "隔离 worktrees"
    info = await create_worktree(git_repo / ".", base, "abcdef123456")

    assert info.repo_root == git_repo.resolve()
    assert info.path == (base / "abcdef123456").resolve()
    assert info.path.is_dir()
    assert info.task_id == "abcdef123456"
    assert info.branch == "emiya/task-abcdef12"
    assert not info.detached
    assert _git(info.path, "rev-parse", "HEAD").stdout.strip() == info.head
    assert _git(info.path, "branch", "--show-current").stdout.strip() == info.branch

    await cleanup_worktree(info)

    assert not info.path.exists()
    listed = _git(git_repo, "worktree", "list", "--porcelain").stdout
    assert str(info.path) not in listed


@pytest.mark.asyncio
async def test_create_detached_worktree(git_repo: Path, tmp_path: Path) -> None:
    info = await create_worktree(
        git_repo,
        tmp_path / "worktree base",
        "只读任务",
        detached=True,
    )

    assert info.branch is None
    assert info.detached
    symbolic = _git(info.path, "symbolic-ref", "--quiet", "HEAD", check=False)
    assert symbolic.returncode == 1

    await cleanup_worktree(info)
    assert not info.path.exists()


@pytest.mark.asyncio
async def test_create_rejects_dirty_source(git_repo: Path, tmp_path: Path) -> None:
    (git_repo / "untracked.txt").write_text("local edit\n", encoding="utf-8")
    target = tmp_path / "worktrees" / "dirty-task"

    with pytest.raises(GitWorktreeError, match="source worktree is dirty"):
        await create_worktree(git_repo, target.parent, target.name)

    assert not target.exists()


@pytest.mark.asyncio
async def test_cleanup_rejects_dirty_worktree_without_force(
    git_repo: Path, tmp_path: Path
) -> None:
    info = await create_worktree(git_repo, tmp_path / "worktrees", "dirty-cleanup")
    (info.path / "untracked child.txt").write_text("keep me\n", encoding="utf-8")

    with pytest.raises(GitWorktreeError, match="dirty worktree"):
        await cleanup_worktree(info)

    assert info.path.exists()
    assert (info.path / "untracked child.txt").exists()

    await cleanup_worktree(info, force=True)
    assert not info.path.exists()


@pytest.mark.asyncio
async def test_task_id_cannot_escape_base(git_repo: Path, tmp_path: Path) -> None:
    with pytest.raises(GitWorktreeError, match="path separators"):
        await create_worktree(git_repo, tmp_path / "worktrees", "../escape")


@pytest.mark.asyncio
async def test_worktree_creation_does_not_run_repository_hooks(
    git_repo: Path, tmp_path: Path
) -> None:
    marker = git_repo / "post-checkout-ran.txt"
    hook = git_repo / ".git" / "hooks" / "post-checkout"
    hook.write_text(
        "#!/bin/sh\nprintf invoked > post-checkout-ran.txt\n",
        encoding="utf-8",
        newline="\n",
    )
    hook.chmod(0o755)

    info = await create_worktree(git_repo, tmp_path / "worktrees", "no-hooks")
    try:
        assert not marker.exists()
        assert not (info.path / marker.name).exists()
    finally:
        await cleanup_worktree(info)


@pytest.mark.asyncio
async def test_cleanup_rejects_replaced_worktree_path(
    git_repo: Path, tmp_path: Path
) -> None:
    info = await create_worktree(git_repo, tmp_path / "worktrees", "redirected")
    renamed = info.path.with_name("original-worktree")
    info.path.rename(renamed)
    try:
        try:
            info.path.symlink_to(renamed, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks are unavailable")
        with pytest.raises(GitWorktreeError, match="replaced or redirected"):
            await cleanup_worktree(info, force=True)
    finally:
        if info.path.is_symlink():
            info.path.unlink()
        _git(git_repo, "worktree", "repair", str(renamed), check=False)
        _git(git_repo, "worktree", "remove", "--force", str(renamed), check=False)


@pytest.mark.asyncio
async def test_failed_creation_rolls_back_registration_target_and_branch(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run_git = worktree_module._run_git
    injected = False

    async def fail_after_add(args, **kwargs):
        nonlocal injected
        result = await real_run_git(args, **kwargs)
        argv = tuple(str(arg) for arg in args)
        if not injected and "worktree" in argv and "add" in argv:
            injected = True
            raise GitWorktreeError("injected failure after add")
        return result

    monkeypatch.setattr(worktree_module, "_run_git", fail_after_add)
    base = tmp_path / "worktrees"
    with pytest.raises(GitWorktreeError, match="injected failure"):
        await create_worktree(git_repo, base, "rollback-task")

    assert not (base / "rollback-task").exists()
    assert (
        _git(
            git_repo,
            "show-ref",
            "--verify",
            "--quiet",
            "refs/heads/emiya/task-rollback",
            check=False,
        ).returncode
        == 1
    )
    listed = _git(git_repo, "worktree", "list", "--porcelain").stdout
    assert str(base / "rollback-task") not in listed


@pytest.mark.asyncio
async def test_cancelled_creation_finishes_rollback_despite_repeated_cancel(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run_git = worktree_module._run_git
    real_rollback = worktree_module._rollback_partial_worktree
    add_finished = asyncio.Event()
    rollback_entered = asyncio.Event()
    release_rollback = asyncio.Event()

    async def block_after_add(args, **kwargs):
        result = await real_run_git(args, **kwargs)
        argv = tuple(str(arg) for arg in args)
        if "worktree" in argv and "add" in argv:
            add_finished.set()
            await asyncio.Event().wait()
        return result

    async def slow_rollback(*args, **kwargs):
        rollback_entered.set()
        await release_rollback.wait()
        await real_rollback(*args, **kwargs)

    monkeypatch.setattr(worktree_module, "_run_git", block_after_add)
    monkeypatch.setattr(
        worktree_module, "_rollback_partial_worktree", slow_rollback
    )
    base = tmp_path / "worktrees"
    creation = asyncio.create_task(
        create_worktree(git_repo, base, "cancel-rollback")
    )
    await add_finished.wait()
    creation.cancel()
    await rollback_entered.wait()
    creation.cancel()
    await asyncio.sleep(0)
    assert not creation.done()

    release_rollback.set()
    with pytest.raises(asyncio.CancelledError):
        await creation

    assert not (base / "cancel-rollback").exists()
    assert (
        _git(
            git_repo,
            "show-ref",
            "--verify",
            "--quiet",
            "refs/heads/emiya/task-cancel-r",
            check=False,
        ).returncode
        == 1
    )


def test_git_environment_excludes_ambient_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROVIDER_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "secret")
    monkeypatch.setenv("GIT_ASKPASS", "malicious-helper")

    env = worktree_module._git_environment()

    assert "PROVIDER_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_LFS_SKIP_SMUDGE"] == "1"
