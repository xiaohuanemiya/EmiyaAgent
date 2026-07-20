"""会话 JSONL 持久化：回放、命名检查点、分叉与损坏恢复。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as RealDateTime
import multiprocessing
from pathlib import Path

import pytest

from emiya_agent.compaction.manager import COMPACT_MARKER, CompactionManager
from emiya_agent.safe_paths import UnsafePathError
from emiya_agent.session import manager as session_manager_module
from emiya_agent.session import store as session_store_module
from emiya_agent.session.manager import SessionManager, repaired_message_copy
from emiya_agent.session.store import (
    CheckpointExistsError,
    SessionBusyError,
    SessionStore,
    validate_checkpoint_name,
)
from emiya_agent.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    user_message,
)
from emiya_agent.ui import commands
from emiya_agent.ui.repl import Repl


def sample_messages() -> list[Message]:
    return [
        user_message("帮我修 bug"),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="", signature="sig-x"),
                TextBlock(text="I'll look."),
                ToolUseBlock(id="t1", name="read_file", input={"path": "a.py"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="x = 1")],
        ),
        Message(role="assistant", content=[TextBlock(text="Fixed.")]),
    ]


def _checkpoint_process_worker(path: str, start, results) -> None:
    store = SessionStore(Path(path))
    start.wait(timeout=10)
    try:
        store.append_checkpoint("shared", [user_message("from worker")])
    except CheckpointExistsError:
        results.put("exists")
    else:
        results.put("created")


def _writer_lease_process_worker(path: str, ready, release) -> None:
    store = SessionStore(Path(path))
    store.acquire_writer()
    ready.set()
    release.wait(timeout=10)
    store.release_writer()


def test_roundtrip(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    original = sample_messages()
    for m in original:
        store.append_message(m)

    loaded = SessionStore.load_messages(path)
    assert loaded == original  # pydantic 等值 → 完整往返


def test_meta_lines_skipped_on_load(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    store.append_message(user_message("hi"))
    store.append_meta("spec", {"requirement_zh": "需求", "spec_en": "spec"})
    store.append_message(user_message("again"))

    loaded = SessionStore.load_messages(path)
    assert len(loaded) == 2


def test_active_context_snapshot_replaces_replay_prefix(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    for message in sample_messages():
        store.append_message(message)

    active = [user_message("[summary] compacted history")]
    store.append_snapshot(active)
    later = Message(role="assistant", content=[TextBlock(text="continued")])
    store.append_message(later)

    assert SessionStore.load_messages(path) == [*active, later]


def test_corrupt_tail_tolerated(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    store.append_message(user_message("ok"))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"type": "message", "data": {"role": "user"')  # 崩溃写一半

    loaded = SessionStore.load_messages(path)
    assert len(loaded) == 1


def test_append_after_corrupt_tail_preserves_new_records(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    first = user_message("first")
    second = user_message("second")
    store.append_message(first)
    with path.open("ab") as file:
        file.write(b'{"type":"message","data":')

    store.append_message(second)
    store.append_checkpoint("after-crash", [first, second])

    assert SessionStore.load_messages(path) == [first, second]
    assert store.load_checkpoint("after-crash") == [first, second]


def test_manager_new_and_resume(tmp_path):
    mgr = SessionManager(tmp_path)
    session_id, store = mgr.new_store()
    for m in sample_messages():
        store.append_message(m)

    resumed = mgr.resume()
    assert resumed is not None
    rid, rstore, messages = resumed
    assert rid == session_id
    assert len(messages) == 4

    # 续写后再次恢复能看到新消息
    rstore.append_message(user_message("追加"))
    _, _, messages2 = mgr.resume(session_id)
    assert len(messages2) == 5


def test_manager_list_previews(tmp_path):
    mgr = SessionManager(tmp_path)
    _, store = mgr.new_store()
    store.append_message(user_message("第一条需求内容"))

    infos = mgr.list_sessions()
    assert len(infos) == 1
    assert infos[0].preview.startswith("第一条需求")


def test_resume_missing_returns_none(tmp_path):
    mgr = SessionManager(tmp_path)
    assert mgr.resume("nonexistent") is None
    assert mgr.resume() is None


def test_session_symlink_is_neither_loaded_nor_appended(tmp_path):
    manager = SessionManager(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text(
        '{"type":"message","data":{"role":"user","content":[]}}\n',
        encoding="utf-8",
    )
    link = manager.dir / "linked.jsonl"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    assert SessionStore.load_messages(link) == []
    assert manager.list_sessions() == []
    with pytest.raises(UnsafePathError):
        SessionStore(link)


@pytest.mark.asyncio
async def test_clear_persists_empty_active_context(tmp_path, quiet_console):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    old_messages = sample_messages()
    for message in old_messages:
        store.append_message(message)

    repl = object.__new__(Repl)
    repl.messages = list(old_messages)
    repl.store = store
    repl.console = quiet_console

    await commands._clear(repl, "")

    assert repl.messages == []
    assert SessionStore.load_messages(path) == []
    fresh = user_message("fresh context")
    store.append_message(fresh)
    assert SessionStore.load_messages(path) == [fresh]


@pytest.mark.asyncio
async def test_compaction_snapshot_is_resume_baseline(
    tmp_path, config, quiet_console
):
    async def fake_complete(system: str, text: str) -> str:
        return "dense summary"

    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    messages = [
        user_message("old request"),
        Message(role="assistant", content=[TextBlock(text="old answer")]),
        user_message("recent request"),
        Message(role="assistant", content=[TextBlock(text="recent answer")]),
    ]
    for message in messages:
        store.append_message(message)

    repl = object.__new__(Repl)
    repl.messages = list(messages)
    repl.store = store
    repl.session_id = "s"
    repl.config = config
    repl.route = "fake/model"
    repl.compactor = CompactionManager(fake_complete, keep_recent_turns=1)
    repl.console = quiet_console

    await repl.maybe_compact(force=True)

    assert COMPACT_MARKER in repl.messages[0].text()
    assert SessionStore.load_messages(path) == repl.messages


@pytest.mark.asyncio
async def test_resume_repairs_and_persists_dangling_tool_use(
    tmp_path, quiet_console
):
    manager = SessionManager(tmp_path)
    session_id, store = manager.new_store()
    store.append_message(user_message("inspect"))
    store.append_message(
        Message(
            role="assistant",
            content=[ToolUseBlock(id="dangling", name="read_file", input={"path": "a.py"})],
        )
    )

    repl = object.__new__(Repl)
    repl.session_mgr = manager
    repl.session_id = None
    repl.store = None
    repl.messages = []
    repl.console = quiet_console

    await commands._resume(repl, session_id)

    assert repl.session_id == session_id
    assert repl.messages[-1].role == "user"
    (repair,) = [
        block
        for block in repl.messages[-1].content
        if isinstance(block, ToolResultBlock)
    ]
    assert repair.tool_use_id == "dangling"
    assert repair.is_error
    assert SessionStore.load_messages(store.path) == repl.messages
    repl.store.release_writer()


def test_repair_persists_in_memory_tool_result_without_duplication(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "s.jsonl")
    request = user_message("inspect")
    assistant = Message(
        role="assistant",
        content=[ToolUseBlock(id="call-1", name="read_file", input={})],
    )
    result = Message(
        role="user",
        content=[
            ToolResultBlock(
                tool_use_id="call-1", content="already executed"
            )
        ],
    )
    store.append_message(request)
    store.append_message(assistant)

    repl = object.__new__(Repl)
    repl.messages = [request, assistant, result]
    repl.store = store

    repl._repair_dangling()
    repl._repair_dangling()

    assert SessionStore.load_messages(store.path) == repl.messages


@pytest.mark.asyncio
async def test_execute_turn_repairs_result_persistence_failure(
    tmp_path, config, quiet_console, monkeypatch
) -> None:
    store = SessionStore(tmp_path / "s.jsonl")
    original_append = store.append_message
    failed_once = False

    def flaky_append(message: Message) -> None:
        nonlocal failed_once
        has_result = any(
            isinstance(block, ToolResultBlock) for block in message.content
        )
        if has_result and not failed_once:
            failed_once = True
            raise OSError("simulated result persistence failure")
        original_append(message)

    monkeypatch.setattr(store, "append_message", flaky_append)

    async def fail_while_persisting_result(
        loop, messages, *, system, ctx
    ):
        del system, ctx
        assistant = Message(
            role="assistant",
            content=[ToolUseBlock(id="call-1", name="read_file", input={})],
        )
        messages.append(assistant)
        loop.on_message(assistant)
        result = Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call-1", content="done")],
        )
        messages.append(result)
        loop.on_message(result)
        raise AssertionError("persistence failure should have propagated")

    monkeypatch.setattr(
        "emiya_agent.ui.repl.AgentLoop.run", fail_while_persisting_result
    )

    class Providers:
        def get(self, route: str):
            del route
            return object(), "model"

    class Renderer:
        async def render(self, stream):
            del stream
            raise AssertionError("renderer should not be called")

    class Usage:
        def add(self, route, usage) -> None:
            del route, usage

    repl = object.__new__(Repl)
    repl.providers = Providers()
    repl.route = "fake/model"
    repl.messages = []
    repl.store = store
    repl.config = config
    repl.tool_ctx = type("Context", (), {"cancel": asyncio.Event()})()
    repl.tools = object()
    repl.renderer = Renderer()
    repl.usage = Usage()
    repl.controller = None
    repl.hooks = None
    repl.system = "system"
    repl.automation = True
    repl.event_sink = None
    repl.console = quiet_console

    with pytest.raises(OSError, match="simulated result persistence failure"):
        await repl._execute_turn("run")

    assert failed_once
    assert SessionStore.load_messages(store.path) == repl.messages


def test_non_object_json_and_bad_utf8_only_skip_their_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    before = user_message("before")
    after = user_message("after")
    store.append_message(before)
    with path.open("ab") as file:
        file.write(b"null\n[]\n\xff\xfe\n")
    store.append_message(after)

    assert SessionStore.load_messages(path) == [before, after]


def test_unknown_snapshot_version_does_not_replace_context(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    original = user_message("keep")
    store.append_message(original)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            '{"type":"active_context","version":999,"data":[]}\n'
        )

    assert SessionStore.load_messages(path) == [original]


def test_named_checkpoint_roundtrip_without_changing_active_context(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    first = user_message("first")
    later = Message(role="assistant", content=[TextBlock(text="later")])
    store.append_message(first)

    info = store.append_checkpoint("版本一", [first])
    store.append_message(later)

    assert info.name == "版本一"
    assert SessionStore(path).load_checkpoint("版本一") == [first]
    assert SessionStore.load_messages(path) == [first, later]
    assert [item.name for item in store.list_checkpoints()] == ["版本一"]


def test_checkpoint_survives_active_context_replacement(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    original = sample_messages()
    for message in original:
        store.append_message(message)
    store.append_checkpoint("before-clear", original)

    store.append_snapshot([])

    assert SessionStore.load_messages(path) == []
    assert store.load_checkpoint("before-clear") == original


def test_duplicate_checkpoint_is_rejected_without_appending(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    store.append_checkpoint("stable", [user_message("original")])
    before = path.read_bytes()

    with pytest.raises(CheckpointExistsError, match="不可覆盖"):
        store.append_checkpoint("stable", [user_message("replacement")])

    assert path.read_bytes() == before
    assert store.load_checkpoint("stable") == [user_message("original")]


@pytest.mark.parametrize(
    "name",
    ["", "   ", "../x", "a/b", "a\\b", ".hidden", "-head", "a b", "a\n", "a\x00"],
)
def test_invalid_checkpoint_names(name):
    with pytest.raises(ValueError):
        validate_checkpoint_name(name)


def test_checkpoint_name_limits_and_unicode_normalization():
    assert validate_checkpoint_name("版本-1") == "版本-1"
    assert validate_checkpoint_name("e\u0301tape") == "étape"
    assert validate_checkpoint_name("a" * 64) == "a" * 64
    with pytest.raises(ValueError):
        validate_checkpoint_name("a" * 65)


def test_corrupt_checkpoint_records_do_not_hide_valid_one(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    valid = [user_message("valid")]
    store.append_checkpoint("good", valid)
    with path.open("ab") as file:
        file.write(
            b'{"type":"checkpoint","version":1,"name":"broken",'
            b'"created_at":"now","data":[{"bad":true}]}\n'
        )
        file.write(
            b'{"type":"checkpoint","version":999,"name":"future",'
            b'"created_at":"now","data":[]}\n'
        )
        file.write(b'{"type":"checkpoint"')

    assert [item.name for item in store.list_checkpoints()] == ["good"]
    assert store.load_checkpoint("good") == valid
    assert store.load_checkpoint("broken") is None


def test_new_store_ids_are_atomically_unique(tmp_path):
    manager = SessionManager(tmp_path)

    def create_one(_number: int) -> str:
        return manager.new_store()[0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(create_one, range(16)))

    assert len(set(ids)) == 16
    assert all((manager.dir / f"{session_id}.jsonl").is_file() for session_id in ids)


def test_checkpoint_name_is_unique_across_processes(tmp_path):
    path = tmp_path / "s.jsonl"
    SessionStore(path).append_message(user_message("base"))
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_checkpoint_process_worker,
            args=(str(path), start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0

    outcomes = sorted(results.get(timeout=2) for _ in processes)
    assert outcomes == ["created", "exists"]
    assert [info.name for info in SessionStore(path).list_checkpoints()] == [
        "shared"
    ]


def test_writer_lease_rejects_second_process(tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    SessionStore(path).append_message(user_message("base"))
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_writer_lease_process_worker,
        args=(str(path), ready, release),
    )
    holder.start()
    assert ready.wait(timeout=10)
    monkeypatch.setattr(session_store_module, "_LOCK_TIMEOUT_SECONDS", 0.2)

    with pytest.raises(SessionBusyError, match="另一个进程"):
        SessionStore(path).acquire_writer()

    release.set()
    holder.join(timeout=10)
    assert not holder.is_alive()
    assert holder.exitcode == 0


def test_session_hardlink_is_rejected_for_read_and_append(tmp_path):
    manager = SessionManager(tmp_path)
    outside = tmp_path / "outside.jsonl"
    SessionStore(outside).append_message(user_message("secret"))
    linked = manager.dir / "20300102-030405.jsonl"
    try:
        session_store_module.os.link(outside, linked)
    except OSError:
        pytest.skip("hard links are unavailable on this filesystem")

    with pytest.raises(UnsafePathError, match="硬链接"):
        SessionStore.load_messages(linked)
    with pytest.raises(UnsafePathError, match="硬链接"):
        SessionStore(linked).append_message(user_message("overwrite"))
    assert manager.list_sessions() == []


def test_fork_from_checkpoint_persists_lineage_and_isolation(tmp_path):
    manager = SessionManager(tmp_path)
    parent_id, parent_store = manager.new_store()
    base = [user_message("base")]
    parent_store.append_message(base[0])
    parent_store.append_checkpoint("before-fix", base)
    parent_store.append_message(
        Message(role="assistant", content=[TextBlock(text="parent later")])
    )
    parent_before = parent_store.path.read_bytes()

    child_id, child_store, child_messages = manager.fork(
        parent_id, checkpoint_name="before-fix"
    )

    assert parent_store.path.read_bytes() == parent_before
    assert child_messages == base
    assert child_messages[0] is not base[0]
    lineage = SessionStore.load_lineage(child_store.path)
    assert lineage is not None
    assert lineage.parent_session_id == parent_id
    assert lineage.checkpoint_name == "before-fix"
    listed_child = next(
        info for info in manager.list_sessions() if info.session_id == child_id
    )
    assert listed_child.parent_session_id == parent_id
    assert listed_child.checkpoint_name == "before-fix"

    child_messages[0].content.append(TextBlock(text="memory only"))
    assert SessionStore.load_messages(child_store.path) == base
    child_store.append_message(user_message("child only"))
    parent_store.append_message(user_message("parent only"))
    assert SessionStore.load_messages(child_store.path)[-1].text() == "child only"
    assert SessionStore.load_messages(parent_store.path)[-1].text() == "parent only"


def test_fork_current_context_repairs_dangling_in_child_only(tmp_path):
    manager = SessionManager(tmp_path)
    parent_id, parent_store = manager.new_store()
    current = [
        user_message("inspect"),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="pending", name="read_file", input={})],
        ),
    ]
    for message in current:
        parent_store.append_message(message)
    parent_before = parent_store.path.read_bytes()

    _child_id, child_store, child_messages = manager.fork(parent_id, current)

    assert parent_store.path.read_bytes() == parent_before
    assert SessionStore.load_messages(parent_store.path) == current
    assert len(child_messages) == 3
    repair = child_messages[-1].content[0]
    assert isinstance(repair, ToolResultBlock)
    assert repair.tool_use_id == "pending"
    assert repair.is_error
    assert SessionStore.load_messages(child_store.path) == child_messages


def test_fork_missing_checkpoint_has_no_side_effect(tmp_path):
    manager = SessionManager(tmp_path)
    parent_id, parent_store = manager.new_store()
    parent_store.append_message(user_message("parent"))
    before_files = {path.name: path.read_bytes() for path in manager.dir.glob("*.jsonl")}

    with pytest.raises(KeyError, match="不存在"):
        manager.fork(parent_id, checkpoint_name="missing")

    after_files = {path.name: path.read_bytes() for path in manager.dir.glob("*.jsonl")}
    assert after_files == before_files


def test_session_manager_rejects_linklike_sessions_root(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (project / "sessions").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(UnsafePathError):
        SessionManager(project)


def test_new_store_skips_symlink_id_collision(tmp_path, monkeypatch):
    class FrozenDateTime:
        @classmethod
        def now(cls):
            return RealDateTime(2030, 1, 2, 3, 4, 5)

    monkeypatch.setattr(session_manager_module, "datetime", FrozenDateTime)
    manager = SessionManager(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("untouched", encoding="utf-8")
    collision = manager.dir / "20300102-030405.jsonl"
    try:
        collision.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    session_id, _store = manager.new_store()

    assert session_id == "20300102-030405-1"
    assert outside.read_text(encoding="utf-8") == "untouched"


def test_fork_write_failure_removes_partial_child(tmp_path, monkeypatch):
    manager = SessionManager(tmp_path)
    parent_id, parent_store = manager.new_store()
    parent_store.append_message(user_message("parent"))
    before = {path.name for path in manager.dir.glob("*.jsonl")}

    def fail_publish(_source, _target, *, follow_symlinks):
        raise OSError("disk full")

    monkeypatch.setattr(session_store_module.os, "link", fail_publish)

    with pytest.raises(OSError, match="disk full"):
        manager.fork(parent_id, [user_message("fork")])

    assert {path.name for path in manager.dir.glob("*.jsonl")} == before
    assert list(manager.dir.glob("*.tmp")) == []


def test_fork_fsync_failure_does_not_publish_child(tmp_path, monkeypatch):
    manager = SessionManager(tmp_path)
    parent_id, parent_store = manager.new_store()
    parent_store.append_message(user_message("parent"))
    before = {path.name for path in manager.dir.glob("*.jsonl")}

    def fail_fsync(_fd):
        raise PermissionError("temporarily unwritable")

    monkeypatch.setattr(session_store_module.os, "fsync", fail_fsync)

    with pytest.raises(PermissionError, match="unwritable"):
        manager.fork(parent_id, [user_message("fork")])

    assert {path.name for path in manager.dir.glob("*.jsonl")} == before
    assert list(manager.dir.glob("*.tmp")) == []


def test_atomic_publish_temp_unlink_failure_rolls_back_target(
    tmp_path, monkeypatch
):
    manager = SessionManager(tmp_path)
    parent_id, parent_store = manager.new_store()
    parent_store.append_message(user_message("parent"))
    before = {path.name for path in manager.dir.glob("*.jsonl")}
    original_unlink = Path.unlink
    failed_once = False

    def fail_first_temp_unlink(self, *args, **kwargs):
        nonlocal failed_once
        if self.suffix == ".tmp" and not failed_once:
            failed_once = True
            raise PermissionError("scanner holds temp link")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_temp_unlink)

    with pytest.raises(OSError, match="最终名称已回滚"):
        manager.fork(parent_id, [user_message("fork")])

    assert {path.name for path in manager.dir.glob("*.jsonl")} == before
    assert list(manager.dir.glob("*.tmp")) == []


def test_protocol_repair_handles_missing_middle_tool_result():
    original = [
        user_message("inspect"),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="t1", name="read_file", input={})],
        ),
        Message(role="assistant", content=[TextBlock(text="final")]),
    ]

    repaired, changed = repaired_message_copy(original, "recovered")

    assert changed
    assert len(repaired) == 4
    result_message = repaired[2]
    assert result_message.role == "user"
    result = result_message.content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.tool_use_id == "t1"
    assert result.is_error
    assert repaired[-1].text() == "final"


@pytest.mark.asyncio
async def test_resume_persists_middle_protocol_repair_as_snapshot(
    tmp_path, quiet_console
):
    manager = SessionManager(tmp_path)
    target_id, target_store = manager.new_store()
    target_store.append_message(user_message("inspect"))
    target_store.append_message(
        Message(
            role="assistant",
            content=[ToolUseBlock(id="t1", name="read_file", input={})],
        )
    )
    with target_store.path.open("ab") as file:
        file.write(b'{"type":"message","data":broken}\n')
    target_store.append_message(
        Message(role="assistant", content=[TextBlock(text="final")])
    )
    current_id, current_store = manager.new_store()
    current_messages = [user_message("current")]
    current_store.append_message(current_messages[0])

    repl = object.__new__(Repl)
    repl.session_mgr = manager
    repl.session_id = current_id
    repl.store = current_store
    repl.messages = current_messages
    repl.console = quiet_console

    await commands._resume(repl, target_id)

    assert repl.session_id == target_id
    assert len(repl.messages) == 4
    assert isinstance(repl.messages[2].content[0], ToolResultBlock)
    assert SessionStore.load_messages(target_store.path) == repl.messages
    repl.store.release_writer()


@pytest.mark.asyncio
async def test_resume_read_failure_keeps_current_state(
    tmp_path, quiet_console, monkeypatch
):
    manager = SessionManager(tmp_path)
    target_id, target_store = manager.new_store()
    target_store.append_message(user_message("target"))
    current_id, current_store = manager.new_store()
    current_messages = [user_message("current")]
    current_store.append_message(current_messages[0])
    original_load = SessionStore.load_messages

    def fail_target(path):
        if path == target_store.path:
            raise PermissionError("read denied")
        return original_load(path)

    monkeypatch.setattr(SessionStore, "load_messages", staticmethod(fail_target))
    repl = object.__new__(Repl)
    repl.session_mgr = manager
    repl.session_id = current_id
    repl.store = current_store
    repl.messages = current_messages
    repl.console = quiet_console

    await commands._resume(repl, target_id)

    assert repl.session_id == current_id
    assert repl.store is current_store
    assert repl.messages is current_messages
    assert "当前会话未改变" in quiet_console.file.getvalue()


@pytest.mark.asyncio
async def test_resume_repair_write_failure_keeps_current_state(
    tmp_path, quiet_console, monkeypatch
):
    manager = SessionManager(tmp_path)
    target_id, target_store = manager.new_store()
    target_store.append_message(user_message("target"))
    target_store.append_message(
        Message(
            role="assistant",
            content=[ToolUseBlock(id="pending", name="read_file", input={})],
        )
    )
    target_before = target_store.path.read_bytes()
    current_id, current_store = manager.new_store()
    current_messages = [user_message("current")]
    current_store.append_message(current_messages[0])
    original_snapshot = SessionStore.append_snapshot

    def fail_target(self, messages):
        if self.path == target_store.path:
            raise OSError("disk full")
        return original_snapshot(self, messages)

    monkeypatch.setattr(SessionStore, "append_snapshot", fail_target)
    repl = object.__new__(Repl)
    repl.session_mgr = manager
    repl.session_id = current_id
    repl.store = current_store
    repl.messages = current_messages
    repl.console = quiet_console

    await commands._resume(repl, target_id)

    assert repl.session_id == current_id
    assert repl.store is current_store
    assert repl.messages is current_messages
    assert target_store.path.read_bytes() == target_before


@pytest.mark.asyncio
async def test_clear_write_failure_keeps_memory_context(
    tmp_path, quiet_console, monkeypatch
):
    store = SessionStore(tmp_path / "s.jsonl")
    messages = [user_message("keep")]
    store.append_message(messages[0])

    def fail_snapshot(_messages):
        raise OSError("disk full")

    monkeypatch.setattr(store, "append_snapshot", fail_snapshot)
    repl = object.__new__(Repl)
    repl.store = store
    repl.messages = messages
    repl.console = quiet_console

    await commands._clear(repl, "")

    assert repl.messages == messages
    assert SessionStore.load_messages(store.path) == messages


def test_explicit_resume_does_not_depend_on_session_list_limit(
    tmp_path, monkeypatch
):
    manager = SessionManager(tmp_path)
    session_id, store = manager.new_store()
    store.append_message(user_message("direct"))

    def fail_list(*_args, **_kwargs):
        raise AssertionError("explicit resume must not enumerate sessions")

    monkeypatch.setattr(manager, "list_sessions", fail_list)

    resumed = manager.resume(session_id)
    assert resumed is not None
    assert resumed[2] == [user_message("direct")]


@pytest.mark.asyncio
async def test_checkpoint_and_fork_commands_switch_only_after_success(
    tmp_path, quiet_console
):
    manager = SessionManager(tmp_path)
    parent_id, parent_store = manager.new_store()
    messages = [user_message("current")]
    parent_store.append_message(messages[0])
    repl = object.__new__(Repl)
    repl.session_mgr = manager
    repl.session_id = parent_id
    repl.store = parent_store
    repl.messages = messages
    repl.console = quiet_console

    assert await commands._checkpoint(repl, "stable") is False
    parent_after_checkpoint = parent_store.path.read_bytes()
    assert await commands._checkpoints(repl, "") is False
    assert "stable" in quiet_console.file.getvalue()

    assert await commands._fork(repl, "missing") is False
    assert repl.session_id == parent_id
    assert repl.store is parent_store
    assert repl.messages is messages
    assert parent_store.path.read_bytes() == parent_after_checkpoint

    assert await commands._fork(repl, "stable") is False
    assert repl.session_id != parent_id
    assert repl.store is not parent_store
    assert repl.messages == messages
    assert parent_store.path.read_bytes() == parent_after_checkpoint
    repl._persist(user_message("child continuation"))
    assert SessionStore.load_messages(parent_store.path) == messages
    assert SessionStore.load_messages(repl.store.path)[-1].text() == "child continuation"
    repl.store.release_writer()
