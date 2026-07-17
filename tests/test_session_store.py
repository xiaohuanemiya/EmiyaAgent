"""会话 JSONL 持久化：写入→回放字节等价、损坏行容忍、恢复流程。"""

from emiya_agent.session.manager import SessionManager
from emiya_agent.session.store import SessionStore
from emiya_agent.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    user_message,
)


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


def test_corrupt_tail_tolerated(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    store.append_message(user_message("ok"))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"type": "message", "data": {"role": "user"')  # 崩溃写一半

    loaded = SessionStore.load_messages(path)
    assert len(loaded) == 1


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
