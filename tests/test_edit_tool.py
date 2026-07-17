"""edit_file 精确替换语义：唯一匹配、CRLF 保留、Unicode。"""

import pytest

from emiya_agent.tools.base import ToolContext
from emiya_agent.tools.edit import EditFileTool

from tests.conftest import FakeUI


@pytest.fixture
def ctx(tmp_path, config):
    return ToolContext(cwd=tmp_path, config=config, ui=FakeUI())


@pytest.fixture
def tool():
    return EditFileTool()


async def test_simple_replace(tmp_path, ctx, tool):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n", encoding="utf-8")
    result = await tool.run(
        {"path": "a.py", "old_string": "y = 2", "new_string": "y = 20"}, ctx
    )
    assert not result.is_error
    assert f.read_text(encoding="utf-8") == "x = 1\ny = 20\n"


async def test_zero_match_errors(tmp_path, ctx, tool):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = await tool.run(
        {"path": "a.py", "old_string": "nope", "new_string": "yes"}, ctx
    )
    assert result.is_error
    assert "未在文件中找到" in result.content


async def test_multi_match_errors_without_replace_all(tmp_path, ctx, tool):
    (tmp_path / "a.py").write_text("a\na\n", encoding="utf-8")
    result = await tool.run({"path": "a.py", "old_string": "a", "new_string": "b"}, ctx)
    assert result.is_error
    assert "2 次" in result.content


async def test_replace_all(tmp_path, ctx, tool):
    f = tmp_path / "a.py"
    f.write_text("a\na\n", encoding="utf-8")
    result = await tool.run(
        {"path": "a.py", "old_string": "a", "new_string": "b", "replace_all": True},
        ctx,
    )
    assert not result.is_error
    assert f.read_text(encoding="utf-8") == "b\nb\n"


async def test_crlf_preserved(tmp_path, ctx, tool):
    f = tmp_path / "win.txt"
    f.write_bytes(b"line1\r\nline2\r\nline3\r\n")
    result = await tool.run(
        {"path": "win.txt", "old_string": "line2", "new_string": "LINE2"}, ctx
    )
    assert not result.is_error
    assert f.read_bytes() == b"line1\r\nLINE2\r\nline3\r\n"


async def test_multiline_old_string_with_crlf_file(tmp_path, ctx, tool):
    f = tmp_path / "win.py"
    f.write_bytes(b"def foo():\r\n    return 1\r\n")
    # 模型给的 old_string 通常是 LF 风格，必须也能匹配 CRLF 文件
    result = await tool.run(
        {
            "path": "win.py",
            "old_string": "def foo():\n    return 1",
            "new_string": "def foo():\n    return 2",
        },
        ctx,
    )
    assert not result.is_error
    assert f.read_bytes() == b"def foo():\r\n    return 2\r\n"


async def test_unicode_content(tmp_path, ctx, tool):
    f = tmp_path / "zh.md"
    f.write_text("# 标题\n内容甲\n", encoding="utf-8")
    result = await tool.run(
        {"path": "zh.md", "old_string": "内容甲", "new_string": "内容乙"}, ctx
    )
    assert not result.is_error
    assert "内容乙" in f.read_text(encoding="utf-8")


async def test_identical_strings_error(tmp_path, ctx, tool):
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    result = await tool.run({"path": "a.py", "old_string": "x", "new_string": "x"}, ctx)
    assert result.is_error


async def test_missing_file_error(ctx, tool):
    result = await tool.run(
        {"path": "missing.py", "old_string": "a", "new_string": "b"}, ctx
    )
    assert result.is_error
