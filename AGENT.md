# EmiyaAgent 项目指令

Python 3.12+，uv 管理（`uv sync` / `uv run pytest` / `uv run emiya`）。

架构铁律：

- 内部消息格式以 `emiya_agent/types.py` 的 content-block 模型为唯一真源；OpenAI/DeepSeek 格式转换只允许出现在 `providers/openai_compat.py`。
- 系统提示词会话内字节级稳定（禁止时间戳/每轮变化内容）；工具列表按名称排序。
- 工具循环不变量：每个 tool_use 必有配对 tool_result（中断时补 is_error 合成结果）；并行结果合入同一条 user 消息。
- fable-5 不传 thinking 参数；所有现役 Claude 模型不传 temperature/top_p/top_k。
- Windows 保持 Proactor 事件循环；子进程 finally 中 terminate+kill。
- 运行时数据目录一律 `.emiyacode`（用户级 `~/.emiyacode/`，项目级 `./.emiyacode/`）。
