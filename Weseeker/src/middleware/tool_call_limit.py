from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langchain.agents.middleware.types import AgentMiddleware

_LIMIT_MSG = (
    "你已经调用了 {count} 次工具，已达到本轮上限。"
    "请立即停止继续调用工具，用自然语言总结你已经尝试过的搜索策略，"
    "并引导用户补充更多线索（例如：文件类型、大概存放位置、修改时间、文件名关键词等）。"
)


class ToolCallLimitMiddleware(AgentMiddleware):
    """限制单轮对话中的工具调用次数。

    超出限制时向 state 注入一条 SystemMessage，指示 LLM 停止搜索并引导用户补充信息。
    注入的消息会进入历史记录，LLM 在下一次 model 调用时能看到它。
    """

    def __init__(self, *, run_limit: int = 5) -> None:
        super().__init__()
        self.run_limit = run_limit

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        count = _count_tools_since_last_human(messages)
        if count >= self.run_limit:
            return {"messages": [SystemMessage(content=_LIMIT_MSG.format(count=count))]}
        return None

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


def _count_tools_since_last_human(messages: list[Any]) -> int:
    last_human_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            last_human_index = index
            break

    window = messages[last_human_index + 1 :] if last_human_index >= 0 else messages
    return sum(1 for message in window if isinstance(message, ToolMessage))
