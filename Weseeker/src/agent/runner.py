from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.factory import create_weseeker_agent


@dataclass
class ToolTrace:
    tool_name: str
    args: dict[str, Any]
    result_preview: str


@dataclass
class AgentResponse:
    reply: str
    tool_traces: list[ToolTrace]


class AgentRunner:
    def __init__(self) -> None:
        self._agent = None
        self._mcp_client = None
        self._thread_id = str(uuid.uuid4())

    async def initialize(self) -> None:
        self._agent, self._mcp_client = await create_weseeker_agent()

    async def process_message(self, user_input: str) -> str:
        response = await self.process_message_with_trace(user_input)
        return response.reply

    async def process_message_with_trace(self, user_input: str) -> AgentResponse:
        if self._agent is None:
            raise RuntimeError("Agent 尚未初始化。")

        result = await self._agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": self._thread_id}}, 
            # LangGraph/LangChain 语境里的“会话标识” 不是多线程的意思
            # 通常它的用途是：
            # - 标识同一个对话线程
            # - 配合 checkpointer 做多轮状态持久化
            # - 支持 interrupt()/resume() 恢复同一会话
        )
        return AgentResponse(
            reply=self._extract_reply(result),
            tool_traces=self._extract_tool_traces(result),
        )

    async def new_conversation(self) -> None:
        self._thread_id = str(uuid.uuid4())

    async def cleanup(self) -> None:
        if self._mcp_client is not None:
            close = getattr(self._mcp_client, "close", None)
            aclose = getattr(self._mcp_client, "aclose", None)
            if callable(aclose):
                await aclose()
            elif callable(close):
                result = close()
                if hasattr(result, "__await__"):
                    await result

    @staticmethod
    def _extract_reply(result) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        ai_messages = [message for message in messages if isinstance(message, AIMessage)]
        if ai_messages:
            content = ai_messages[-1].content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(part) for part in content)
        return "抱歉，我暂时没有拿到有效结果。"

    @staticmethod
    def _extract_tool_traces(result) -> list[ToolTrace]:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        tool_results_by_id: dict[str, ToolMessage] = {}

        for message in messages:
            if isinstance(message, ToolMessage): # ToolMessage本身就是工具作用后产生的消息结果
                tool_call_id = getattr(message, "tool_call_id", None)
                if tool_call_id:
                    tool_results_by_id[tool_call_id] = message

        traces: list[ToolTrace] = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue

            for tool_call in getattr(message, "tool_calls", []) or []: # = for tool_call in tool_calls
                tool_result = tool_results_by_id.get(tool_call.get("id", ""))
                traces.append(
                    ToolTrace(
                        tool_name=tool_call.get("name", "unknown"),
                        args=tool_call.get("args", {}),
                        result_preview=AgentRunner._stringify_tool_result(
                            tool_call.get("name", "unknown"),
                            tool_result,
                        ),
                    )
                )

        return traces

    @staticmethod
    def _stringify_tool_result(tool_name: str, tool_message: ToolMessage | None) -> str: # 负责把工具原始返回值变成“可展示字符串”
        if tool_message is None:
            return "<未捕获到工具返回结果>"

        text = AgentRunner._extract_tool_text(tool_message)

        if tool_name == "search_files":
            summary = AgentRunner._summarize_search_result(text)
            if summary:
                return summary

        text = text.replace("\r", " ").replace("\n", " ").strip()
        if len(text) > 240:
            return f"{text[:240]}..."
        return text

    @staticmethod
    def _extract_tool_text(tool_message: ToolMessage) -> str: # 负责把 ToolMessage.content 统一转成字符串
        content = tool_message.content
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                else:
                    text_parts.append(str(item))
            return "\n".join(part for part in text_parts if part)

        return str(content)

    @staticmethod
    def _summarize_search_result(text: str) -> str | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        if not payload.get("ok"):
            return f"搜索失败: {payload.get('error', '未知错误')}"

        keyword = payload.get("keyword", "")
        path = payload.get("path") or "全局"
        count = payload.get("count", 0)
        results = payload.get("results", [])

        lines = [f"命中 {count} 项 | keyword={keyword} | path={path}"]
        for item in results[:3]:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "<unknown>")
            full_path = item.get("full_path") or item.get("path") or ""
            lines.append(f"- {name}")
            if full_path:
                lines.append(f"  {full_path}")

        if count > 3:
            lines.append(f"- ... 还有 {count - 3} 项")

        return "\n".join(lines)
