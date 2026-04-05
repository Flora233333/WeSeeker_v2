from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent.factory import create_weseeker_agent


green = "\033[92m"
reset = "\033[0m"


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
        self._messages: list[BaseMessage] = []

    async def initialize(self) -> None:
        self._agent, self._mcp_client = await create_weseeker_agent()

    async def process_message(self, user_input: str) -> str:
        response = await self.process_message_with_trace(user_input)
        return response.reply

    async def process_message_with_trace(self, user_input: str) -> AgentResponse:
        if self._agent is None:
            raise RuntimeError("Agent 尚未初始化。")

        current_messages = [*self._messages, HumanMessage(content=user_input)]
        result = await self._agent.ainvoke(
            {"messages": current_messages},
            config={"configurable": {"thread_id": self._thread_id}},
        )
        all_messages = self._extract_messages(result)
        
        # print('------------------------------')
        # print('all_messages:')
        # for i, msg in enumerate(all_messages, 1):
        #     print(f'[{i}] type={green}{type(msg).__name__}{reset}, msg={msg}')
        # print('------------------------------')

        new_messages = self._extract_new_messages(self._messages, all_messages)
        self._messages = all_messages

        return AgentResponse(
            reply=self._extract_reply(new_messages),
            tool_traces=self._extract_tool_traces(new_messages),
        )

    async def new_conversation(self) -> None:
        self._thread_id = str(uuid.uuid4())
        self._messages = []

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
    def _extract_messages(result: object) -> list[BaseMessage]:
        if not isinstance(result, dict):
            return []

        messages = result.get("messages", [])
        return [message for message in messages if isinstance(message, BaseMessage)]

    @staticmethod
    def _extract_new_messages(
        previous_messages: list[BaseMessage],
        all_messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        previous_count = len(previous_messages)
        if len(all_messages) < previous_count:
            return all_messages
        return all_messages[previous_count:]

    @staticmethod
    def _extract_reply(messages: list[BaseMessage]) -> str:
        ai_messages = [message for message in messages if isinstance(message, AIMessage)]
        if ai_messages:
            content = ai_messages[-1].content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(part) for part in content)
        return "抱歉，我暂时没有拿到有效结果。"

    @staticmethod
    def _extract_tool_traces(messages: list[BaseMessage]) -> list[ToolTrace]:
        tool_results_by_id: dict[str, ToolMessage] = {}

        for message in messages:
            if isinstance(message, ToolMessage):
                tool_call_id = getattr(message, "tool_call_id", None)
                if tool_call_id:
                    tool_results_by_id[tool_call_id] = message

        traces: list[ToolTrace] = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue

            for tool_call in getattr(message, "tool_calls", []) or []:
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
    def _stringify_tool_result(tool_name: str, tool_message: ToolMessage | None) -> str:
        if tool_message is None:
            return "<未捕获到工具返回结果>"

        text = AgentRunner._extract_tool_text(tool_message)

        if tool_name == "search_files":
            summary = AgentRunner._summarize_search_result(text)
            if summary:
                return summary
        if tool_name == "read_file_content":
            summary = AgentRunner._summarize_read_result(text)
            if summary:
                return summary

        text = text.replace("\r", " ").replace("\n", " ").strip()
        if len(text) > 240:
            return f"{text[:240]}..."
        return text

    @staticmethod
    def _extract_tool_text(tool_message: ToolMessage) -> str:
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
            error_type = payload.get("error_type")
            message = payload.get("message") or payload.get("user_hint") or payload.get("error") or "未知错误"
            if error_type:
                return f"搜索失败[{error_type}]: {message}"
            return f"搜索失败: {message}"

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
            size_display = item.get("size_display")
            size = item.get("size")
            lines.append(f"- {name}")
            if full_path:
                lines.append(f"  {full_path}")
            if size_display:
                lines.append(f"  size={size_display}")
            elif size is not None:
                lines.append(f"  size={size}")

        if count > 3:
            lines.append(f"- ... 还有 {count - 3} 项")

        return "\n".join(lines)

    @staticmethod
    def _summarize_read_result(text: str) -> str | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        if not payload.get("ok"):
            error_type = payload.get("error_type")
            message = payload.get("message") or payload.get("user_hint") or payload.get("error") or "未知错误"
            if error_type:
                return f"预览失败[{error_type}]: {message}"
            return f"预览失败: {message}"

        file_name = payload.get("file_name", "<unknown>")
        file_type = payload.get("file_type", "<unknown>")
        depth = payload.get("depth", "L1")
        metadata = payload.get("metadata", {})
        preview_text = str(payload.get("preview_text", "")).replace("\r", " ").replace("\n", " ").strip()

        lines = [f"读取成功 | file={file_name} | type={file_type} | depth={depth}"]
        if isinstance(metadata, dict):
            size_display = metadata.get("size_display")
            size = metadata.get("size")
            modified = metadata.get("modified")
            if size_display:
                lines.append(f"- size={size_display}")
            elif size is not None:
                lines.append(f"- size={size}")
            if modified:
                lines.append(f"- modified={modified}")
        if preview_text:
            if len(preview_text) > 120:
                preview_text = f"{preview_text[:120]}..."
            lines.append(f"- preview={preview_text}")

        return "\n".join(lines)
