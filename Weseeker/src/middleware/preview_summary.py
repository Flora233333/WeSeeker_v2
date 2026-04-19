from __future__ import annotations

import asyncio
import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger

from adapters.preview_summarizer import summarize_read_preview_payload


class PreviewSummaryMiddleware(AgentMiddleware):
    """在 read_file_content 返回后生成 summary，并永久替换 ToolMessage。"""

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if isinstance(result, Command):
            return result
        if not isinstance(result, ToolMessage):
            return result
        if request.tool_call.get("name") != "read_file_content":
            return result

        payload = _parse_tool_payload(result)
        if not isinstance(payload, dict) or not payload.get("ok"):
            return result

        try:
            rewritten_payload = await asyncio.to_thread(summarize_read_preview_payload, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"read_file_content 摘要替换失败，回退原始工具结果: {exc}")
            return result

        return ToolMessage(
            content=json.dumps(rewritten_payload, ensure_ascii=False),
            tool_call_id=result.tool_call_id,
            name=result.name,
            id=result.id,
            artifact=result.artifact,
            additional_kwargs=dict(result.additional_kwargs or {}),
            status=result.status,
        )


def _parse_tool_payload(message: ToolMessage) -> dict[str, object] | None:
    text = _extract_tool_text(message)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_tool_text(message: ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)
