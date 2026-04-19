"""Agent 工厂。

根据 settings.llm.provider 自动选择模型和中间件组合。
"""

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from adapters.mcp_client import create_mcp_client
from adapters.model_provider import create_chat_model
from config.settings import get_settings
from middleware.message_normalizer import (
    DeepSeekReasoningMiddleware,
    ToolContentNormalizerMiddleware,
)
from middleware.preview_summary import PreviewSummaryMiddleware
from middleware.tool_call_limit import ToolCallLimitMiddleware

PROMPT_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts" / "system_prompt.md"
INTERNAL_TOOL_NAMES = {"clear_client_state"}


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _build_middleware() -> list:
    """根据 provider 构建中间件链。"""
    settings = get_settings()
    provider = (settings.llm.provider or "").lower().strip()
    debug = settings.debug  # 统一用全局 debug 开关

    middleware = []

    # 通用：ToolMessage list content → str（所有模型都需要）
    middleware.append(ToolContentNormalizerMiddleware(debug=False))

    # read_file_content 工具返回后，在 Agent 侧生成 summary 并永久替换结果
    middleware.append(PreviewSummaryMiddleware())

    # DeepSeek 专用：reasoning_content 管理
    if provider == "deepseek":
        middleware.append(DeepSeekReasoningMiddleware(debug=debug))

    # 工具调用次数限制
    middleware.append(ToolCallLimitMiddleware(run_limit=10))

    return middleware


def _filter_agent_tools(tools: list[BaseTool]) -> list[BaseTool]:
    return [tool for tool in tools if tool.name not in INTERNAL_TOOL_NAMES]
async def create_weseeker_agent(
    *,
    mcp_client: MultiServerMCPClient | None = None,
    thread_id: str | None = None,
):
    model = create_chat_model()
    effective_mcp_client = mcp_client or create_mcp_client(thread_id=thread_id)
    tools = _filter_agent_tools(await effective_mcp_client.get_tools())
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=_load_system_prompt(),
        middleware=_build_middleware(),
    )
    return agent, effective_mcp_client
