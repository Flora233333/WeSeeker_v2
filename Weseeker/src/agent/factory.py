from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent

from adapters.mcp_client import create_mcp_client
from adapters.model_provider import create_chat_model


PROMPT_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts" / "system_prompt.md"


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


async def create_weseeker_agent():
    model = create_chat_model()
    mcp_client = create_mcp_client()
    tools = await mcp_client.get_tools()
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=_load_system_prompt(),
    )
    return agent, mcp_client
