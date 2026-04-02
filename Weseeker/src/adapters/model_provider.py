from __future__ import annotations

from langchain_openai import ChatOpenAI

from config.settings import get_settings


def create_chat_model() -> ChatOpenAI:
    settings = get_settings()
    if not settings.llm.api_key:
        raise ValueError("未找到 LLM API Key，请在 .env 中配置 DASHSCOPE_API_KEY")

    return ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        api_key=settings.llm.api_key,
        base_url=settings.llm.api_base,
        timeout=settings.llm.timeout,
    )
