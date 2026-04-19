"""模型创建工厂。

根据 settings.llm.provider 判断模型类型：
- provider == "deepseek" → ChatDeepSeek（支持 reasoning_content）
- 其他 → ChatOpenAI（通用 OpenAI 兼容端点）
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from config.settings import get_settings


def create_chat_model() -> BaseChatModel:
    """根据配置创建主对话模型。"""
    settings = get_settings()
    if not settings.llm.api_key:
        raise ValueError("未找到 LLM API Key，请在 .env 中配置对应的 API Key")

    provider = (settings.llm.provider or "").lower().strip()

    if provider == "deepseek":
        return _create_deepseek_model()
    else:
        return _create_openai_compatible_model()


def create_summary_model() -> BaseChatModel:
    """创建预览摘要专用模型。

    当前与主对话模型共用同一套 provider/model 配置，但调用链与
    Agent middleware 解耦，只保留最轻量的直接 `model.invoke(...)` 用法。
    """

    return create_chat_model()


def _create_deepseek_model() -> BaseChatModel:
    """创建 DeepSeek 模型。

    ChatDeepSeek 会自动解析 reasoning_content 并放入 additional_kwargs。

    注意：thinking mode 下 max_tokens 控制 CoT + 最终回答的总长度，
    默认 32K，最大 64K。不要随意缩小，否则 reasoning 会被截断。
    """
    from langchain_deepseek import ChatDeepSeek

    settings = get_settings()

    return ChatDeepSeek(
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        # deepseek-reasoner 不支持 temperature/top_p，设了不报错但没效果
        temperature=settings.llm.temperature,
        # 不设 max_tokens，让 DeepSeek 用默认的 32K
        timeout=settings.llm.timeout,
    )


def _create_openai_compatible_model() -> BaseChatModel:
    """创建 OpenAI 兼容模型（DashScope/Kimi/GLM/LMStudio/Ollama 等）。"""
    from langchain_openai import ChatOpenAI

    settings = get_settings()

    extra_body: dict = {}
    if settings.llm.enable_thinking is not None:
        extra_body["enable_thinking"] = settings.llm.enable_thinking

    return ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        api_key=settings.llm.api_key,
        base_url=settings.llm.api_base,
        timeout=settings.llm.timeout,
        model_kwargs={"extra_body": extra_body} if extra_body else {},
    )
