"""模型创建工厂。

根据 settings.llm.provider 判断模型类型：
- provider == "deepseek" → ChatDeepSeek（支持 reasoning_content）
- 其他 → ChatOpenAI（通用 OpenAI 兼容端点）
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_settings

SUMMARY_PROMPT_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts" / "summary_prompt.md"


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


def _load_summary_prompt() -> str:
    return SUMMARY_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _build_summary_system_prompt(*, depth: str, file_name: str, file_type: str) -> str:
    summary_prompt = _load_summary_prompt()
    return (
        f"{summary_prompt}\n\n"
        f"当前文件名：{file_name}\n"
        f"当前文件类型：{file_type}\n"
        f"当前预览深度：{depth}"
    )


def summarize_visual_assets(
    assets: Sequence[object],
    *,
    depth: str,
    file_name: str,
    file_type: str,
) -> str:
    if not assets:
        raise ValueError("没有可用于图片预览的图像资产。")

    model = create_chat_model()
    content: list[dict[str, object]] = []

    for asset in assets:
        image_bytes = getattr(asset, "image_bytes", None)
        mime_type = getattr(asset, "mime_type", "image/png")
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("存在无法用于图片预览的空图像资产。")
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )

    try:
        response = model.invoke(
            [
                SystemMessage(
                    content=_build_summary_system_prompt(
                        depth=depth,
                        file_name=file_name,
                        file_type=file_type,
                    )
                ),
                HumanMessage(content=content),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"视觉摘要调用失败：{exc}") from exc

    text = _coerce_response_text(response.content)
    if not text:
        raise ValueError("图片预览模型返回空结果。")
    return f"由视觉模型查看生成：{text}"


def _coerce_response_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    return str(content).strip()
