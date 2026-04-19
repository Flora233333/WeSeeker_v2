from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import fitz
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from PIL import Image

from adapters.model_provider import create_summary_model
from config.settings import get_settings

INTERNAL_PREVIEW_SUMMARY_TAG = "internal_preview_summary"
INTERNAL_PREVIEW_SUMMARY_CONFIG = {
    "run_name": INTERNAL_PREVIEW_SUMMARY_TAG,
    "tags": [INTERNAL_PREVIEW_SUMMARY_TAG],
    "metadata": {INTERNAL_PREVIEW_SUMMARY_TAG: True},
}

SUMMARY_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "prompts" / "summary_prompt.md"
)
IMAGE_FILE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
MIME_TYPE_BY_SUFFIX = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class SummaryVisualAsset:
    mime_type: str
    image_bytes: bytes


def summarize_read_preview_payload(payload: dict[str, object]) -> dict[str, object]:
    """把 read_file_content 的原始预览结果改写成 summary 版本。"""

    if not payload.get("ok"):
        return payload

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return payload

    file_name = str(payload.get("file_name") or "")
    file_type = str(payload.get("file_type") or "")
    depth = str(payload.get("depth") or "L1")
    file_path = Path(str(payload.get("file_path") or ""))
    preview_text = str(payload.get("preview_text") or "")
    preview_mode = str(metadata.get("preview_mode") or "")

    summary_text = _summarize_preview(
        file_name=file_name,
        file_type=file_type,
        depth=depth,
        file_path=file_path,
        preview_text=preview_text,
        preview_mode=preview_mode,
    )

    rewritten = dict(payload)
    rewritten["preview_text"] = summary_text
    return rewritten


def _summarize_preview(
    *,
    file_name: str,
    file_type: str,
    depth: str,
    file_path: Path,
    preview_text: str,
    preview_mode: str,
) -> str:
    if preview_mode in {"image_visual_source", "pdf_visual_source"}:
        try:
            return _summarize_visual_source(
                file_name=file_name,
                file_type=file_type,
                depth=depth,
                file_path=file_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"视觉预览总结失败，将回退文本总结: {exc}")
            if preview_text.strip():
                return _summarize_text_source(
                    file_name=file_name,
                    file_type=file_type,
                    depth=depth,
                    preview_text=preview_text,
                )
            raise

    if preview_text.strip():
        return _summarize_text_source(
            file_name=file_name,
            file_type=file_type,
            depth=depth,
            preview_text=preview_text,
        )

    return preview_text


def _summarize_text_source(*, file_name: str, file_type: str, depth: str, preview_text: str) -> str:
    model = create_summary_model().with_config(INTERNAL_PREVIEW_SUMMARY_CONFIG)
    response = model.invoke(
        [
            SystemMessage(
                content=_build_summary_system_prompt(
                    depth=depth,
                    file_name=file_name,
                    file_type=file_type,
                    material_kind="text",
                )
            ),
            HumanMessage(
                content=(
                    "以下是文件预览原始材料，请生成适合文件预览场景的中文摘要：\n\n"
                    f"{preview_text}"
                )
            ),
        ]
    )
    text = _coerce_response_text(response.content)
    if not text:
        raise ValueError("文本预览摘要模型返回空结果。")
    return text


def _summarize_visual_source(*, file_name: str, file_type: str, depth: str, file_path: Path) -> str:
    assets = _render_visual_assets(file_path=file_path, file_type=file_type, depth=depth)
    if not assets:
        raise ValueError("没有可用于视觉摘要的图片资产。")

    model = create_summary_model().with_config(INTERNAL_PREVIEW_SUMMARY_CONFIG)
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": "以下是当前文件的视觉预览材料，请直接输出摘要正文。",
        }
    ]
    for asset in assets:
        encoded = base64.b64encode(asset.image_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{asset.mime_type};base64,{encoded}"},
            }
        )

    response = model.invoke(
        [
            SystemMessage(
                content=_build_summary_system_prompt(
                    depth=depth,
                    file_name=file_name,
                    file_type=file_type,
                    material_kind="visual",
                )
            ),
            HumanMessage(content=content),
        ]
    )
    text = _coerce_response_text(response.content)
    if not text:
        raise ValueError("视觉预览摘要模型返回空结果。")
    return text


def _load_summary_prompt() -> str:
    return SUMMARY_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _build_summary_system_prompt(
    *,
    depth: str,
    file_name: str,
    file_type: str,
    material_kind: str,
) -> str:
    summary_prompt = _load_summary_prompt()
    return (
        f"{summary_prompt}\n\n"
        f"当前文件名：{file_name}\n"
        f"当前文件类型：{file_type}\n"
        f"当前预览深度：{depth}\n"
        f"当前输入材料类型：{material_kind}"
    )


def _render_visual_assets(
    *,
    file_path: Path,
    file_type: str,
    depth: str,
) -> tuple[SummaryVisualAsset, ...]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf" or file_type.lower() == "pdf":
        return _render_pdf_assets(file_path, depth)
    if suffix in IMAGE_FILE_SUFFIXES:
        return (_render_image_asset(file_path),)
    raise ValueError(f"当前文件类型不支持视觉总结：{file_type}")


def _render_image_asset(file_path: Path) -> SummaryVisualAsset:
    with Image.open(file_path) as image:
        image.load()
        prepared = _prepare_image_for_summary(image)

    buffer = BytesIO()
    prepared.save(buffer, format="PNG")
    return SummaryVisualAsset(
        mime_type=MIME_TYPE_BY_SUFFIX.get(file_path.suffix.lower(), "image/png"),
        image_bytes=buffer.getvalue(),
    )


def _render_pdf_assets(file_path: Path, depth: str) -> tuple[SummaryVisualAsset, ...]:
    document = fitz.open(file_path)
    try:
        page_limit = min(document.page_count, _pdf_page_limit(depth))
        render_scale = get_settings().preview.pdf_render_scale
        assets: list[SummaryVisualAsset] = []

        for page_index in range(page_limit):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), alpha=False)
            image_bytes = pixmap.tobytes("png")
            if not image_bytes:
                raise ValueError(f"PDF 第 {page_index + 1} 页渲染为空。")
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                prepared = _prepare_image_for_summary(image)
            buffer = BytesIO()
            prepared.save(buffer, format="PNG")
            assets.append(SummaryVisualAsset(mime_type="image/png", image_bytes=buffer.getvalue()))

        if not assets:
            raise ValueError("PDF 没有可用于视觉总结的页面。")
        return tuple(assets)
    finally:
        document.close()


def _prepare_image_for_summary(image: Image.Image) -> Image.Image:
    prepared = image.copy()
    if prepared.mode not in {"RGB", "RGBA"}:
        prepared = prepared.convert("RGB")
    max_edge = max(get_settings().preview.image_max_edge, 1)
    prepared.thumbnail((max_edge, max_edge))
    return prepared


def _pdf_page_limit(depth: str) -> int:
    return get_settings().preview.pdf_depth_pages[depth]


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
