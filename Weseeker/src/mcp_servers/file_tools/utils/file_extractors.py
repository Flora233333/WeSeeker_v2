from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import fitz
from docx import Document
from openpyxl import load_workbook
from PIL import Image
from pptx import Presentation

from config.settings import get_settings


TEXT_FILE_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
IMAGE_FILE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
SUPPORTED_DEPTHS = {"L1", "L2", "L3"}


class UnsupportedFileTypeError(ValueError):
    pass


@dataclass(frozen=True)
class FilePreview:
    file_type: str
    preview_text: str
    metadata: dict[str, object]


def normalize_depth(depth: str) -> str:
    normalized = depth.strip().upper()
    if normalized not in SUPPORTED_DEPTHS:
        raise ValueError("depth 仅支持 L1、L2、L3。")
    return normalized


def _format_decimal_size(size: int, unit_base: int) -> str:
    value = Decimal(size) / Decimal(unit_base)
    rounded = value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{rounded}"


def format_size_display(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 ** 2:
        return f"{_format_decimal_size(size, 1024)} KB"
    if size < 1024 ** 3:
        return f"{_format_decimal_size(size, 1024 ** 2)} MB"
    return f"{_format_decimal_size(size, 1024 ** 3)} GB"


def extract_preview(file_path: Path, depth: str) -> FilePreview:
    normalized_depth = normalize_depth(depth)
    suffix = file_path.suffix.lower()

    if suffix in TEXT_FILE_SUFFIXES:
        return _extract_text_preview(file_path, normalized_depth, file_type=suffix.lstrip("."))
    if suffix == ".docx":
        return _extract_docx_preview(file_path, normalized_depth)
    if suffix == ".xlsx":
        return _extract_xlsx_preview(file_path, normalized_depth)
    if suffix == ".pptx":
        return _extract_pptx_preview(file_path, normalized_depth)
    if suffix == ".pdf":
        return _extract_pdf_preview(file_path, normalized_depth)
    if suffix in IMAGE_FILE_SUFFIXES:
        return _extract_image_preview(file_path, normalized_depth)

    raise UnsupportedFileTypeError(f"暂不支持读取 {suffix or '无扩展名'} 文件。")


def build_common_metadata(file_path: Path) -> dict[str, object]:
    stat_result = file_path.stat()
    return {
        "size": stat_result.st_size,
        "size_display": format_size_display(stat_result.st_size),
        "modified": datetime.fromtimestamp(stat_result.st_mtime).isoformat(timespec="seconds"),
    }


def _text_char_limit(depth: str) -> int:
    return get_settings().preview.text_depth_chars[depth]


def _excel_row_limit(depth: str) -> int:
    return get_settings().preview.excel_depth_rows[depth]


def _pdf_page_limit(depth: str) -> int:
    return get_settings().preview.pdf_depth_pages[depth]


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(normalized_text) <= limit:
        return normalized_text, False
    return normalized_text[:limit].rstrip() + "...", True


def _merge_metadata(file_path: Path, extra_metadata: dict[str, object]) -> dict[str, object]:
    metadata = build_common_metadata(file_path)
    metadata.update(extra_metadata)
    return metadata


def _read_text_content(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="utf-8", errors="replace")


def _extract_text_preview(file_path: Path, depth: str, *, file_type: str) -> FilePreview:
    content = _read_text_content(file_path)
    preview_text, truncated = _truncate_text(content, _text_char_limit(depth))
    metadata = _merge_metadata(
        file_path,
        {
            "line_count": len(content.splitlines()),
            "char_count": len(content),
            "truncated": truncated,
        },
    )
    return FilePreview(file_type=file_type, preview_text=preview_text, metadata=metadata)


def _extract_docx_preview(file_path: Path, depth: str) -> FilePreview:
    document = Document(file_path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    content = "\n".join(paragraphs)
    preview_text, truncated = _truncate_text(content, _text_char_limit(depth))
    metadata = _merge_metadata(
        file_path,
        {
            "paragraph_count": len(paragraphs),
            "char_count": len(content),
            "truncated": truncated,
        },
    )
    return FilePreview(file_type="docx", preview_text=preview_text, metadata=metadata)


def _extract_xlsx_preview(file_path: Path, depth: str) -> FilePreview:
    workbook = load_workbook(file_path, read_only=True, data_only=True)
    row_limit = _excel_row_limit(depth)
    sheet_names: list[str] = []
    preview_sections: list[str] = []

    for worksheet in workbook.worksheets:
        sheet_names.append(worksheet.title)
        preview_sections.append(f"[Sheet] {worksheet.title}")
        for row_index, row in enumerate(worksheet.iter_rows(values_only=True), start=1):
            if row_index > row_limit:
                break
            values = [str(cell) if cell is not None else "" for cell in row]
            preview_sections.append(" | ".join(values).rstrip())

    content = "\n".join(section for section in preview_sections if section)
    preview_text, truncated = _truncate_text(content, _text_char_limit(depth))
    metadata = _merge_metadata(
        file_path,
        {
            "sheet_count": len(sheet_names),
            "sheet_names": sheet_names,
            "preview_rows_per_sheet": row_limit,
            "truncated": truncated,
        },
    )
    workbook.close()
    return FilePreview(file_type="xlsx", preview_text=preview_text, metadata=metadata)


def _extract_pptx_preview(file_path: Path, depth: str) -> FilePreview:
    presentation = Presentation(file_path)
    slide_limit = _pdf_page_limit(depth)
    preview_sections: list[str] = []

    for slide_index, slide in enumerate(presentation.slides, start=1):
        if slide_index > slide_limit:
            break
        shape_texts: list[str] = []
        for shape in slide.shapes:
            text = getattr(shape, "text", "")
            if text and text.strip():
                shape_texts.append(text.strip())
        preview_sections.append(f"[Slide {slide_index}]")
        preview_sections.append("\n".join(shape_texts) if shape_texts else "<无可提取文本>")

    content = "\n".join(preview_sections)
    preview_text, truncated = _truncate_text(content, _text_char_limit(depth))
    metadata = _merge_metadata(
        file_path,
        {
            "slide_count": len(presentation.slides),
            "preview_slides": min(len(presentation.slides), slide_limit),
            "truncated": truncated,
        },
    )
    return FilePreview(file_type="pptx", preview_text=preview_text, metadata=metadata)


def _extract_pdf_preview(file_path: Path, depth: str) -> FilePreview:
    document = fitz.open(file_path)
    page_limit = _pdf_page_limit(depth)
    preview_sections: list[str] = []

    for page_index in range(min(document.page_count, page_limit)):
        page = document.load_page(page_index)
        preview_sections.append(f"[Page {page_index + 1}]")
        preview_sections.append(page.get_text("text").strip() or "<当前页可提取文本较少>")

    content = "\n".join(preview_sections)
    preview_text, truncated = _truncate_text(content, _text_char_limit(depth))
    metadata = _merge_metadata(
        file_path,
        {
            "page_count": document.page_count,
            "preview_pages": min(document.page_count, page_limit),
            "truncated": truncated,
        },
    )
    document.close()
    return FilePreview(file_type="pdf", preview_text=preview_text, metadata=metadata)


def _extract_image_preview(file_path: Path, depth: str) -> FilePreview:
    del depth

    with Image.open(file_path) as image:
        metadata = _merge_metadata(
            file_path,
            {
                "format": image.format or "unknown",
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            },
        )

    return FilePreview(
        file_type=file_path.suffix.lower().lstrip("."),
        preview_text="当前仅提供图片基础信息预览，暂未启用 OCR 或视觉摘要。",
        metadata=metadata,
    )
