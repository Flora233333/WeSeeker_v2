from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

from langchain_core.documents import Document

from mcp_servers.rag_tools.indexing.scanner import FileRecord

ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\ufeff\u2060")
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
IMAGE_TAG_PATTERN = re.compile(r"<img\b[^>]*>", flags=re.IGNORECASE)
TOC_PATTERN = re.compile(r"^\s*\[TOC\]\s*$", flags=re.IGNORECASE)
DIVIDER_PATTERN = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
EMPTY_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[*+-]|\d+\.)\s*(?:[:：])?\s*$")


def enhance_documents(
    documents: list[Document],
    *,
    kb_name: str,
    file_record: FileRecord,
) -> list[Document]:
    doc_type = documents[0].metadata.get("doc_type") if documents else ""
    if doc_type == "pdf":
        repeated_headers, repeated_footers = _find_repeated_pdf_edges(documents)
    else:
        repeated_headers, repeated_footers = set(), set()
    modified = datetime.fromtimestamp(file_record.mtime).strftime("%Y-%m-%d %H:%M:%S")

    enhanced: list[Document] = []
    for document in documents:
        text = _normalize_text(document.page_content)
        if document.metadata.get("doc_type") == "md":
            text = _strip_markdown_noise(text)
        if document.metadata.get("doc_type") == "pdf":
            text = _strip_pdf_noise(text, repeated_headers, repeated_footers)
        if not text.strip():
            continue

        metadata = dict(document.metadata)
        metadata["kb_name"] = kb_name
        metadata["mtime"] = modified
        metadata.setdefault("section_title", _extract_section_title(text, metadata))
        enhanced.append(Document(page_content=text, metadata=metadata))

    return enhanced


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _strip_markdown_noise(text: str) -> str:
    """Remove markdown-only indexing noise without touching technical tokens like <mask>."""

    cleaned_lines: list[str] = []
    in_fence = False

    for raw_line in text.translate(ZERO_WIDTH_TRANSLATION).splitlines():
        line = raw_line.rstrip()
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            cleaned_lines.append(line)
            continue

        if in_fence:
            cleaned_lines.append(line)
            continue

        line = _remove_markdown_images(line)
        line = IMAGE_TAG_PATTERN.sub("", line)

        if TOC_PATTERN.match(line):
            continue
        if DIVIDER_PATTERN.match(line):
            continue
        if EMPTY_LIST_ITEM_PATTERN.match(line):
            continue

        cleaned_lines.append(line)

    return _normalize_text("\n".join(cleaned_lines))


def _remove_markdown_images(text: str) -> str:
    """Strip ![alt](path) spans while preserving surrounding prose on the same line."""

    parts: list[str] = []
    index = 0
    text_length = len(text)

    while index < text_length:
        if not text.startswith("![", index):
            parts.append(text[index])
            index += 1
            continue

        close_bracket = text.find("](", index + 2)
        if close_bracket == -1:
            parts.append(text[index])
            index += 1
            continue

        cursor = close_bracket + 2
        depth = 1
        while cursor < text_length and depth > 0:
            char = text[cursor]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            cursor += 1

        if depth != 0:
            parts.append(text[index])
            index += 1
            continue

        index = cursor

    return "".join(parts)


def _find_repeated_pdf_edges(documents: list[Document]) -> tuple[set[str], set[str]]:
    first_lines: Counter[str] = Counter()
    last_lines: Counter[str] = Counter()
    for document in documents:
        lines = [line.strip() for line in document.page_content.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines[0]) <= 80:
            first_lines[lines[0]] += 1
        if len(lines[-1]) <= 80:
            last_lines[lines[-1]] += 1

    repeated_headers = {line for line, count in first_lines.items() if count >= 2}
    repeated_footers = {line for line, count in last_lines.items() if count >= 2}
    return repeated_headers, repeated_footers


def _strip_pdf_noise(text: str, repeated_headers: set[str], repeated_footers: set[str]) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned = [
        line
        for line in lines
        if line
        and line not in repeated_headers
        and line not in repeated_footers
        and not re.fullmatch(r"(?:page\s+)?\d+", line, flags=re.IGNORECASE)
    ]
    return _normalize_text("\n".join(cleaned))


def _extract_section_title(text: str, metadata: dict[str, object]) -> str:
    if metadata.get("sheet_name"):
        return str(metadata["sheet_name"])
    if metadata.get("heading_path"):
        return str(metadata["heading_path"])
    if metadata.get("page_number"):
        return f"Page {metadata['page_number']}"

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip()
    return ""
