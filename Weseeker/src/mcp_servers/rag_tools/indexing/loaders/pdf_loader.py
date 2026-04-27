from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from langchain_core.documents import Document

_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百\d]+[章节部分篇]"
    r"|附录(?:\s*[A-Za-z0-9一二三四五六七八九十]+)?(?:\s+|$)"
    r"|[1-9]\d*(?:\.\d+)+\s+"
    r"|[1-9]\d*[.、]\s+"
    r"|[一二三四五六七八九十]+[、.．]"
    r"|(?:chapter|section|part)\s+\d+"
    r")",
    flags=re.IGNORECASE,
)
_CHINESE_TOP_HEADING_PATTERN = re.compile(r"^[一二三四五六七八九十]+[、.．]\S+")
_NUMERIC_DOTTED_HEADING_PATTERN = re.compile(r"^([1-9]\d*(?:\.\d+)+)\s+\S+")
_APPENDIX_HEADING_PATTERN = re.compile(
    r"^附录(?:\s*[A-Za-z0-9一二三四五六七八九十]+)?(?:\s+|$)"
)
_BARE_APPENDIX_TITLE_PATTERN = re.compile(r"^附录$")
_PAGE_FOOTER_PATTERN = re.compile(r"^第\s*\d+\s*页$")
_TOC_DOTS_PATTERN = re.compile(r"(?:\.{3,}|…{2,}|·{3,})")
_TOC_TRAILING_PAGE_PATTERN = re.compile(r"(?:\.{3,}|…{2,}|·{3,})\s*\d+\s*$")
_COVER_DECORATION_PATTERN = re.compile(r"^[\s\-—–_·]+$")
_PDF_MICRO_PARENT_LIMIT = 80
_PDF_PROMOTED_ROOT_MIN = 120
_PDF_SUBSECTION_PROMOTE_CHARS = 300
_PDF_SUBSECTION_PROMOTE_PAGE_SPAN = 200
_PDF_SUBSECTION_PROMOTE_BLOCKS = 3
_PDF_SUBSECTION_PROMOTE_BLOCK_CHARS = 220
_REPEATED_HEADER_MIN_PAGES = 3
_REPEATED_HEADER_MAX_CHARS = 80
_TOC_PAGE_MIN_LINES_WITH_TITLE = 3
_TOC_PAGE_MIN_LINES_WITHOUT_TITLE = 6


@dataclass(frozen=True)
class PdfTextBlock:
    text: str
    page_number: int
    top: float
    max_font_size: float
    line_count: int


@dataclass(frozen=True)
class PdfSectionCandidate:
    level: int
    title: str
    body_texts: tuple[str, ...]
    page_start: int
    page_end: int

    @property
    def body_char_count(self) -> int:
        return len(_normalize_block_text("\n\n".join(self.body_texts)))

    def to_text(self, *, include_title: bool = True) -> str:
        parts: list[str] = []
        if include_title and self.title:
            parts.append(self.title)
        parts.extend(text for text in self.body_texts if text.strip())
        return _normalize_block_text("\n\n".join(parts))


@dataclass
class PdfParentDraft:
    title_hierarchy: list[str] = field(default_factory=list)
    section_title: str = ""
    level: int = 1
    texts: list[str] = field(default_factory=list)
    page_start: int = 1
    page_end: int = 1

    @property
    def char_count(self) -> int:
        return len(_normalize_block_text("\n\n".join(self.texts)))


def load_pdf(path: Path) -> list[Document]:
    pdf = fitz.open(path)
    try:
        # 先尝试恢复“标题 + 正文块”结构，恢复失败时再回退到逐页文本。
        text_blocks = _extract_text_blocks(pdf)
        structured_documents = _build_structured_documents(path, text_blocks)
        if structured_documents:
            return structured_documents
        return _build_page_fallback_documents(path, pdf)
    finally:
        pdf.close()


def _extract_text_blocks(pdf) -> list[PdfTextBlock]:
    blocks: list[PdfTextBlock] = []
    for page_index in range(pdf.page_count):
        page = pdf.load_page(page_index)
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            lines: list[str] = []
            max_font_size = 0.0
            for line in block.get("lines", []):
                # PDF 一行可能被切成多个 span，这里先把同一行重新拼回去。
                span_text = "".join(
                    span.get("text", "")
                    for span in line.get("spans", [])
                    if span.get("text", "").strip()
                ).strip()
                if not span_text:
                    continue
                lines.append(span_text)
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        max_font_size = max(max_font_size, float(span.get("size", 0.0)))

            text = _normalize_block_text("\n".join(lines))
            if not text:
                continue

            bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            blocks.append(
                PdfTextBlock(
                    text=text,
                    page_number=page_index + 1,
                    top=float(bbox[1]),
                    max_font_size=max_font_size,
                    line_count=max(1, len(lines)),
                )
            )

    blocks.sort(key=lambda block: (block.page_number, block.top))
    return blocks


def _build_structured_documents(path: Path, blocks: list[PdfTextBlock]) -> list[Document]:
    if not blocks:
        return []

    blocks = _prepare_structure_blocks(blocks)
    if not blocks:
        return []

    body_font_size = _infer_body_font_size(blocks)
    document_title = _detect_document_title(blocks, body_font_size)
    candidates = _build_section_candidates(blocks, body_font_size, document_title)
    if not candidates:
        return []

    drafts = _build_dynamic_parent_drafts(candidates)
    drafts = _merge_micro_parent_drafts(drafts)
    drafts = _remove_cover_metadata_drafts(drafts)
    return [_draft_to_document(path, draft, document_title) for draft in drafts]


def _prepare_structure_blocks(blocks: list[PdfTextBlock]) -> list[PdfTextBlock]:
    blocks = _remove_repeated_headers_and_footers(blocks)
    blocks = _remove_table_of_contents_blocks(blocks)
    return [
        block
        for block in blocks
        if not _is_cover_decoration(block.text) and not _is_bare_appendix_title(block.text)
    ]


def _remove_repeated_headers_and_footers(blocks: list[PdfTextBlock]) -> list[PdfTextBlock]:
    repeated_headers = _detect_repeated_headers(blocks)
    return [
        block
        for block in blocks
        if _normalize_repeated_text(block.text) not in repeated_headers
        and not _is_page_footer(block.text)
    ]


def _detect_repeated_headers(blocks: list[PdfTextBlock]) -> set[str]:
    # PDF 页眉通常是短文本并跨多页重复；用页数门槛避免误删正文标题。
    page_numbers_by_text: dict[str, set[int]] = {}
    for block in blocks:
        normalized = _normalize_repeated_text(block.text)
        if not normalized or len(normalized) > _REPEATED_HEADER_MAX_CHARS:
            continue
        if _is_page_footer(normalized):
            continue
        page_numbers_by_text.setdefault(normalized, set()).add(block.page_number)

    return {
        text
        for text, page_numbers in page_numbers_by_text.items()
        if len(page_numbers) >= _REPEATED_HEADER_MIN_PAGES
    }


def _remove_table_of_contents_blocks(blocks: list[PdfTextBlock]) -> list[PdfTextBlock]:
    toc_pages = _detect_toc_pages(blocks)
    if not toc_pages:
        return blocks
    return [block for block in blocks if block.page_number not in toc_pages]


def _detect_toc_pages(blocks: list[PdfTextBlock]) -> set[int]:
    # 目录页的强信号是“目录标题 + 多行点线页码”，不做宽泛文本猜测。
    blocks_by_page: dict[int, list[PdfTextBlock]] = {}
    for block in blocks:
        blocks_by_page.setdefault(block.page_number, []).append(block)

    toc_pages: set[int] = set()
    for page_number, page_blocks in blocks_by_page.items():
        toc_line_count = sum(1 for block in page_blocks if _looks_like_toc_line(block.text))
        has_toc_title = any(_is_toc_title(block.text) for block in page_blocks)
        if has_toc_title and toc_line_count >= _TOC_PAGE_MIN_LINES_WITH_TITLE:
            toc_pages.add(page_number)
            continue
        if toc_line_count >= _TOC_PAGE_MIN_LINES_WITHOUT_TITLE:
            toc_pages.add(page_number)
    return toc_pages


def _build_section_candidates(
    blocks: list[PdfTextBlock],
    body_font_size: float,
    document_title: str,
) -> list[PdfSectionCandidate]:
    candidates: list[PdfSectionCandidate] = []
    pending_preface_texts: list[str] = []
    pending_preface_start: int | None = None
    pending_preface_end: int | None = None
    current_level: int | None = None
    current_title = ""
    current_body: list[str] = []
    current_page_start = 1
    current_page_end = 1
    saw_heading = False

    for block in blocks:
        if document_title and block.page_number == 1 and block.text == document_title:
            continue

        heading_level = _infer_heading_level(block, body_font_size, document_title)
        if heading_level is not None:
            saw_heading = True
            if current_level is not None:
                candidates.append(
                    PdfSectionCandidate(
                        level=current_level,
                        title=current_title,
                        body_texts=tuple(current_body),
                        page_start=current_page_start,
                        page_end=current_page_end,
                    )
                )
            elif pending_preface_texts and document_title:
                # 标题前的说明性文字并到文档标题下，避免丢失首页前言。
                candidates.append(
                    PdfSectionCandidate(
                        level=1,
                        title=document_title,
                        body_texts=tuple(pending_preface_texts),
                        page_start=pending_preface_start or block.page_number,
                        page_end=pending_preface_end or block.page_number,
                    )
                )
                pending_preface_texts = []
                pending_preface_start = None
                pending_preface_end = None

            current_level = heading_level
            current_title = block.text
            current_body = []
            current_page_start = block.page_number
            current_page_end = block.page_number
            continue

        if current_level is None:
            if not document_title:
                continue
            if pending_preface_start is None:
                pending_preface_start = block.page_number
            pending_preface_texts.append(block.text)
            pending_preface_end = block.page_number
            continue

        current_body.append(block.text)
        current_page_end = block.page_number

    if current_level is not None:
        candidates.append(
            PdfSectionCandidate(
                level=current_level,
                title=current_title,
                body_texts=tuple(current_body),
                page_start=current_page_start,
                page_end=current_page_end,
            )
        )

    return candidates if saw_heading else []


def _build_dynamic_parent_drafts(candidates: list[PdfSectionCandidate]) -> list[PdfParentDraft]:
    drafts: list[PdfParentDraft] = []
    current_root: PdfParentDraft | None = None
    current_promoted: PdfParentDraft | None = None

    for candidate in candidates:
        if candidate.level <= 1 or current_root is None:
            current_promoted = None
            current_root = _create_draft([candidate.title], candidate)
            drafts.append(current_root)
            continue

        if candidate.level == 2:
            current_promoted = None

        parent_for_nesting = current_root
        if current_promoted and current_promoted.level < candidate.level:
            parent_for_nesting = current_promoted

        if (
            _should_promote_subsection(candidate)
            and current_root.char_count >= _PDF_PROMOTED_ROOT_MIN
        ):
            current_promoted = _create_draft(
                [*parent_for_nesting.title_hierarchy, candidate.title],
                candidate,
            )
            drafts.append(current_promoted)
            continue

        _append_candidate_to_draft(parent_for_nesting, candidate)

    return drafts


def _merge_micro_parent_drafts(drafts: list[PdfParentDraft]) -> list[PdfParentDraft]:
    if not drafts:
        return []

    merged: list[PdfParentDraft] = [drafts[0]]
    for draft in drafts[1:]:
        previous = merged[-1]
        # 当上一个块短到几乎只剩标题，且下一个块是它的子标题时，直接合并。
        if _should_merge_micro_parent(previous, draft):
            previous.texts.extend(draft.texts)
            previous.page_end = max(previous.page_end, draft.page_end)
            continue
        merged.append(draft)
    return merged


def _remove_cover_metadata_drafts(drafts: list[PdfParentDraft]) -> list[PdfParentDraft]:
    return [draft for draft in drafts if not _is_cover_metadata_draft(draft)]


def _is_cover_metadata_draft(draft: PdfParentDraft) -> bool:
    if draft.page_start != 1 or draft.page_end != 1:
        return False
    if draft.char_count > 180:
        return False
    text = _normalize_block_text("\n".join(draft.texts))
    # 封面编制信息很容易和标题 query 高度重合，但对问答上下文几乎没有价值。
    return len(draft.title_hierarchy) == 1 or any(
        marker in text for marker in ("编制单位", "编制日期", "版本号")
    )


def _should_merge_micro_parent(previous: PdfParentDraft, current: PdfParentDraft) -> bool:
    if previous.char_count >= _PDF_MICRO_PARENT_LIMIT:
        return False
    if len(current.title_hierarchy) <= len(previous.title_hierarchy):
        return False
    return current.title_hierarchy[: len(previous.title_hierarchy)] == previous.title_hierarchy


def _should_promote_subsection(candidate: PdfSectionCandidate) -> bool:
    if candidate.level <= 1:
        return False
    if candidate.body_char_count >= _PDF_SUBSECTION_PROMOTE_CHARS:
        return True
    if (
        candidate.page_end > candidate.page_start
        and candidate.body_char_count >= _PDF_SUBSECTION_PROMOTE_PAGE_SPAN
    ):
        return True
    if (
        len(candidate.body_texts) >= _PDF_SUBSECTION_PROMOTE_BLOCKS
        and candidate.body_char_count >= _PDF_SUBSECTION_PROMOTE_BLOCK_CHARS
    ):
        return True
    return False


def _create_draft(title_hierarchy: list[str], candidate: PdfSectionCandidate) -> PdfParentDraft:
    return PdfParentDraft(
        title_hierarchy=title_hierarchy,
        section_title=title_hierarchy[-1],
        level=candidate.level,
        texts=[candidate.to_text()],
        page_start=candidate.page_start,
        page_end=candidate.page_end,
    )


def _append_candidate_to_draft(draft: PdfParentDraft, candidate: PdfSectionCandidate) -> None:
    draft.texts.append(candidate.to_text())
    draft.page_end = max(draft.page_end, candidate.page_end)


def _draft_to_document(path: Path, draft: PdfParentDraft, document_title: str) -> Document:
    heading_path = _compose_heading_path(document_title, draft.title_hierarchy)
    return Document(
        page_content=_normalize_block_text("\n\n".join(draft.texts)),
        metadata={
            "doc_type": "pdf",
            "file_path": path.as_posix(),
            "file_name": path.name,
            "file_ext": path.suffix.lower(),
            "page_number": draft.page_start,
            "page_end": draft.page_end,
            "section_title": draft.section_title or heading_path,
            "heading_path": heading_path,
            "pdf_structure_source": "font_blocks_dynamic",
        },
    )


def _build_page_fallback_documents(path: Path, pdf) -> list[Document]:
    documents: list[Document] = []
    for page_index in range(pdf.page_count):
        page = pdf.load_page(page_index)
        documents.append(
            Document(
                page_content=page.get_text("text"),
                metadata={
                    "doc_type": "pdf",
                    "file_path": path.as_posix(),
                    "file_name": path.name,
                    "file_ext": path.suffix.lower(),
                    "page_number": page_index + 1,
                    # 标记这是“没恢复出结构，只能按页兜底”的结果。
                    "pdf_structure_source": "page_fallback",
                },
            )
        )
    return documents


def _infer_body_font_size(blocks: list[PdfTextBlock]) -> float:
    # 正文字号通常是出现次数最多的字号，比取平均值更能抗标题/图注干扰。
    sizes = [round(block.max_font_size, 1) for block in blocks if len(block.text) >= 20]
    if not sizes:
        sizes = [round(block.max_font_size, 1) for block in blocks]
    if not sizes:
        return 0.0
    return float(Counter(sizes).most_common(1)[0][0])


def _detect_document_title(blocks: list[PdfTextBlock], body_font_size: float) -> str:
    first_page_blocks = [block for block in blocks if block.page_number == 1]
    if not first_page_blocks:
        return ""

    title_candidates = [
        block
        for block in first_page_blocks
        if len(block.text) <= 80 and block.max_font_size >= body_font_size + 4
    ]
    if not title_candidates:
        return ""

    return max(title_candidates, key=lambda block: (block.max_font_size, -block.top)).text


def _infer_heading_level(
    block: PdfTextBlock,
    body_font_size: float,
    document_title: str,
) -> int | None:
    if document_title and block.page_number == 1 and block.text == document_title:
        return None
    if _is_toc_title(block.text) or _looks_like_toc_line(block.text):
        return None
    if _is_cover_decoration(block.text):
        return None
    if len(block.text) > 120:
        return None
    if block.line_count > 3:
        return None
    numbered_level = _infer_numbered_heading_level(block.text)
    if numbered_level is not None and not _looks_like_sentence(block.text):
        return numbered_level

    looks_like_heading = _looks_like_heading(block.text)
    is_large_font = block.max_font_size >= body_font_size + 1.5
    if not looks_like_heading and not is_large_font:
        return None
    if _looks_like_sentence(block.text):
        return None

    # 当前只需要 3 级近似层级，后续再根据真实 PDF 调细。
    if block.max_font_size >= body_font_size + 5:
        return 1
    if block.max_font_size >= body_font_size + 2.5:
        return 2
    return 3


def _looks_like_heading(text: str) -> bool:
    compact = text.replace("\n", " ").strip()
    if not compact:
        return False
    return bool(_HEADING_PATTERN.match(compact))


def _infer_numbered_heading_level(text: str) -> int | None:
    compact = _normalize_repeated_text(text)
    if not compact:
        return None
    if _APPENDIX_HEADING_PATTERN.match(compact):
        return 1
    if _CHINESE_TOP_HEADING_PATTERN.match(compact):
        return 1
    numeric_match = _NUMERIC_DOTTED_HEADING_PATTERN.match(compact)
    if not numeric_match:
        return None
    depth = numeric_match.group(1).count(".") + 1
    if depth <= 2:
        return 2
    return 3


def _is_toc_title(text: str) -> bool:
    compact = _normalize_repeated_text(text)
    return compact in {"目录", "目 录"}


def _looks_like_toc_line(text: str) -> bool:
    compact = _normalize_repeated_text(text)
    if not compact or len(compact) > 220:
        return False
    if _is_toc_title(compact):
        return False
    return bool(_TOC_DOTS_PATTERN.search(compact) and _TOC_TRAILING_PAGE_PATTERN.search(compact))


def _is_page_footer(text: str) -> bool:
    return bool(_PAGE_FOOTER_PATTERN.match(_normalize_repeated_text(text)))


def _is_cover_decoration(text: str) -> bool:
    compact = _normalize_repeated_text(text)
    return len(compact) <= 10 and bool(_COVER_DECORATION_PATTERN.match(compact))


def _is_bare_appendix_title(text: str) -> bool:
    return bool(_BARE_APPENDIX_TITLE_PATTERN.match(_normalize_repeated_text(text)))


def _looks_like_sentence(text: str) -> bool:
    compact = text.replace("\n", " ").strip()
    if len(compact) > 160:
        return True
    return compact.endswith(("。", "！", "？", "；", ".", "!", "?", ";"))


def _compose_heading_path(document_title: str, title_hierarchy: list[str]) -> str:
    parts = []
    if document_title:
        parts.append(document_title)
    parts.extend(part for part in title_hierarchy if part and part != document_title)
    return " / ".join(parts)


def _normalize_block_text(text: str) -> str:
    normalized = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _normalize_repeated_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split()).strip()
