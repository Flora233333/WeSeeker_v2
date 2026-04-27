from __future__ import annotations

import re
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config.settings import RAGSettings

_PARENT_SEPARATORS = ["\n\n", "\n", "。", ". ", " ", ""]
_CHILD_FALLBACK_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ". ",
    "! ",
    "? ",
    "; ",
    "：",
    ":",
    "，",
    "、",
    " ",
    "",
]
_CHINESE_SENTENCE_SPLIT_REGEX = (
    r"(?<=[。！？；!?])\s*"
    r"|(?<=[.?!;])\s+"
    r"|\n{2,}"
    r"|\n(?=\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|[（(]?\d+[）)]\s+|"
    r"[一二三四五六七八九十]+[、.．]\s+))"
)
_CHILD_NOISE_MAX_CHARS = 80
_SEMANTIC_CHAR_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_STRUCTURAL_ONLY_PATTERN = re.compile(
    r"^(?:`{3,}|~{3,}|\${1,2}|-{3,}|\*{3,}|_{3,}|#{1,6})$"
)
_SHORT_HEADING_PATTERN = re.compile(
    r"^(?:#{1,6}\s+|第[一二三四五六七八九十百\d]+[章节部分篇]|"
    r"[一二三四五六七八九十]+[、.．]|[1-9]\d*(?:\.\d+)*[.、]?\s+)"
)


class ChineseAwareSemanticChunker(SemanticChunker):
    """Keep SemanticChunker logic, but feed it cleaner Chinese-aware units first."""

    def _get_single_sentences_list(self, text: str) -> list[str]:
        return [
            unit.strip()
            for unit in re.split(_CHINESE_SENTENCE_SPLIT_REGEX, text)
            if unit and unit.strip()
        ]


def split_documents(
    documents: list[Document],
    *,
    embeddings: Embeddings,
    settings: RAGSettings,
) -> tuple[list[Document], list[Document]]:
    parent_docs: list[Document] = []
    child_docs: list[Document] = []

    semantic_splitter = ChineseAwareSemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=settings.semantic_breakpoint_percentile,
    )
    fallback_child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        separators=_CHILD_FALLBACK_SEPARATORS,
    )

    for document in documents:
        parents = _build_parent_chunks(document, settings)
        for parent in parents:
            parent_id = uuid4().hex
            parent_metadata = dict(parent.metadata)
            parent_metadata["parent_id"] = parent_id
            parent_metadata["chunk_level"] = "parent"
            parent_metadata["heading_path"] = _resolve_heading_path(parent_metadata)
            parent_metadata["char_count"] = len(parent.page_content)
            parent_doc = Document(
                page_content=parent.page_content.strip(),
                metadata=parent_metadata,
            )
            parent_docs.append(parent_doc)

            child_texts = _build_child_texts(
                parent_doc.page_content,
                semantic_splitter=semantic_splitter,
                fallback_splitter=fallback_child_splitter,
                settings=settings,
                metadata=parent_doc.metadata,
            )
            for child_text in child_texts:
                child_metadata = dict(parent_metadata)
                child_metadata["child_id"] = uuid4().hex
                child_metadata["chunk_level"] = "child"
                child_metadata["char_count"] = len(child_text)
                child_docs.append(Document(page_content=child_text, metadata=child_metadata))

    return parent_docs, child_docs


def _build_parent_chunks(document: Document, settings: RAGSettings) -> list[Document]:
    doc_type = str(document.metadata.get("doc_type") or "")
    if doc_type in {"md", "docx"}:
        return _split_markdown_like(document, settings)
    if doc_type == "xlsx":
        metadata = dict(document.metadata)
        metadata["heading_path"] = str(document.metadata.get("sheet_name") or "")
        return _split_recursively(document.page_content, metadata, settings)
    if doc_type == "pdf":
        metadata = dict(document.metadata)
        metadata["heading_path"] = str(
            document.metadata.get("heading_path") or f"Page {document.metadata.get('page_number')}"
        )
        return _split_recursively(document.page_content, metadata, settings)
    return _split_recursively(document.page_content, dict(document.metadata), settings)


def _split_markdown_like(document: Document, settings: RAGSettings) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(document.page_content)
    if not sections:
        sections = [Document(page_content=document.page_content, metadata={})]

    parent_docs: list[Document] = []
    for section in sections:
        metadata = dict(document.metadata)
        metadata.update(section.metadata)
        metadata["heading_path"] = _resolve_heading_path(metadata)
        parent_docs.extend(_split_recursively(section.page_content, metadata, settings))
    return parent_docs


def _split_recursively(
    text: str,
    metadata: dict[str, object],
    settings: RAGSettings,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
        separators=_PARENT_SEPARATORS,
    )
    return splitter.create_documents([text], metadatas=[metadata])


def _build_child_texts(
    text: str,
    *,
    semantic_splitter: SemanticChunker,
    fallback_splitter: RecursiveCharacterTextSplitter,
    settings: RAGSettings,
    metadata: dict[str, object] | None = None,
) -> list[str]:
    raw_chunks: list[str]
    # SemanticChunker 依赖 embedding，异常必须暴露；只有正常空结果才走 fallback。
    try:
        split_docs = semantic_splitter.create_documents([text])
        raw_chunks = [doc.page_content.strip() for doc in split_docs if doc.page_content.strip()]
    except Exception as exc:  # noqa: BLE001
        context = _format_splitter_error_context(metadata or {})
        raise RuntimeError(f"语义 Child 分块失败，{context}: {exc}") from exc

    if not raw_chunks:
        raw_chunks = [
            doc.page_content.strip()
            for doc in fallback_splitter.create_documents([text])
        ]
    # 单块已经超过目标 Child 大小时，再给中文友好的递归切分一次机会。
    elif len(raw_chunks) == 1 and len(raw_chunks[0]) > settings.child_chunk_size:
        raw_chunks = [
            doc.page_content.strip()
            for doc in fallback_splitter.create_documents([text])
        ]

    merged_chunks = _merge_small_chunks(raw_chunks, settings.semantic_min_chunk_chars)
    final_chunks: list[str] = []
    for chunk in merged_chunks:
        if len(chunk) <= settings.semantic_max_chunk_chars:
            final_chunks.append(chunk)
            continue
        split_docs = fallback_splitter.create_documents([chunk])
        final_chunks.extend(
            doc.page_content.strip()
            for doc in split_docs
            if doc.page_content.strip()
        )

    return _merge_small_chunks(
        final_chunks,
        settings.semantic_min_chunk_chars,
        max_chars=settings.semantic_max_chunk_chars,
    )


def _merge_small_chunks(
    chunks: list[str],
    min_chars: int,
    *,
    max_chars: int | None = None,
) -> list[str]:
    stripped = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not stripped:
        return []

    cleaned = [chunk for chunk in stripped if not _is_noise_child_chunk(chunk)]
    if not cleaned:
        # 保底保留一个 child，避免短 parent 因清理过度而完全不可检索。
        return [stripped[0]]

    merged: list[str] = []
    pending_prefix: list[str] = []
    for index, chunk in enumerate(cleaned):
        is_short = len(chunk) < min_chars
        has_next = index < len(cleaned) - 1

        if pending_prefix:
            if is_short and has_next:
                pending_prefix.append(chunk)
                continue
            combined = _join_child_parts([*pending_prefix, chunk])
            if not max_chars or len(combined) <= max_chars:
                merged.append(combined)
                pending_prefix = []
                continue
            _flush_pending_prefix(merged, pending_prefix, max_chars=max_chars)
            pending_prefix = []

        if is_short and _should_attach_to_next(chunk) and has_next:
            # 短标题更像“下一段正文的标签”，优先并到后文，避免错贴到上一段。
            pending_prefix.append(chunk)
            continue

        if is_short and _should_preserve_standalone_short(chunk):
            merged.append(chunk)
            continue

        if is_short and merged:
            combined = _join_child_parts([merged[-1], chunk])
            if not max_chars or len(combined) <= max_chars:
                merged[-1] = combined
                continue

        if is_short and has_next:
            pending_prefix.append(chunk)
            continue

        merged.append(chunk)

    if pending_prefix:
        _flush_pending_prefix(merged, pending_prefix, max_chars=max_chars)
    return merged


def _is_noise_child_chunk(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact or len(compact) > _CHILD_NOISE_MAX_CHARS:
        return False
    if _STRUCTURAL_ONLY_PATTERN.fullmatch(compact):
        return True
    return _SEMANTIC_CHAR_PATTERN.search(compact) is None


def _should_attach_to_next(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact:
        return False
    if _SHORT_HEADING_PATTERN.match(compact):
        return True
    return compact.endswith(("：", ":"))


def _should_attach_to_previous(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact:
        return False
    return compact.startswith(("- ", "* ", "+ "))


def _should_preserve_standalone_short(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact or not _SEMANTIC_CHAR_PATTERN.search(compact):
        return False
    if _should_attach_to_next(compact) or _should_attach_to_previous(compact):
        return False
    if len(compact) > 32:
        return False
    # 短命令/短结论本身可能就是高价值检索目标，明显带标点的残片则继续合并。
    return not re.search(r"[。！？；，、,.!?;:：()（）\[\]{}]", compact)


def _join_child_parts(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _format_splitter_error_context(metadata: dict[str, object]) -> str:
    file_name = str(metadata.get("file_name") or "<unknown>")
    heading_path = str(metadata.get("heading_path") or "<empty>")
    parent_id = str(metadata.get("parent_id") or "<missing>")
    return f"file_name={file_name}, heading_path={heading_path}, parent_id={parent_id}"


def _flush_pending_prefix(
    merged: list[str],
    pending_prefix: list[str],
    *,
    max_chars: int | None,
) -> None:
    pending_text = _join_child_parts(pending_prefix)
    if not pending_text:
        return
    if not merged:
        merged.append(pending_text)
        return
    combined = _join_child_parts([merged[-1], pending_text])
    if not max_chars or len(combined) <= max_chars:
        merged[-1] = combined
        return
    merged.append(pending_text)


def _resolve_heading_path(metadata: dict[str, object]) -> str:
    parts = [
        str(metadata[key]).strip()
        for key in ("h1", "h2", "h3")
        if metadata.get(key) and str(metadata[key]).strip()
    ]
    if parts:
        return " / ".join(parts)
    if metadata.get("heading_path"):
        return str(metadata["heading_path"])
    if metadata.get("section_title"):
        return str(metadata["section_title"])
    return ""
