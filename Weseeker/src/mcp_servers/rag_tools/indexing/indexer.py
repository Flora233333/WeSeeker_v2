from __future__ import annotations

import re
import time
from dataclasses import dataclass

from langchain_core.documents import Document

from config.settings import KBConfig, RAGSettings
from mcp_servers.rag_tools.adapters import BatchedEmbedder, create_embedding_model
from mcp_servers.rag_tools.indexing.enhancer import enhance_documents
from mcp_servers.rag_tools.indexing.llm_enhancement_cache import load_cache
from mcp_servers.rag_tools.indexing.llm_enhancer import (
    _build_llm_enhanced_embedding_text,
    build_initial_llm_enhancement_stats,
    enhance_file_chunks_with_llm,
    merge_llm_enhancement_stats,
)
from mcp_servers.rag_tools.indexing.loaders import load_file_documents
from mcp_servers.rag_tools.indexing.persist import (
    BM25Store,
    ChromaChildStore,
    ParentDocStore,
    load_manifest,
    write_manifest,
)
from mcp_servers.rag_tools.indexing.scanner import FileRecord, ScanResult, scan_kb
from mcp_servers.rag_tools.indexing.splitter import split_documents

_EMBEDDING_HEADING_PREFIX_MAX_CHARS = 160
_EMBEDDING_HEADING_MATCH_WINDOW = 300
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_DECORATION_PATTERN = re.compile(r"[#*_`>\[\]()]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class IndexBuildResult:
    kb_name: str
    scan_result: ScanResult
    parent_documents: list[Document]
    child_documents: list[Document]
    embedding_batches: int
    embedding_dimension: int
    chroma_path: str
    docstore_path: str
    bm25_path: str
    manifest_path: str
    elapsed_ms: int
    llm_enhancement_stats: dict[str, object]


def build_kb_index(kb_config: KBConfig, settings: RAGSettings) -> IndexBuildResult:
    started_at = time.perf_counter()
    manifest = load_manifest(settings.manifest_dir, kb_config.name)
    scan_result = scan_kb(
        kb_config,
        manifest_files=manifest.get("files") if isinstance(manifest.get("files"), dict) else None,
        max_file_size_mb=settings.max_file_size_mb,
    )

    embedding_model = create_embedding_model(settings)
    batched_embedder = BatchedEmbedder(
        embedding_model,
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
    )

    parent_documents: list[Document] = []
    child_documents: list[Document] = []
    file_manifest: dict[str, dict[str, object]] = {}
    llm_enhancement_stats = build_initial_llm_enhancement_stats(settings)
    llm_cache = (
        load_cache(settings.llm_enhancement.cache_dir, kb_config.name)
        if settings.llm_enhancement.enabled and settings.llm_enhancement.cache_enabled
        else None
    )

    for file_record in scan_result.files:
        loaded_documents = load_file_documents(file_record)
        enhanced_documents = enhance_documents(
            loaded_documents,
            kb_name=kb_config.name,
            file_record=file_record,
        )
        file_parents, file_children = split_documents(
            enhanced_documents,
            embeddings=embedding_model,
            settings=settings,
        )
        if settings.llm_enhancement.enabled:
            llm_result = enhance_file_chunks_with_llm(
                file_record,
                file_parents,
                file_children,
                settings=settings,
                kb_name=kb_config.name,
                cache_records=llm_cache,
            )
            file_parents = llm_result.parent_documents
            file_children = llm_result.child_documents
            llm_enhancement_stats = merge_llm_enhancement_stats(
                llm_enhancement_stats,
                llm_result.stats,
            )
        parent_documents.extend(file_parents)
        child_documents.extend(file_children)
        file_manifest[file_record.path_str] = _build_file_manifest_record(
            file_record,
            file_parents,
            file_children,
        )

    child_texts = [_build_child_embedding_text(document) for document in child_documents]
    embed_result = batched_embedder.embed_documents(child_texts)
    if embed_result.dimension and settings.embedding_dimension != embed_result.dimension:
        raise ValueError(
            "embedding 维度不匹配：配置为 "
            f"{settings.embedding_dimension}，实际为 {embed_result.dimension}"
        )

    chroma_store = ChromaChildStore(settings.chroma_persist_dir, kb_config.name)
    chroma_store.rebuild(child_documents, embed_result.vectors)

    doc_store = ParentDocStore(settings.docstore_dir, kb_config.name)
    doc_store.rebuild(parent_documents)

    bm25_store = BM25Store(settings.bm25_dir, kb_config.name)
    bm25_store.rebuild(child_documents)

    manifest_payload = {
        "kb_name": kb_config.name,
        "root": kb_config.root,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": embed_result.dimension or settings.embedding_dimension,
        "embedder_signature": _build_embedder_signature(settings, embed_result.dimension),
        "files": file_manifest,
        "skipped": [skipped.__dict__ for skipped in scan_result.skipped],
        "stats": {
            "files_total": len(scan_result.files),
            "parents_total": len(parent_documents),
            "children_total": len(child_documents),
        },
        "bm25": {
            "path": bm25_store.persist_path,
            "corpus_count": len(child_documents),
            "tokenizer": "jieba",
            "index_text": "heading_path + child_content",
        },
        "llm_enhancement": llm_enhancement_stats,
    }
    manifest_path = write_manifest(settings.manifest_dir, kb_config.name, manifest_payload)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    return IndexBuildResult(
        kb_name=kb_config.name,
        scan_result=scan_result,
        parent_documents=parent_documents,
        child_documents=child_documents,
        embedding_batches=embed_result.batch_count,
        embedding_dimension=embed_result.dimension,
        chroma_path=chroma_store.persist_path,
        docstore_path=doc_store.db_path,
        bm25_path=bm25_store.persist_path,
        manifest_path=manifest_path,
        elapsed_ms=elapsed_ms,
        llm_enhancement_stats=llm_enhancement_stats,
    )


def _build_file_manifest_record(
    file_record: FileRecord,
    parent_documents: list[Document],
    child_documents: list[Document],
) -> dict[str, object]:
    return {
        "mtime": file_record.mtime,
        "size": file_record.size,
        "parent_ids": [str(document.metadata["parent_id"]) for document in parent_documents],
        "child_count": len(child_documents),
    }


def _build_embedder_signature(settings: RAGSettings, dimension: int) -> str:
    resolved_dimension = dimension or settings.embedding_dimension
    return f"{settings.embedding_provider}:{settings.embedding_model}:{resolved_dimension}"


def _build_child_embedding_text(document: Document) -> str:
    if document.metadata.get("llm_enhancement_model"):
        llm_text = _build_llm_enhanced_embedding_text(document)
        heading_prefix = _build_markdown_heading_embedding_prefix(document)
        if heading_prefix:
            return f"{heading_prefix}\n\n{llm_text}"
        return llm_text

    heading_prefix = _build_markdown_heading_embedding_prefix(document)
    if not heading_prefix:
        return document.page_content
    return f"{heading_prefix}\n\n{document.page_content}"


def _build_markdown_heading_embedding_prefix(document: Document) -> str:
    text = document.page_content
    metadata = document.metadata
    if str(metadata.get("doc_type") or "") != "md":
        return ""
    heading_path = str(metadata.get("heading_path") or "").strip()
    if not heading_path:
        return ""

    missing_segments = _find_missing_heading_segments(text, heading_path)
    if not missing_segments:
        return ""

    # 只给 embedding 输入补充缺失的 Markdown 标题上下文，不改实际存储的 child 文本。
    prefix = _trim_heading_prefix(" / ".join(missing_segments))
    if not prefix:
        return ""
    return f"章节：{prefix}"


def _find_missing_heading_segments(text: str, heading_path: str) -> list[str]:
    normalized_text = _normalize_heading_match_text(text[:_EMBEDDING_HEADING_MATCH_WINDOW])
    missing: list[str] = []
    clean_heading_path = _MARKDOWN_LINK_PATTERN.sub(r"\1", heading_path)
    for raw_segment in clean_heading_path.split("/"):
        cleaned_segment = _clean_heading_segment(raw_segment)
        if not cleaned_segment:
            continue
        if _normalize_heading_match_text(cleaned_segment) in normalized_text:
            continue
        missing.append(cleaned_segment)
    return missing


def _clean_heading_segment(text: str) -> str:
    cleaned = _MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    cleaned = _MARKDOWN_DECORATION_PATTERN.sub(" ", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip(" /")


def _normalize_heading_match_text(text: str) -> str:
    return _clean_heading_segment(text).casefold()


def _trim_heading_prefix(text: str) -> str:
    compact = _WHITESPACE_PATTERN.sub(" ", text).strip()
    if len(compact) <= _EMBEDDING_HEADING_PREFIX_MAX_CHARS:
        return compact
    return compact[:_EMBEDDING_HEADING_PREFIX_MAX_CHARS].rstrip(" /")
