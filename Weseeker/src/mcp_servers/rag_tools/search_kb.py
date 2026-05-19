from __future__ import annotations

import time
from pathlib import Path

from langchain_core.documents import Document

from config.settings import get_settings
from mcp_servers.rag_tools.adapters import BatchedEmbedder, create_embedding_model
from mcp_servers.rag_tools.indexing.persist.bm25_store import BM25Store
from mcp_servers.rag_tools.indexing.persist.chroma_store import ChromaChildStore
from mcp_servers.rag_tools.indexing.persist.doc_store import ParentDocStore
from mcp_servers.rag_tools.retrieval import (
    BM25Retriever,
    HybridRetriever,
    ParentPromoter,
    VectorRetriever,
)
from mcp_servers.rag_tools.retrieval.query_profile import (
    RetrievalThresholds,
    analyze_query,
    thresholds_for_query,
)

_SNIPPET_MAX_CHARS = 800


def search_kb(
    query: str,
    kb_name: str,
    *,
    top_k: int = 5,
    candidate_k: int = 30,
) -> dict[str, object]:
    """Search an indexed knowledge base and return parent-level evidence.

    Args:
        query: 用户原始检索问题。当前只会去掉首尾空白，不做 Query Rewrite、HyDE、
            同义词扩展或 rerank。
        kb_name: 已完成离线索引的知识库名称。必须显式传入，例如 `test_kb_notes`。
        top_k: 最终返回的 parent evidence 数量。默认 5；必须大于 0。
        candidate_k: Vector / BM25 两路各自召回的 child candidate 数量。默认 30；
            实际执行时会使用 `max(top_k, candidate_k)`，避免候选数小于最终返回数。

    Returns:
        成功时返回 `ok=true` 的 dict，核心字段包括：
        - `query_profile`: 原始分词 `tokens`、过滤泛词后的 `content_tokens`，
          以及是否短 query 的 `is_short`。
        - `results`: parent-level evidence 列表。每项包含文件信息、章节、`parent_id`、
          `rank_score`、检索来源、细分分数、parent 摘要和命中的 child 摘要。
        - `diagnostics`: `top_k`、`candidate_k`、候选数量、实际阈值和耗时。

        失败时返回 `ok=false` 的 dict。参数错误、KB 索引产物缺失、embedding 服务不可用、
        parent 缺失会用明确 `error_type` 表达，不会伪装成空结果。

    Notes:
        `rank_score` 是 RRF 排名融合分，只用于排序诊断，不是相似度、置信度或百分制相关性。
        该函数是内部稳定 API；MCP、Agent 决策、Query Rewrite、HyDE 和 rerank
        都在外层或后续阶段处理。
    """
    normalized_query = query.strip()
    normalized_kb_name = kb_name.strip()
    if not normalized_query:
        return _error_response("INVALID_QUERY", "query 不能为空。")
    if not normalized_kb_name:
        return _error_response("INVALID_KB_NAME", "kb_name 不能为空。")
    if top_k < 1:
        return _error_response("INVALID_TOP_K", "top_k 必须大于 0。")
    if candidate_k < 1:
        return _error_response("INVALID_CANDIDATE_K", "candidate_k 必须大于 0。")

    started_at = time.perf_counter()
    resolved_candidate_k = max(top_k, candidate_k)
    profile = analyze_query(normalized_query)
    thresholds = thresholds_for_query(normalized_query)

    try:
        settings = get_settings()
        _validate_artifacts(settings, normalized_kb_name)
        embedding_model = create_embedding_model(settings.rag)
        embedder = BatchedEmbedder(
            embedding_model,
            batch_size=settings.rag.embedding_batch_size,
            max_retries=settings.rag.embedding_max_retries,
        )
        chroma_store = ChromaChildStore(settings.rag.chroma_persist_dir, normalized_kb_name)
        bm25_store = BM25Store(settings.rag.bm25_dir, normalized_kb_name)
        doc_store = ParentDocStore(settings.rag.docstore_dir, normalized_kb_name)

        vector_retriever = VectorRetriever(chroma_store, embedder)
        bm25_retriever = BM25Retriever(bm25_store)
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
        parent_promoter = ParentPromoter(doc_store)

        child_documents = hybrid_retriever.search(
            normalized_query,
            top_k=resolved_candidate_k,
            candidate_k=resolved_candidate_k,
        )
        # Query-aware parent gates stay here; HybridRetriever only gates child candidates.
        parent_documents = parent_promoter.promote(
            child_documents,
            top_k=top_k,
            min_vector_only_score=thresholds.vector_only_parent_min_score,
            min_bm25_only_score=thresholds.bm25_only_parent_min_score,
            min_vector_score=thresholds.vector_parent_min_score,
            min_bm25_score=thresholds.bm25_parent_min_score,
        )
    except FileNotFoundError as exc:
        return _error_response(
            "KB_NOT_INDEXED",
            "知识库索引不存在或不完整，请先构建 KB。",
            operator_hint=str(exc),
        )
    except ConnectionError as exc:
        return _error_response(
            "EMBEDDING_UNAVAILABLE",
            "embedding 服务不可用，无法完成向量检索。",
            retryable=True,
            operator_hint=str(exc),
        )
    except ValueError as exc:
        if "parent_id=" not in str(exc):
            raise
        return _error_response(
            "PARENT_NOT_FOUND",
            "检索命中了 child，但找不到对应 parent 文档。",
            operator_hint=str(exc),
        )

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return _success_response(
        kb_name=normalized_kb_name,
        query=normalized_query,
        tokens=list(profile.tokens),
        content_tokens=list(profile.content_tokens),
        is_short=profile.is_short,
        thresholds=thresholds,
        child_documents=child_documents,
        parent_documents=parent_documents,
        top_k=top_k,
        candidate_k=resolved_candidate_k,
        elapsed_ms=elapsed_ms,
    )


def _success_response(
    *,
    kb_name: str,
    query: str,
    tokens: list[str],
    content_tokens: list[str],
    is_short: bool,
    thresholds: RetrievalThresholds,
    child_documents: list[Document],
    parent_documents: list[Document],
    top_k: int,
    candidate_k: int,
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        "ok": True,
        "kb_name": kb_name,
        "query": query,
        "query_profile": {
            "tokens": tokens,
            "content_tokens": content_tokens,
            "is_short": is_short,
        },
        "results": [
            _format_result(document, rank)
            for rank, document in enumerate(parent_documents, start=1)
        ],
        "diagnostics": {
            "top_k": top_k,
            "candidate_k": candidate_k,
            "child_candidates": len(child_documents),
            "parent_results": len(parent_documents),
            "thresholds": _format_thresholds(thresholds),
            "elapsed_ms": {"total": elapsed_ms},
        },
    }


def _format_result(document: Document, rank: int) -> dict[str, object]:
    metadata = document.metadata
    rank_score = _float_or_none(metadata.get("parent_score"))
    if rank_score is None:
        rank_score = _float_or_none(metadata.get("rrf_score")) or 0.0

    return {
        "rank": rank,
        "file_name": _str_or_none(metadata.get("file_name")),
        "file_path": _str_or_none(metadata.get("file_path")),
        "doc_type": _str_or_none(metadata.get("doc_type")),
        "heading_path": _str_or_none(metadata.get("heading_path")),
        "section_title": _str_or_none(metadata.get("section_title")),
        "page_number": metadata.get("page_number"),
        "parent_id": _str_or_none(metadata.get("parent_id")),
        "rank_score": rank_score,
        "retrieval_sources": _string_list(metadata.get("retrieval_sources")),
        "scores": {
            "rrf_score": _float_or_none(metadata.get("rrf_score")),
            "vector_score": _float_or_none(metadata.get("vector_score")),
            "bm25_score": _float_or_none(metadata.get("bm25_score")),
            "vector_rank": _int_or_none(metadata.get("vector_rank")),
            "bm25_rank": _int_or_none(metadata.get("bm25_rank")),
        },
        "snippet": _snippet(document.page_content),
        "matched_children": _format_matched_children(metadata),
    }


def _format_matched_children(metadata: dict[str, object]) -> list[dict[str, object]]:
    child_ids = _string_list(metadata.get("matched_child_ids"))
    snippets = _string_list(metadata.get("matched_child_snippets"))
    best_child_id = str(metadata.get("best_child_id") or "")

    # ParentPromoter stores child snippets as diagnostics, not as full child records.
    children: list[dict[str, object]] = []
    for index, child_id in enumerate(child_ids):
        snippet = snippets[index] if index < len(snippets) else ""
        children.append(
            {
                "child_id": child_id,
                "snippet": snippet,
                "is_best": bool(best_child_id and child_id == best_child_id),
            }
        )
    if not children and best_child_id:
        children.append(
            {
                "child_id": best_child_id,
                "snippet": str(metadata.get("best_child_snippet") or ""),
                "is_best": True,
            }
        )
    return children


def _validate_artifacts(settings, kb_name: str) -> None:
    rag = settings.rag
    missing: list[str] = []
    chroma_path = Path(rag.chroma_persist_dir) / kb_name
    bm25_corpus_path = Path(rag.bm25_dir) / kb_name / "corpus.json"
    docstore_path = Path(rag.docstore_dir) / f"{kb_name}.sqlite"
    if not chroma_path.exists():
        missing.append(str(chroma_path))
    if not bm25_corpus_path.exists():
        missing.append(str(bm25_corpus_path))
    if not docstore_path.exists():
        missing.append(str(docstore_path))
    if missing:
        raise FileNotFoundError("缺少 RAG 索引产物: " + "; ".join(missing))


def _format_thresholds(thresholds: RetrievalThresholds) -> dict[str, object]:
    return {
        "vector_min_score": thresholds.vector_min_score,
        "bm25_child_min_score": thresholds.bm25_child_min_score,
        "vector_only_parent_min_score": thresholds.vector_only_parent_min_score,
        "bm25_only_parent_min_score": thresholds.bm25_only_parent_min_score,
        "vector_parent_min_score": thresholds.vector_parent_min_score,
        "bm25_parent_min_score": thresholds.bm25_parent_min_score,
    }


def _error_response(
    error_type: str,
    message: str,
    *,
    retryable: bool = False,
    user_hint: str | None = None,
    operator_hint: str | None = None,
) -> dict[str, object]:
    return {
        "ok": False,
        "error_type": error_type,
        "retryable": retryable,
        "message": message,
        "error": message,
        "user_hint": user_hint or message,
        "operator_hint": operator_hint or "",
    }


def _snippet(text: str, max_chars: int = _SNIPPET_MAX_CHARS) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
