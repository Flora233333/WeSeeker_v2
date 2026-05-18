from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document

from mcp_servers.rag_tools.retrieval.query_profile import (
    analyze_query,
    has_query_token_overlap,
    thresholds_for_query,
)


class _ChildRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[Document]: ...


_DEFAULT_MIN_VECTOR_SCORE = 0.45
_DEFAULT_MIN_BM25_SCORE = 4.0


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: _ChildRetriever,
        bm25_retriever: _ChildRetriever,
        *,
        rrf_k: int = 60,
        min_vector_score: float | None = _DEFAULT_MIN_VECTOR_SCORE,
        min_bm25_score: float | None = _DEFAULT_MIN_BM25_SCORE,
        query_aware_thresholds: bool = True,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k 必须大于 0。")
        if min_vector_score is not None and min_vector_score < 0:
            raise ValueError("min_vector_score 不能小于 0。")
        if min_bm25_score is not None and min_bm25_score < 0:
            raise ValueError("min_bm25_score 不能小于 0。")
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._rrf_k = rrf_k
        self._min_vector_score = min_vector_score
        self._min_bm25_score = min_bm25_score
        self._query_aware_thresholds = query_aware_thresholds

    def search(self, query: str, *, top_k: int = 10, candidate_k: int = 30) -> list[Document]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("hybrid query 不能为空。")
        if top_k < 1:
            raise ValueError("top_k 必须大于 0。")
        if candidate_k < 1:
            raise ValueError("candidate_k 必须大于 0。")

        thresholds = thresholds_for_query(
            normalized_query,
            vector_min_score=self._min_vector_score,
            bm25_child_min_score=self._min_bm25_score,
        )
        vector_min_score = (
            thresholds.vector_min_score
            if self._query_aware_thresholds
            else self._min_vector_score
        )
        bm25_min_score = (
            thresholds.bm25_child_min_score
            if self._query_aware_thresholds
            else self._min_bm25_score
        )

        merged: dict[str, Document] = {}
        self._merge_source_results(
            merged,
            self._vector_retriever.search(
                normalized_query,
                top_k=candidate_k,
                min_score=vector_min_score,
            ),
            source="vector",
        )
        bm25_documents = self._bm25_retriever.search(
            normalized_query,
            top_k=candidate_k,
            min_score=bm25_min_score,
        )
        if self._query_aware_thresholds and analyze_query(normalized_query).is_short:
            bm25_documents = [
                document
                for document in bm25_documents
                if has_query_token_overlap(normalized_query, document)
            ]
        self._merge_source_results(
            merged,
            bm25_documents,
            source="bm25",
        )

        ranked = sorted(
            merged.values(),
            key=lambda document: (
                -float(document.metadata["rrf_score"]),
                str(document.metadata["child_id"]),
            ),
        )[:top_k]
        for rank, document in enumerate(ranked, start=1):
            document.metadata["rank"] = rank
        return ranked

    def _merge_source_results(
        self,
        merged: dict[str, Document],
        documents: list[Document],
        *,
        source: str,
    ) -> None:
        for fallback_rank, document in enumerate(documents, start=1):
            metadata = dict(document.metadata)
            child_id = _require_metadata(metadata, "child_id")
            source_rank = _resolve_source_rank(metadata, source, fallback_rank)
            contribution = 1.0 / (self._rrf_k + source_rank)

            if child_id not in merged:
                metadata["retrieval_source"] = "hybrid"
                metadata["retrieval_sources"] = [source]
                metadata[f"{source}_rank"] = source_rank
                metadata["rrf_score"] = contribution
                merged[child_id] = Document(page_content=document.page_content, metadata=metadata)
                continue

            existing = merged[child_id]
            existing_metadata = existing.metadata
            sources = list(existing_metadata.get("retrieval_sources") or [])
            if source not in sources:
                sources.append(source)
            existing_metadata["retrieval_sources"] = sources
            existing_metadata[f"{source}_rank"] = min(
                _positive_int(existing_metadata.get(f"{source}_rank")) or source_rank,
                source_rank,
            )
            if f"{source}_score" in metadata:
                existing_metadata[f"{source}_score"] = metadata[f"{source}_score"]
            if f"{source}_distance" in metadata:
                existing_metadata[f"{source}_distance"] = metadata[f"{source}_distance"]
            existing_metadata["rrf_score"] = float(existing_metadata["rrf_score"]) + contribution


def _require_metadata(metadata: dict[str, object], key: str) -> str:
    value = str(metadata.get(key) or "").strip()
    if not value:
        raise ValueError(f"hybrid result metadata 缺少 {key}。")
    return value


def _resolve_source_rank(metadata: dict[str, object], source: str, fallback_rank: int) -> int:
    return (
        _positive_int(metadata.get(f"{source}_rank"))
        or _positive_int(metadata.get("rank"))
        or fallback_rank
    )


def _positive_int(value: object) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None
