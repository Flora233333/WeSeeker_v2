from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document

_SNIPPET_MAX_CHARS = 240


class _ParentStore(Protocol):
    def get(self, parent_id: str) -> Document | None: ...


class ParentPromoter:
    def __init__(
        self,
        doc_store: _ParentStore,
        *,
        min_vector_only_score: float = 0.50,
        min_bm25_only_score: float = 5.0,
        min_vector_score: float = 0.45,
        min_bm25_score: float = 4.0,
    ) -> None:
        if min_vector_only_score < 0:
            raise ValueError("min_vector_only_score 不能小于 0。")
        if min_bm25_only_score < 0:
            raise ValueError("min_bm25_only_score 不能小于 0。")
        if min_vector_score < 0:
            raise ValueError("min_vector_score 不能小于 0。")
        if min_bm25_score < 0:
            raise ValueError("min_bm25_score 不能小于 0。")
        self._doc_store = doc_store
        self._min_vector_only_score = min_vector_only_score
        self._min_bm25_only_score = min_bm25_only_score
        self._min_vector_score = min_vector_score
        self._min_bm25_score = min_bm25_score

    def promote(
        self,
        child_documents: list[Document],
        *,
        top_k: int = 10,
        min_vector_only_score: float | None = None,
        min_bm25_only_score: float | None = None,
        min_vector_score: float | None = None,
        min_bm25_score: float | None = None,
    ) -> list[Document]:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0。")
        vector_only_score = _resolve_min_score(
            min_vector_only_score,
            self._min_vector_only_score,
            "min_vector_only_score",
        )
        bm25_only_score = _resolve_min_score(
            min_bm25_only_score,
            self._min_bm25_only_score,
            "min_bm25_only_score",
        )
        vector_score = _resolve_min_score(
            min_vector_score,
            self._min_vector_score,
            "min_vector_score",
        )
        bm25_score = _resolve_min_score(
            min_bm25_score,
            self._min_bm25_score,
            "min_bm25_score",
        )

        children_by_parent: dict[str, list[Document]] = {}
        for child in child_documents:
            parent_id = _require_metadata(child.metadata, "parent_id")
            _require_metadata(child.metadata, "child_id")
            children_by_parent.setdefault(parent_id, []).append(child)

        parent_documents: list[Document] = []
        for parent_id, children in children_by_parent.items():
            retrieval_sources = _merge_retrieval_sources(children)
            if not _passes_relevance_gate(
                children,
                retrieval_sources,
                min_vector_only_score=vector_only_score,
                min_bm25_only_score=bm25_only_score,
                min_vector_score=vector_score,
                min_bm25_score=bm25_score,
            ):
                continue

            parent = self._doc_store.get(parent_id)
            if parent is None:
                raise ValueError(f"ParentPromoter 找不到 parent_id={parent_id} 的 parent 文档。")

            best_child = max(children, key=_child_score)
            metadata = dict(parent.metadata)
            parent_score = _child_score(best_child)
            # Parent is the final evidence unit; child snippets stay as diagnostics.
            metadata.update(
                {
                    "retrieval_source": "hybrid_parent",
                    "parent_score": parent_score,
                    "rrf_score": parent_score,
                    "matched_child_count": len(children),
                    "matched_child_ids": [str(child.metadata["child_id"]) for child in children],
                    "matched_child_snippets": [_snippet(child.page_content) for child in children],
                    "best_child_id": str(best_child.metadata["child_id"]),
                    "best_child_snippet": _snippet(best_child.page_content),
                    "retrieval_sources": retrieval_sources,
                }
            )
            _copy_best_source_fields(metadata, children)
            parent_documents.append(Document(page_content=parent.page_content, metadata=metadata))

        ranked = sorted(
            parent_documents,
            key=lambda document: (
                -float(document.metadata["parent_score"]),
                str(document.metadata["parent_id"]),
            ),
        )[:top_k]
        for rank, document in enumerate(ranked, start=1):
            document.metadata["parent_rank"] = rank
            document.metadata["rank"] = rank
        return ranked


def _require_metadata(metadata: dict[str, object], key: str) -> str:
    value = str(metadata.get(key) or "").strip()
    if not value:
        raise ValueError(f"ParentPromoter child metadata 缺少 {key}。")
    return value


def _resolve_min_score(value: float | None, default: float, name: str) -> float:
    resolved = default if value is None else value
    if resolved < 0:
        raise ValueError(f"{name} 不能小于 0。")
    return resolved


def _child_score(document: Document) -> float:
    value = document.metadata.get("rrf_score")
    if value is None:
        raise ValueError("ParentPromoter child metadata 缺少 rrf_score。")
    return float(value)


def _merge_retrieval_sources(children: list[Document]) -> list[str]:
    sources: list[str] = []
    for child in children:
        raw_sources = child.metadata.get("retrieval_sources")
        if isinstance(raw_sources, list):
            candidates = [str(source) for source in raw_sources]
        else:
            candidates = [str(child.metadata.get("retrieval_source") or "")]
        for source in candidates:
            if source and source != "hybrid" and source not in sources:
                sources.append(source)
    return sources


def _copy_best_source_fields(target: dict[str, object], children: list[Document]) -> None:
    vector_ranks = _numeric_values(children, "vector_rank")
    bm25_ranks = _numeric_values(children, "bm25_rank")
    vector_scores = _numeric_values(children, "vector_score")
    vector_distances = _numeric_values(children, "vector_distance")
    bm25_scores = _numeric_values(children, "bm25_score")

    if vector_ranks:
        target["vector_rank"] = int(min(vector_ranks))
    if bm25_ranks:
        target["bm25_rank"] = int(min(bm25_ranks))
    if vector_scores:
        target["vector_score"] = max(vector_scores)
    if vector_distances:
        target["vector_distance"] = min(vector_distances)
    if bm25_scores:
        target["bm25_score"] = max(bm25_scores)


def _numeric_values(children: list[Document], key: str) -> list[float]:
    values: list[float] = []
    for child in children:
        if key in child.metadata:
            values.append(float(child.metadata[key]))
    return values


def _passes_relevance_gate(
    children: list[Document],
    retrieval_sources: list[str],
    *,
    min_vector_only_score: float,
    min_bm25_only_score: float,
    min_vector_score: float,
    min_bm25_score: float,
) -> bool:
    has_vector = "vector" in retrieval_sources
    has_bm25 = "bm25" in retrieval_sources
    if not has_vector and not has_bm25:
        return True

    vector_scores = _numeric_values(children, "vector_score") if has_vector else []
    bm25_scores = _numeric_values(children, "bm25_score") if has_bm25 else []
    if has_vector and not vector_scores:
        raise ValueError("ParentPromoter vector child metadata 缺少 vector_score。")
    if has_bm25 and not bm25_scores:
        raise ValueError("ParentPromoter BM25 child metadata 缺少 bm25_score。")

    if has_vector and not has_bm25:
        # Vector-only low-score hits are usually embedding nearest-neighbor noise.
        return max(vector_scores) >= min_vector_only_score
    if has_bm25 and not has_vector:
        return max(bm25_scores) >= min_bm25_only_score

    return max(vector_scores) >= min_vector_score or max(bm25_scores) >= min_bm25_score


def _snippet(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= _SNIPPET_MAX_CHARS:
        return compact
    return compact[:_SNIPPET_MAX_CHARS].rstrip() + "..."
