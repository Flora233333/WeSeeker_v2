from __future__ import annotations

from langchain_core.documents import Document

from mcp_servers.rag_tools.adapters import BatchedEmbedder
from mcp_servers.rag_tools.indexing.persist.chroma_store import ChromaChildStore


class VectorRetriever:
    def __init__(self, chroma_store: ChromaChildStore, embedder: BatchedEmbedder) -> None:
        self._chroma_store = chroma_store
        self._embedder = embedder

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[Document]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("vector query 不能为空。")
        if top_k < 1:
            raise ValueError("top_k 必须大于 0。")
        if min_score is not None and min_score < 0:
            raise ValueError("min_score 不能小于 0。")

        query_embedding = self._embedder.embed_query(normalized_query)
        raw_result = self._chroma_store.query(query_embedding, top_k=top_k)
        documents = _first_nested_list(raw_result.get("documents"))
        metadatas = _first_nested_list(raw_result.get("metadatas"))
        distances = _first_nested_list(raw_result.get("distances"))

        results: list[Document] = []
        for rank, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances, strict=False),
            start=1,
        ):
            distance_value = float(distance)
            payload = dict(metadata) if isinstance(metadata, dict) else {}
            # Keep vector-specific fields explicit so RRF and diagnostics do not parse labels.
            payload["retrieval_source"] = "vector"
            payload["rank"] = rank
            payload["vector_rank"] = rank
            payload["vector_distance"] = distance_value
            vector_score = max(0.0, 1.0 - distance_value)
            if min_score is not None and vector_score < min_score:
                continue
            payload["vector_score"] = vector_score
            results.append(Document(page_content=str(document), metadata=payload))
        return results


def _first_nested_list(value: object) -> list[object]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    if not isinstance(first, list):
        return []
    return first
