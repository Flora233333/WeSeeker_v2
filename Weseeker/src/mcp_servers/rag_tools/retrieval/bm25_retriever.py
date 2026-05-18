from __future__ import annotations

from langchain_core.documents import Document

from mcp_servers.rag_tools.indexing.persist.bm25_store import BM25Store


class BM25Retriever:
    def __init__(self, store: BM25Store) -> None:
        self._store = store

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[Document]:
        if min_score is not None and min_score < 0:
            raise ValueError("min_score 不能小于 0。")

        documents: list[Document] = []
        for hit in self._store.query(query, top_k=top_k):
            if min_score is not None and hit.score < min_score:
                continue
            metadata = dict(hit.metadata)
            # 后续 Hybrid/RRF 阶段可直接读取 retrieval_source 与 bm25_score。
            metadata["bm25_score"] = hit.score
            metadata["retrieval_source"] = "bm25"
            metadata["rank"] = hit.rank
            metadata["bm25_rank"] = hit.rank
            documents.append(Document(page_content=hit.text, metadata=metadata))
        return documents
