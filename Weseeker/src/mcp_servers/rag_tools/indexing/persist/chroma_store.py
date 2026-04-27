from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_core.documents import Document


class ChromaChildStore:
    def __init__(self, persist_root: str, kb_name: str) -> None:
        self._persist_path = Path(persist_root) / kb_name
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_path))
        self._collection_name = "child_chunks"

    @property
    def persist_path(self) -> str:
        return str(self._persist_path)

    def rebuild(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception as exc:  # noqa: BLE001
            if not _is_collection_not_found_error(exc):
                raise RuntimeError(
                    "删除旧 Chroma collection 失败，"
                    f"collection={self._collection_name}, path={self._persist_path}, error={exc}"
                ) from exc

        collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        collection.upsert(
            ids=[str(document.metadata["child_id"]) for document in documents],
            documents=[document.page_content for document in documents],
            metadatas=[_sanitize_metadata(document.metadata) for document in documents],
            embeddings=embeddings,
        )

    def query(self, query_embedding: list[float], *, top_k: int) -> dict[str, list[object]]:
        collection = self._client.get_collection(self._collection_name)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )


def _sanitize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) and value not in {None, ""}
    }


def _is_collection_not_found_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return (
        "notfound" in name
        or "not_found" in name
        or "not found" in message
        or "does not exist" in message
        or "doesn't exist" in message
    )
