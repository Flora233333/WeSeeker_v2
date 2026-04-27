from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from langchain_core.embeddings import Embeddings


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    batch_count: int
    dimension: int


class BatchedEmbedder:
    def __init__(
        self,
        embeddings: Embeddings,
        *,
        batch_size: int,
        max_retries: int,
    ) -> None:
        self._embeddings = embeddings
        self._batch_size = max(1, batch_size)
        self._max_retries = max(0, max_retries)

    def embed_documents(
        self,
        texts: list[str],
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> EmbedResult:
        vectors: list[list[float]] = []
        total = len(texts)
        done = 0
        batch_count = 0

        for batch_start in range(0, total, self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            batch_vectors = self._embed_batch_with_retry(batch)
            vectors.extend(batch_vectors)
            batch_count += 1
            done += len(batch)
            if on_progress is not None:
                on_progress(done, total)

        dimension = len(vectors[0]) if vectors else 0
        return EmbedResult(vectors=vectors, batch_count=batch_count, dimension=dimension)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._embeddings.embed_documents(batch)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self._max_retries:
                    break
                time.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"embedding 批处理失败: {last_error}") from last_error
