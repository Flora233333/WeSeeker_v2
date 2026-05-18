from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import bm25s
import jieba
from langchain_core.documents import Document

jieba.setLogLevel(logging.WARNING)

_CORPUS_FILE = "corpus.json"
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


@dataclass(frozen=True)
class BM25Hit:
    rank: int
    score: float
    text: str
    metadata: dict[str, object]


class BM25Store:
    def __init__(self, persist_root: str, kb_name: str) -> None:
        self._persist_path = Path(persist_root) / kb_name
        self._bm25 = None
        self._corpus: list[dict[str, object]] | None = None

    @property
    def persist_path(self) -> str:
        return str(self._persist_path)

    def rebuild(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("BM25 rebuild 至少需要 1 个 child document。")

        self._persist_path.mkdir(parents=True, exist_ok=True)
        corpus = [_build_corpus_record(document) for document in documents]
        index_texts = [_build_index_text(document) for document in documents]
        tokenized_corpus = [_tokenize(text) for text in index_texts]

        bm25 = bm25s.BM25()
        bm25.index(tokenized_corpus, show_progress=False)
        # bm25s 负责保存稀疏矩阵和词表；corpus.json 保留 WeSeeker metadata。
        bm25.save(str(self._persist_path), show_progress=False)
        self._write_corpus(corpus)

        self._bm25 = bm25
        self._corpus = corpus

    def load(self) -> None:
        corpus_path = self._persist_path / _CORPUS_FILE
        if not self._persist_path.exists() or not corpus_path.exists():
            raise FileNotFoundError(
                "BM25 index 不存在，请先重建知识库，"
                f"kb_path={self._persist_path}"
            )

        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        if not isinstance(corpus, list):
            raise ValueError(f"BM25 corpus 格式无效，必须是 list: {corpus_path}")

        self._bm25 = bm25s.BM25.load(str(self._persist_path), load_corpus=False)
        self._corpus = [_validate_corpus_record(record, corpus_path) for record in corpus]

    def query(self, query: str, *, top_k: int) -> list[BM25Hit]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("BM25 query 不能为空。")
        if top_k < 1:
            raise ValueError("top_k 必须大于 0。")

        if self._bm25 is None or self._corpus is None:
            self.load()
        if not self._corpus:
            return []

        query_tokens = [_tokenize(normalized_query)]
        results = self._bm25.retrieve(
            query_tokens,
            k=min(top_k, len(self._corpus)),
            show_progress=False,
        )

        hits: list[BM25Hit] = []
        for rank, (doc_index, score) in enumerate(
            zip(results.documents[0], results.scores[0], strict=False),
            start=1,
        ):
            score_value = float(score)
            if score_value <= 0:
                continue
            record = self._corpus[int(doc_index)]
            hits.append(
                BM25Hit(
                    rank=rank,
                    score=score_value,
                    text=str(record["text"]),
                    metadata=dict(record["metadata"]),
                )
            )
        return hits

    def _write_corpus(self, corpus: list[dict[str, object]]) -> None:
        path = self._persist_path / _CORPUS_FILE
        path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_corpus_record(document: Document) -> dict[str, object]:
    metadata = _sanitize_metadata(document.metadata)
    _require_metadata(metadata, "child_id")
    _require_metadata(metadata, "parent_id")
    text = document.page_content.strip()
    if not text:
        raise ValueError(f"BM25 child 文本为空，child_id={metadata.get('child_id')}")
    return {"text": text, "metadata": metadata}


def _build_index_text(document: Document) -> str:
    heading_path = str(document.metadata.get("heading_path") or "").strip()
    if not heading_path:
        return document.page_content
    # heading_path 只参与 BM25 索引文本，不改写实际 corpus child 原文。
    return f"{heading_path}\n\n{document.page_content}"


def _sanitize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) and value not in {None, ""}
    }


def _require_metadata(metadata: dict[str, object], key: str) -> None:
    if not str(metadata.get(key) or "").strip():
        raise ValueError(f"BM25 child metadata 缺少 {key}。")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in jieba.cut(text, cut_all=False):
        for token in _TOKEN_PATTERN.findall(raw_token.casefold()):
            if token.strip():
                tokens.append(token.strip())
    return tokens


def _validate_corpus_record(record: object, corpus_path: Path) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError(f"BM25 corpus 记录必须是 object: {corpus_path}")
    text = record.get("text")
    metadata = record.get("metadata")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"BM25 corpus 记录缺少 text: {corpus_path}")
    if not isinstance(metadata, dict):
        raise ValueError(f"BM25 corpus 记录缺少 metadata: {corpus_path}")
    _require_metadata(metadata, "child_id")
    _require_metadata(metadata, "parent_id")
    return {"text": text, "metadata": metadata}
