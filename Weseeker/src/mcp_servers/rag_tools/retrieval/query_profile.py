from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import jieba
from langchain_core.documents import Document

jieba.setLogLevel(logging.WARNING)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")
_SHORT_QUERY_MAX_TOKENS = 3
_SHORT_QUERY_BM25_PARENT_MIN_SCORE = 2.0


@dataclass(frozen=True)
class QueryProfile:
    query: str
    tokens: tuple[str, ...]
    is_short: bool


@dataclass(frozen=True)
class RetrievalThresholds:
    vector_min_score: float | None
    bm25_child_min_score: float | None
    vector_only_parent_min_score: float
    bm25_only_parent_min_score: float
    vector_parent_min_score: float
    bm25_parent_min_score: float


def analyze_query(query: str) -> QueryProfile:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空。")
    tokens = tuple(_tokenize(normalized_query))
    return QueryProfile(
        query=normalized_query,
        tokens=tokens,
        is_short=0 < len(tokens) <= _SHORT_QUERY_MAX_TOKENS,
    )


def thresholds_for_query(
    query: str,
    *,
    vector_min_score: float | None = 0.45,
    bm25_child_min_score: float | None = 4.0,
    vector_only_parent_min_score: float = 0.50,
    bm25_only_parent_min_score: float = 5.0,
    vector_parent_min_score: float = 0.45,
    bm25_parent_min_score: float = 4.0,
) -> RetrievalThresholds:
    profile = analyze_query(query)
    if profile.is_short:
        return RetrievalThresholds(
            vector_min_score=vector_min_score,
            bm25_child_min_score=None,
            vector_only_parent_min_score=vector_only_parent_min_score,
            bm25_only_parent_min_score=_SHORT_QUERY_BM25_PARENT_MIN_SCORE,
            vector_parent_min_score=vector_parent_min_score,
            bm25_parent_min_score=_SHORT_QUERY_BM25_PARENT_MIN_SCORE,
        )
    return RetrievalThresholds(
        vector_min_score=vector_min_score,
        bm25_child_min_score=bm25_child_min_score,
        vector_only_parent_min_score=vector_only_parent_min_score,
        bm25_only_parent_min_score=bm25_only_parent_min_score,
        vector_parent_min_score=vector_parent_min_score,
        bm25_parent_min_score=bm25_parent_min_score,
    )


def has_query_token_overlap(query: str, document: Document) -> bool:
    tokens = analyze_query(query).tokens
    if not tokens:
        return False
    haystack = _document_search_text(document)
    return any(token in haystack for token in tokens)


def _document_search_text(document: Document) -> str:
    metadata = document.metadata
    parts = [
        document.page_content,
        str(metadata.get("heading_path") or ""),
        str(metadata.get("file_name") or ""),
    ]
    return "\n".join(parts).casefold()


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in jieba.cut(text, cut_all=False):
        for token in _TOKEN_PATTERN.findall(raw_token.casefold()):
            stripped = token.strip()
            if stripped:
                tokens.append(stripped)
    return tokens
