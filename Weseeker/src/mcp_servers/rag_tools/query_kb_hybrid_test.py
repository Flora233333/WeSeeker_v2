"""Manual hybrid retrieval checker for human review.

Usage:
    python src/mcp_servers/rag_tools/query_kb_hybrid_test.py --kb-name <kb_name> --query <query>

This script does not rebuild the KB. It reads existing Chroma, BM25 and ParentDocStore
artifacts, then prints parent-level evidence after child-level RRF fusion.
"""

from __future__ import annotations

import argparse
import sys

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

if hasattr(sys.stdout, "reconfigure"):
    # Keep the script usable in Windows terminals that still default to gbk.
    sys.stdout.reconfigure(errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 kb_name 和 query 查询本地 Hybrid 知识库索引")
    parser.add_argument("--kb-name")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--show", choices=["child", "parent", "both"], default="child")
    args = parser.parse_args()

    kb_name = (args.kb_name or input("kb_name: ")).strip()
    query = (args.query or input("query: ")).strip()
    top_k = max(1, args.top_k)
    candidate_k = max(1, args.candidate_k)
    show_mode = args.show

    if not kb_name:
        raise ValueError("kb_name 不能为空。")
    if not query:
        raise ValueError("query 不能为空。")

    settings = get_settings()
    embedding_model = create_embedding_model(settings.rag)
    embedder = BatchedEmbedder(
        embedding_model,
        batch_size=settings.rag.embedding_batch_size,
        max_retries=settings.rag.embedding_max_retries,
    )
    chroma_store = ChromaChildStore(settings.rag.chroma_persist_dir, kb_name)
    bm25_store = BM25Store(settings.rag.bm25_dir, kb_name)
    doc_store = ParentDocStore(settings.rag.docstore_dir, kb_name)

    vector_retriever = VectorRetriever(chroma_store, embedder)
    bm25_retriever = BM25Retriever(bm25_store)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    parent_promoter = ParentPromoter(doc_store)
    child_results = hybrid_retriever.search(query, top_k=candidate_k, candidate_k=candidate_k)
    parent_results = parent_promoter.promote(child_results, top_k=top_k)

    _print_line(f"KB: {kb_name}")
    _print_line(f"Query: {query}")
    _print_line(f"Top K: {top_k}")
    _print_line(f"Candidate K: {candidate_k}")
    _print_line(f"Show Mode: {show_mode}")
    _print_line("")

    if not parent_results:
        _print_line(
            "没有查到 Hybrid 结果。请确认索引已构建，或当前 query 只有低相关向量结果。"
        )
        return

    for index, document in enumerate(parent_results, start=1):
        metadata = document.metadata
        _print_line(f"[{index}] {metadata.get('file_name', '')}")
        _print_line(f"章节: {metadata.get('heading_path') or '<无>'}")
        sources = ", ".join(str(source) for source in metadata.get("retrieval_sources", []))
        _print_line(f"来源: {sources}")
        _print_line(f"Parent分数: {float(metadata.get('parent_score') or 0.0):.4f}")
        _print_line(f"命中Child数: {metadata.get('matched_child_count', 0)}")
        _print_line(f"最佳Child: {metadata.get('best_child_id', '')}")
        _print_line(
            "Rank: vector={vector_rank} bm25={bm25_rank}".format(
                vector_rank=metadata.get("vector_rank", "-"),
                bm25_rank=metadata.get("bm25_rank", "-"),
            )
        )
        if show_mode in {"child", "both"}:
            _print_line("命中子段摘要:")
            for snippet in metadata.get("matched_child_snippets", []):
                _print_line(f"- {_format_paragraph(str(snippet))}")
        if show_mode in {"parent", "both"}:
            _print_line("所属父段:")
            _print_line(_format_paragraph(document.page_content))
        _print_line("-" * 80)


def _format_paragraph(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.rstrip() for line in normalized.splitlines()]
    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if line.strip():
            compact_lines.append(line)
            previous_blank = False
            continue
        if not previous_blank:
            compact_lines.append("")
        previous_blank = True
    return "\n".join(compact_lines)


def _print_line(text: str) -> None:
    print(text)


if __name__ == "__main__":
    main()
