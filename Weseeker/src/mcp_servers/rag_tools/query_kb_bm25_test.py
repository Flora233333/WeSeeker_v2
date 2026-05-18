"""Manual BM25 retrieval checker for human review.

Usage:
    python src/mcp_servers/rag_tools/query_kb_bm25_test.py --kb-name <kb_name> --query <query>

Common commands:
    python src/mcp_servers/rag_tools/query_kb_bm25_test.py \
        --kb-name test_kb_notes \
        --query "LangChain 有哪些坑" \
        --top-k 5 \
        --show both
"""

from __future__ import annotations

import argparse
import sys

from config.settings import get_settings
from mcp_servers.rag_tools.indexing.persist.bm25_store import BM25Store
from mcp_servers.rag_tools.indexing.persist.doc_store import ParentDocStore
from mcp_servers.rag_tools.retrieval.bm25_retriever import BM25Retriever

if hasattr(sys.stdout, "reconfigure"):
    # Keep the script usable in Windows terminals that still default to gbk.
    sys.stdout.reconfigure(errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 kb_name 和 query 查询本地 BM25 知识库索引")
    parser.add_argument("--kb-name")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show", choices=["child", "parent", "both"], default="child")
    args = parser.parse_args()

    kb_name = (args.kb_name or input("kb_name: ")).strip()
    query = (args.query or input("query: ")).strip()
    top_k = max(1, args.top_k)
    show_mode = args.show

    if not kb_name:
        raise ValueError("kb_name 不能为空。")
    if not query:
        raise ValueError("query 不能为空。")

    settings = get_settings()
    bm25_store = BM25Store(settings.rag.bm25_dir, kb_name)
    bm25_retriever = BM25Retriever(bm25_store)
    doc_store = ParentDocStore(settings.rag.docstore_dir, kb_name)
    results = bm25_retriever.search(query, top_k=top_k)

    _print_line(f"KB: {kb_name}")
    _print_line(f"Query: {query}")
    _print_line(f"Top K: {top_k}")
    _print_line(f"Show Mode: {show_mode}")
    _print_line("")

    if not results:
        _print_line(
            "没有查到 BM25 结果。请先确认该 kb_name 已完成索引构建，"
            "且 query 含有可匹配关键词。"
        )
        return

    for index, document in enumerate(results, start=1):
        metadata = document.metadata
        parent_heading = str(metadata.get("heading_path") or "")
        parent_paragraph = _get_parent_paragraph(doc_store, metadata)

        _print_line(f"[{index}] {metadata.get('file_name', '')}")
        _print_line(f"章节: {parent_heading or '<无>'}")
        _print_line(f"BM25分数: {float(metadata.get('bm25_score') or 0.0):.4f}")
        if show_mode in {"child", "both"}:
            _print_line("命中子段:")
            _print_line(_format_paragraph(document.page_content))
        if show_mode in {"parent", "both"}:
            _print_line("所属父段:")
            _print_line(_format_paragraph(parent_paragraph))
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


def _get_parent_paragraph(doc_store: ParentDocStore, metadata: dict[str, object]) -> str:
    parent_id = str(metadata.get("parent_id") or "")
    if not parent_id:
        return "<无父段上下文>"
    parent = doc_store.get(parent_id)
    if parent is None or not parent.page_content.strip():
        return "<无父段上下文>"
    return parent.page_content


def _print_line(text: str) -> None:
    print(text)


if __name__ == "__main__":
    main()
