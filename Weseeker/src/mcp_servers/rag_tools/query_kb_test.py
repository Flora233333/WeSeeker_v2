"""Manual KB retrieval checker for human review.

Usage:
    python src/mcp_servers/rag_tools/query_kb_test.py --kb-name <kb_name> --query <query>

Common commands:
    python src/mcp_servers/rag_tools/query_kb_test.py \
        --kb-name test_kb_notes \
        --query "LoRA 或低精度训练踩坑"
    python src/mcp_servers/rag_tools/query_kb_test.py \
        --kb-name test_kb_notes \
        --query "LangChain 有哪些坑" \
        --top-k 5 \
        --show child
    python src/mcp_servers/rag_tools/query_kb_test.py \
        --kb-name test_kb_notes \
        --query "LangChain 有哪些坑" \
        --top-k 5 \
        --show parent
    python src/mcp_servers/rag_tools/query_kb_test.py \
        --kb-name test_kb_notes --query "我的就业计划是什么？" --top-k 5 --show both

Show modes:
    child  -> print matched child chunk only
    parent -> print parent chunk only
    both   -> print both child chunk and parent chunk
"""

from __future__ import annotations

import argparse
import sys

from config.settings import get_settings
from mcp_servers.rag_tools.adapters import BatchedEmbedder, create_embedding_model
from mcp_servers.rag_tools.indexing.persist.chroma_store import ChromaChildStore
from mcp_servers.rag_tools.indexing.persist.doc_store import ParentDocStore

if hasattr(sys.stdout, "reconfigure"):
    # Keep the script usable in Windows terminals that still default to gbk.
    sys.stdout.reconfigure(errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 kb_name 和 query 查询已构建的本地知识库")
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
    embedding_model = create_embedding_model(settings.rag)
    embedder = BatchedEmbedder(
        embedding_model,
        batch_size=settings.rag.embedding_batch_size,
        max_retries=settings.rag.embedding_max_retries,
    )
    chroma_store = ChromaChildStore(settings.rag.chroma_persist_dir, kb_name)
    doc_store = ParentDocStore(settings.rag.docstore_dir, kb_name)

    query_embedding = embedder.embed_query(query)
    raw_result = chroma_store.query(query_embedding, top_k=top_k)
    rows = _flatten_query_result(raw_result)

    _print_line(f"KB: {kb_name}")
    _print_line(f"Query: {query}")
    _print_line(f"Top K: {top_k}")
    _print_line(f"Show Mode: {show_mode}")
    _print_line("")

    if not rows:
        _print_line("没有查到结果。请先确认该 kb_name 已完成索引构建。")
        return

    for index, row in enumerate(rows, start=1):
        metadata = row["metadata"]
        parent_heading = str(metadata.get("heading_path") or "")
        child_paragraph = row["document"]
        parent_paragraph = _get_parent_paragraph(doc_store, metadata)

        _print_line(f"[{index}] {metadata.get('file_name', '')}")
        _print_line(f"章节: {parent_heading or '<无>'}")
        _print_line(f"相似度: {max(0.0, 1.0 - float(row['distance'])):.4f}")
        if show_mode in {"child", "both"}:
            _print_line("命中子段:")
            _print_line(_format_paragraph(child_paragraph))
        if show_mode in {"parent", "both"}:
            _print_line("所属父段:")
            _print_line(_format_paragraph(parent_paragraph))
        _print_line("-" * 80)


def _flatten_query_result(raw_result: dict[str, list[object]]) -> list[dict[str, object]]:
    """Normalize Chroma's nested query response into a flat, printable result list."""

    documents = raw_result.get("documents", [[]])
    metadatas = raw_result.get("metadatas", [[]])
    distances = raw_result.get("distances", [[]])

    if not documents or not metadatas or not distances:
        return []

    rows: list[dict[str, object]] = []
    for document, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
        strict=False,
    ):
        rows.append(
            {
                "document": str(document),
                "metadata": metadata if isinstance(metadata, dict) else {},
                "distance": distance,
            }
        )
    return rows


def _format_paragraph(text: str) -> str:
    """Keep paragraph output human-readable while avoiding giant blank gaps."""

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
    """Resolve the full parent chunk for the current hit when the user wants more context."""

    parent_id = str(metadata.get("parent_id") or "")
    if not parent_id:
        return "<无父段上下文>"
    parent = doc_store.get(parent_id)
    if parent is None or not parent.page_content.strip():
        return "<无父段上下文>"
    return parent.page_content


def _print_line(text: str) -> None:
    """Print with replacement fallback so retrieval demos do not crash on gbk consoles."""

    print(text)


if __name__ == "__main__":
    main()
