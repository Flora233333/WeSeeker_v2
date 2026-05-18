"""Manual hybrid retrieval checker for human review.

Usage:
    python src/mcp_servers/rag_tools/query_kb_hybrid_test.py --kb-name <kb_name> --query <query>

This script does not rebuild the KB. It reads existing Chroma, BM25 and ParentDocStore
artifacts, then prints parent-level evidence after child-level RRF fusion.
"""

from __future__ import annotations

import argparse
import sys

from mcp_servers.rag_tools.search_kb import search_kb

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

    payload = search_kb(query, kb_name, top_k=top_k, candidate_k=candidate_k)

    _print_line(f"KB: {kb_name}")
    _print_line(f"Query: {query}")
    _print_line(f"Top K: {top_k}")
    _print_line(f"Candidate K: {candidate_k}")
    _print_line(f"Show Mode: {show_mode}")
    _print_line("")

    if not payload["ok"]:
        _print_line(f"查询失败: {payload['message']}")
        operator_hint = str(payload.get("operator_hint") or "")
        if operator_hint:
            _print_line(f"诊断: {operator_hint}")
        return

    results = payload["results"]
    if not isinstance(results, list) or not results:
        _print_line(
            "没有查到 Hybrid 结果。请确认索引已构建，或当前 query 只有低相关向量结果。"
        )
        return

    for result in results:
        if not isinstance(result, dict):
            continue
        _print_line(f"[{result.get('rank')}] {result.get('file_name') or ''}")
        _print_line(f"章节: {result.get('heading_path') or '<无>'}")
        sources = ", ".join(str(source) for source in result.get("retrieval_sources", []))
        _print_line(f"来源: {sources}")
        _print_line(f"RRF分数: {float(result.get('rank_score') or 0.0):.4f}")
        matched_children = result.get("matched_children") or []
        matched_count = len(matched_children) if isinstance(matched_children, list) else 0
        _print_line(f"命中Child数: {matched_count}")
        _print_line(f"最佳Child: {_best_child_id(matched_children)}")
        scores = result.get("scores") if isinstance(result.get("scores"), dict) else {}
        _print_line(
            "Rank: vector={vector_rank} bm25={bm25_rank}".format(
                vector_rank=scores.get("vector_rank") or "-",
                bm25_rank=scores.get("bm25_rank") or "-",
            )
        )
        if show_mode in {"child", "both"}:
            _print_line("命中子段摘要:")
            if isinstance(matched_children, list):
                for child in matched_children:
                    if isinstance(child, dict):
                        _print_line(f"- {_format_paragraph(str(child.get('snippet') or ''))}")
        if show_mode in {"parent", "both"}:
            _print_line("所属父段摘要:")
            _print_line(_format_paragraph(str(result.get("snippet") or "")))
        _print_line("-" * 80)


def _best_child_id(children: object) -> str:
    if not isinstance(children, list):
        return ""
    for child in children:
        if isinstance(child, dict) and child.get("is_best"):
            return str(child.get("child_id") or "")
    return ""


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
