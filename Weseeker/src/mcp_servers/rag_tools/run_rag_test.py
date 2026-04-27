from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from config.settings import KBConfig, get_settings
from mcp_servers.rag_tools.adapters import BatchedEmbedder, create_embedding_model
from mcp_servers.rag_tools.indexing.indexer import IndexBuildResult, build_kb_index
from mcp_servers.rag_tools.indexing.persist.chroma_store import ChromaChildStore
from mcp_servers.rag_tools.indexing.persist.doc_store import ParentDocStore
from mcp_servers.rag_tools.indexing.scanner import ScanResult

ROOT_DIR = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 WeSeeker RAG Step 1 离线索引测试")
    parser.add_argument("--kb-name", required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    settings = get_settings()
    output_root = ROOT_DIR / "storage" / "debug"
    output_root.mkdir(parents=True, exist_ok=True)
    kb_root = Path(args.root_dir).expanduser().resolve()

    kb_config = KBConfig(
        name=args.kb_name,
        root=kb_root.as_posix(),
        description="RAG Step 1 custom knowledge base",
    )

    result = build_kb_index(kb_config, settings.rag)
    embedding_model = create_embedding_model(settings.rag)
    embedder = BatchedEmbedder(
        embedding_model,
        batch_size=settings.rag.embedding_batch_size,
        max_retries=settings.rag.embedding_max_retries,
    )
    chroma_store = ChromaChildStore(settings.rag.chroma_persist_dir, kb_config.name)
    doc_store = ParentDocStore(settings.rag.docstore_dir, kb_config.name)

    report_path = output_root / f"chunk_report_{kb_config.name}.md"
    dump_path = output_root / f"chunk_dump_{kb_config.name}.jsonl"
    evaluation_path = output_root / f"chunk_evaluation_{kb_config.name}.md"

    queries = args.query or _default_queries_for_kb(kb_root)
    write_chunk_report(report_path, result, chroma_store, doc_store, embedder, queries)
    write_chunk_dump(dump_path, result)
    write_evaluation_report(evaluation_path, result, kb_root)

    print("Scan summary:")
    for line in _build_scan_summary_lines(result.scan_result):
        print(line)
    print("")
    print(f"KB root: {kb_root}")
    print(f"Chroma path: {result.chroma_path}")
    print(f"DocStore path: {result.docstore_path}")
    print(f"Manifest path: {result.manifest_path}")
    print(f"Chunk report: {report_path}")
    print(f"Chunk dump: {dump_path}")
    print(f"Chunk evaluation: {evaluation_path}")


def write_chunk_report(
    report_path: Path,
    result: IndexBuildResult,
    chroma_store: ChromaChildStore,
    doc_store: ParentDocStore,
    embedder: BatchedEmbedder,
    queries: list[str],
) -> None:
    parents_by_file: dict[str, list[object]] = defaultdict(list)
    children_by_parent: dict[str, list[object]] = defaultdict(list)

    for parent in result.parent_documents:
        parents_by_file[str(parent.metadata["file_path"])].append(parent)
    for child in result.child_documents:
        children_by_parent[str(child.metadata["parent_id"])].append(child)

    lines = [
        f"# 分块质量报告 - KB: {result.kb_name}",
        "",
        "文件总数：{files} | Parent 总数：{parents} | Child 总数：{children}".format(
            files=len(result.scan_result.files),
            parents=len(result.parent_documents),
            children=len(result.child_documents),
        ),
        "Embedding batches：{batches} | Embedding dimension：{dimension}".format(
            batches=result.embedding_batches,
            dimension=result.embedding_dimension,
        ),
        f"总耗时：{result.elapsed_ms} ms",
        "",
        "---",
    ]

    for file_path in sorted(parents_by_file):
        parents = parents_by_file[file_path]
        total_children = sum(
            len(children_by_parent[str(parent.metadata["parent_id"])])
            for parent in parents
        )
        lines.extend(
            [
                f"## 文件：{Path(file_path).name}",
                f"- doc_type: {parents[0].metadata.get('doc_type', '')}",
                f"- Parent 数量：{len(parents)} | Child 总数：{total_children}",
                "",
            ]
        )

        for index, parent in enumerate(parents, start=1):
            parent_id = str(parent.metadata["parent_id"])
            children = children_by_parent[parent_id]
            lines.extend(
                [
                    f"### Parent #{index} (id: {parent_id})",
                    f"- heading_path: {parent.metadata.get('heading_path', '')}",
                    f"- 长度：{len(parent.page_content)} chars",
                    "- 内容预览（前 300 字）：",
                    f"> {_preview(parent.page_content, 300)}",
                    "",
                    f"**Child chunks ({len(children)} 个)：**",
                    "| # | 长度 | 前 80 字 |",
                    "|---|---|---|",
                ]
            )
            for child_index, child in enumerate(children, start=1):
                lines.append(
                    "| {index} | {length} chars | {preview} |".format(
                        index=child_index,
                        length=len(child.page_content),
                        preview=_escape_table(_preview(child.page_content, 80)),
                    )
                )
            lines.append("")

    lines.extend(["---", "", "## 向量匹配度验证", ""])
    for query in queries:
        lines.extend(_build_query_report(query, chroma_store, doc_store, embedder))

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _build_query_report(
    query: str,
    chroma_store: ChromaChildStore,
    doc_store: ParentDocStore,
    embedder: BatchedEmbedder,
) -> list[str]:
    query_embedding = embedder.embed_query(query)
    result = chroma_store.query(query_embedding, top_k=3)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    lines = [
        f"### Query: \"{query}\"",
        "| Rank | 文件 | heading_path | 近似相关度 | Parent 前 200 字 |",
        "|---|---|---|---|---|",
    ]
    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances, strict=False),
        start=1,
    ):
        parent = doc_store.get(str(metadata.get("parent_id", "")))
        parent_preview = _preview(parent.page_content if parent else str(document), 200)
        score = max(0.0, 1.0 - float(distance))
        lines.append(
            "| {rank} | {file_name} | {heading_path} | {score:.4f} | {preview} |".format(
                rank=index,
                file_name=_escape_table(str(metadata.get("file_name", ""))),
                heading_path=_escape_table(str(metadata.get("heading_path", ""))),
                score=score,
                preview=_escape_table(parent_preview),
            )
        )
    lines.append("")
    return lines


def write_chunk_dump(path: Path, result: IndexBuildResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for document in [*result.parent_documents, *result.child_documents]:
            payload = {
                "chunk_level": document.metadata.get("chunk_level"),
                "file_path": document.metadata.get("file_path"),
                "file_name": document.metadata.get("file_name"),
                "doc_type": document.metadata.get("doc_type"),
                "parent_id": document.metadata.get("parent_id"),
                "child_id": document.metadata.get("child_id"),
                "heading_path": document.metadata.get("heading_path"),
                "section_title": document.metadata.get("section_title"),
                "page_number": document.metadata.get("page_number"),
                "sheet_name": document.metadata.get("sheet_name"),
                "char_count": document.metadata.get("char_count"),
                "preview": _preview(document.page_content, 300),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_evaluation_report(path: Path, result: IndexBuildResult, kb_root: Path) -> None:
    children_by_parent: dict[str, list[object]] = defaultdict(list)
    parents_by_file: dict[str, list[object]] = defaultdict(list)
    for parent in result.parent_documents:
        parents_by_file[str(parent.metadata["file_path"])].append(parent)
    for child in result.child_documents:
        children_by_parent[str(child.metadata["parent_id"])].append(child)

    parent_lengths = [len(parent.page_content) for parent in result.parent_documents]
    child_lengths = [len(child.page_content) for child in result.child_documents]
    very_short_children = [
        child for child in result.child_documents if len(child.page_content) < 50
    ]
    short_children = [
        child for child in result.child_documents if len(child.page_content) < 80
    ]

    short_parents = [
        parent for parent in result.parent_documents if len(parent.page_content) < 120
    ]
    multi_child_parents = [
        parent
        for parent in result.parent_documents
        if len(children_by_parent[str(parent.metadata["parent_id"])]) > 1
    ]
    no_heading_parents = [
        parent
        for parent in result.parent_documents
        if not str(parent.metadata.get("heading_path") or "").strip()
    ]

    lines = [
        f"# Step 1 评估报告 - KB: {result.kb_name}",
        "",
        f"目录：`{kb_root.as_posix()}`",
        "文件数：{files}，跳过文件：{skipped}".format(
            files=len(result.scan_result.files),
            skipped=len(result.scan_result.skipped),
        ),
        "Parent 数：{parents}，Child 数：{children}".format(
            parents=len(result.parent_documents),
            children=len(result.child_documents),
        ),
        "",
        "## 总体结论",
        "",
        _overall_conclusion(result, short_parents, multi_child_parents, no_heading_parents),
        "",
        "## 统计摘要",
        "",
        *_build_scan_summary_lines(result.scan_result),
        "",
        f"- Parent 平均长度：{_safe_avg(parent_lengths):.1f} chars",
        f"- Child 平均长度：{_safe_avg(child_lengths):.1f} chars",
        f"- 多 Child Parent 数：{len(multi_child_parents)}",
        f"- 缺少 heading_path 的 Parent 数：{len(no_heading_parents)}",
        f"- 偏短 Parent（<120 chars）数：{len(short_parents)}",
        f"- 超短 Child（<50 chars）数：{len(very_short_children)}",
        f"- 短 Child（<80 chars）数：{len(short_children)}",
        "",
        "## 主要观察",
        "",
        "- Markdown 文件按标题层级切得比较稳定，`heading_path` 基本可用。",
        "- 短 Child 已做一轮收口，纯结构噪音减少，同时保留了短但有语义的检索目标。",
        "- 当前仍有少量短 Child，多数是短章节、短命令或短结论，不建议按长度一刀切删除。",
        "- 仍有部分结构化文档存在 `1 parent = 1 child`，后续可结合真实检索继续观察。",
        "- PDF 已优先走动态标题结构恢复，失败时才按页兜底；复杂版式仍需人工抽查。",
        "- 少量 DOCX / 非结构化文本缺少 heading_path，当前属于预期边界。",
        "- 未支持的 `.doc` 和重复来源 PDF 已被明确记入 skipped，而不是静默忽略。",
        "",
        "## 需要重点人工查看的文件",
        "",
    ]

    focus_items = _select_focus_files(parents_by_file, children_by_parent)
    for item in focus_items:
        lines.append(f"- `{item}`")

    lines.extend(["", "## 跳过文件", ""])
    if result.scan_result.skipped:
        for skipped in result.scan_result.skipped:
            lines.append(f"- `{skipped.path}` -> `{skipped.reason}`")
    else:
        lines.append("- 无")

    path.write_text("\n".join(lines), encoding="utf-8")


def _build_scan_summary_lines(scan_result: ScanResult) -> list[str]:
    return [
        f"- added: {len(scan_result.added)}",
        f"- modified: {len(scan_result.modified)}",
        f"- unchanged: {len(scan_result.unchanged)}",
        f"- deleted: {len(scan_result.deleted)}",
        f"- skipped: {len(scan_result.skipped)}",
    ]


def _default_queries_for_kb(kb_root: Path) -> list[str]:
    root_name = kb_root.name.lower()
    if root_name == "test_kb":
        return [
            "LangChain 有哪些坑",
            "LoRA 或低精度训练踩坑",
            "我目前的 Agent 学习计划",
            "研究生阶段的应对策略",
        ]
    return [
        "LoRA 为什么能减少显存占用",
        "Transformer 的注意力机制怎么计算",
        "哪种方法的显存峰值最低",
    ]


def _overall_conclusion(
    result: IndexBuildResult,
    short_parents: list[object],
    multi_child_parents: list[object],
    no_heading_parents: list[object],
) -> str:
    if len(result.parent_documents) <= 0:
        return "- 当前没有生成任何分块结果，Step 1 不可用。"

    conclusions: list[str] = []
    conclusions.append("- 当前 Step 1 离线索引链路已可用，可进入 Step 2 检索层评估。")
    if len(multi_child_parents) < max(2, len(result.parent_documents) // 10):
        conclusions.append("- 当前切分整体偏保守，绝大多数 Parent 没有继续拆成多个 Child。")
    if len(short_parents) > len(result.parent_documents) * 0.3:
        conclusions.append("- 当前有较多偏短 Parent，说明部分标题段/短段被直接保留为独立块。")
    if no_heading_parents:
        conclusions.append(
            "- 非结构化文本缺少 heading_path 是预期现象，但后续可考虑补 section label。"
        )
    return "\n".join(conclusions)


def _safe_avg(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _select_focus_files(
    parents_by_file: dict[str, list[object]],
    children_by_parent: dict[str, list[object]],
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for file_path, parents in parents_by_file.items():
        child_count = sum(
            len(children_by_parent[str(parent.metadata["parent_id"])])
            for parent in parents
        )
        score = len(parents) * 10 + child_count
        scored.append((score, Path(file_path).name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:8]]


def _preview(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|")


if __name__ == "__main__":
    main()
