from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config.settings import KBConfig, get_settings
from mcp_servers.rag_tools.adapters import BatchedEmbedder, create_embedding_model
from mcp_servers.rag_tools.indexing.indexer import build_kb_index
from mcp_servers.rag_tools.indexing.persist.chroma_store import ChromaChildStore
from mcp_servers.rag_tools.indexing.persist.doc_store import ParentDocStore
from mcp_servers.rag_tools.run_rag_test import (
    write_chunk_dump,
    write_chunk_report,
    write_evaluation_report,
)

ROOT_DIR = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="对单个 PDF 做结构化 / 分块 / 向量命中测试")
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--kb-name")
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"只支持测试单个 PDF 文件，当前为: {pdf_path.suffix}")

    kb_name = args.kb_name or f"pdf_test_{_slugify(pdf_path.stem)}"
    temp_root = ROOT_DIR / "storage" / "debug" / "pdf_chunk_test" / kb_name
    output_root = ROOT_DIR / "storage" / "debug"
    output_root.mkdir(parents=True, exist_ok=True)

    # 复制到单独目录，只留下这一份 PDF，方便观察其纯文本结构化效果。
    _prepare_single_pdf_root(pdf_path, temp_root)

    settings = get_settings()
    kb_config = KBConfig(
        name=kb_name,
        root=temp_root.as_posix(),
        description=f"Single PDF chunk test for {pdf_path.name}",
    )

    result = build_kb_index(kb_config, settings.rag)
    embedding_model = create_embedding_model(settings.rag)
    embedder = BatchedEmbedder(
        embedding_model,
        batch_size=settings.rag.embedding_batch_size,
        max_retries=settings.rag.embedding_max_retries,
    )
    chroma_store = ChromaChildStore(settings.rag.chroma_persist_dir, kb_name)
    doc_store = ParentDocStore(settings.rag.docstore_dir, kb_name)

    report_path = output_root / f"chunk_report_{kb_name}.md"
    dump_path = output_root / f"chunk_dump_{kb_name}.jsonl"
    evaluation_path = output_root / f"chunk_evaluation_{kb_name}.md"
    queries = args.query or _default_queries_for_pdf(pdf_path)

    write_chunk_report(report_path, result, chroma_store, doc_store, embedder, queries)
    write_chunk_dump(dump_path, result)
    write_evaluation_report(evaluation_path, result, temp_root)

    print(f"PDF source: {pdf_path}")
    print(f"Temp KB root: {temp_root}")
    print(f"Chunk report: {report_path}")
    print(f"Chunk dump: {dump_path}")
    print(f"Chunk evaluation: {evaluation_path}")


def _prepare_single_pdf_root(pdf_path: Path, temp_root: Path) -> None:
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, temp_root / pdf_path.name)


def _default_queries_for_pdf(pdf_path: Path) -> list[str]:
    stem = pdf_path.stem
    if "RAG学习" in stem:
        return ["基础RAG怎么工作", "GraphRAG 是什么", "LangChain 与 RAG 的关系"]
    if "LLM微调+低精度训练" in stem:
        return ["LoRA 为什么节省显存", "Prompt-Tuning 和 Prefix-Tuning", "QLoRA 4bits训练"]
    if "LM note" in stem:
        return ["注意力机制", "BERT 微调", "Transformer 是什么"]
    if "本科记忆深刻的游戏" in stem:
        return ["致命公司", "守望先锋", "单机游戏"]
    return ["核心内容是什么", "这个 PDF 主要讲什么", "有哪些关键概念"]


def _slugify(value: str) -> str:
    sanitized = []
    for char in value.lower():
        if char.isascii() and (char.isalnum() or char in {"-", "_"}):
            sanitized.append(char)
        elif char in {" ", ".", "/", "\\"}:
            sanitized.append("_")
    result = "".join(sanitized).strip("_")
    return result or "single_pdf"


if __name__ == "__main__":
    main()
