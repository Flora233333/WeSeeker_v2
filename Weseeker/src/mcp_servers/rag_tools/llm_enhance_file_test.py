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
    parser = argparse.ArgumentParser(description="对单个文件运行 RAG LLM enhancement 验证")
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--kb-name", default="llm_enhance_probe")
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    source_path = Path(args.file_path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"单文件验证源文件不存在: {source_path}")

    settings = get_settings()
    if not settings.rag.llm_enhancement.enabled:
        raise ValueError(
            "单文件 LLM enhancement 验证要求先开启 "
            "WESEEKER_RAG__LLM_ENHANCEMENT__ENABLED=true"
        )

    temp_root = ROOT_DIR / "storage" / "debug" / "llm_enhance_file_test" / args.kb_name
    output_root = ROOT_DIR / "storage" / "debug"
    output_root.mkdir(parents=True, exist_ok=True)
    _prepare_single_file_root(source_path, temp_root)

    kb_config = KBConfig(
        name=args.kb_name,
        root=temp_root.as_posix(),
        description=f"Single file LLM enhancement test for {source_path.name}",
    )
    result = build_kb_index(kb_config, settings.rag)

    embedding_model = create_embedding_model(settings.rag)
    embedder = BatchedEmbedder(
        embedding_model,
        batch_size=settings.rag.embedding_batch_size,
        max_retries=settings.rag.embedding_max_retries,
    )
    chroma_store = ChromaChildStore(settings.rag.chroma_persist_dir, args.kb_name)
    doc_store = ParentDocStore(settings.rag.docstore_dir, args.kb_name)

    report_path = output_root / f"chunk_report_{args.kb_name}.md"
    dump_path = output_root / f"chunk_dump_{args.kb_name}.jsonl"
    evaluation_path = output_root / f"chunk_evaluation_{args.kb_name}.md"
    queries = args.query or _default_queries_for_file(source_path)

    write_chunk_report(report_path, result, chroma_store, doc_store, embedder, queries)
    write_chunk_dump(dump_path, result)
    write_evaluation_report(evaluation_path, result, temp_root)

    print(f"File source: {source_path}")
    print(f"Temp KB root: {temp_root}")
    print(f"LLM enhancement stats: {result.llm_enhancement_stats}")
    print(f"Chunk report: {report_path}")
    print(f"Chunk dump: {dump_path}")
    print(f"Chunk evaluation: {evaluation_path}")


def _prepare_single_file_root(source_path: Path, temp_root: Path) -> None:
    temp_root.mkdir(parents=True, exist_ok=True)
    target_path = temp_root / source_path.name
    existing_files = [path for path in temp_root.iterdir() if path.is_file()]
    unexpected = [path for path in existing_files if path.name != source_path.name]
    if unexpected:
        raise ValueError(
            "单文件验证目录包含其他文件，请换一个 kb_name，"
            f"unexpected={[path.name for path in unexpected]}"
        )
    # 不删除既有目录，只覆盖同名验证文件，避免误删用户调试产物。
    shutil.copy2(source_path, target_path)


def _default_queries_for_file(source_path: Path) -> list[str]:
    stem = source_path.stem.lower()
    if "transformer" in stem:
        return ["Transformer 注意力机制", "Encoder Decoder 架构", "位置编码"]
    if "lora" in stem or "低精度" in stem:
        return ["LoRA 为什么能减少显存", "QLoRA 4bits 训练", "PEFT 参数高效微调"]
    return ["这个文件的核心内容是什么", "有哪些关键概念", "有什么实践注意事项"]


if __name__ == "__main__":
    main()
