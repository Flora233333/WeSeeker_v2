from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document

from mcp_servers.rag_tools.indexing.scanner import FileRecord


def build_cache_key(
    file_record: FileRecord,
    parent_documents: list[Document],
    child_documents: list[Document],
    *,
    model: str,
    prompt_version: str,
) -> str:
    payload = {
        "file_path": file_record.path_str,
        "mtime": file_record.mtime,
        "size": file_record.size,
        "model": model,
        "prompt_version": prompt_version,
        "parents": [
            {
                "heading_path": document.metadata.get("heading_path", ""),
                "content": document.page_content,
            }
            for document in parent_documents
        ],
        "children": [
            {
                "parent_heading_path": document.metadata.get("heading_path", ""),
                "content": document.page_content,
            }
            for document in child_documents
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_cache(cache_dir: str, kb_name: str) -> dict[str, dict[str, object]]:
    path = _cache_path(cache_dir, kb_name)
    if not path.exists():
        return {}

    records: dict[str, dict[str, object]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM enhancement cache JSON 无法解析: {path}:{line_number}") from exc
        cache_key = record.get("cache_key") if isinstance(record, dict) else None
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError(f"LLM enhancement cache 缺少 cache_key: {path}:{line_number}")
        records[cache_key] = record
    return records


def append_cache_record(cache_dir: str, kb_name: str, record: dict[str, object]) -> None:
    path = _cache_path(cache_dir, kb_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _cache_path(cache_dir: str, kb_name: str) -> Path:
    return Path(cache_dir) / f"{kb_name}.jsonl"
