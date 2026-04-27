from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

_CANDIDATE_ENCODINGS = ("utf-8", "utf-8-sig", "gbk")


def load_text(path: Path) -> list[Document]:
    content = _read_text(path)
    return [
        Document(
            page_content=content,
            metadata={
                "doc_type": "txt",
                "file_path": path.as_posix(),
                "file_name": path.name,
                "file_ext": path.suffix.lower(),
            },
        )
    ]


def _read_text(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    tried = ", ".join(_CANDIDATE_ENCODINGS)
    raise ValueError(
        f"TXT 文件无法用候选编码解码: {path.as_posix()}, tried={tried}"
    ) from last_error
