from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


def load_markdown(path: Path) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Markdown 文件不是合法 UTF-8: {path.as_posix()}") from exc
    return [
        Document(
            page_content=text,
            metadata={
                "doc_type": "md",
                "file_path": path.as_posix(),
                "file_name": path.name,
                "file_ext": path.suffix.lower(),
            },
        )
    ]
