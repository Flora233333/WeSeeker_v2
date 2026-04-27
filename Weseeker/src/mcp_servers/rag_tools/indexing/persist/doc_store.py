from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langchain_core.documents import Document


class ParentDocStore:
    def __init__(self, root_dir: str, kb_name: str) -> None:
        self._db_path = Path(root_dir) / f"{kb_name}.sqlite"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> str:
        return str(self._db_path)

    def rebuild(self, documents: list[Document]) -> None:
        with sqlite3.connect(self._db_path) as connection:
            connection.execute("DROP TABLE IF EXISTS parent_docs")
            connection.execute(
                """
                CREATE TABLE parent_docs (
                    parent_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.executemany(
                "INSERT INTO parent_docs (parent_id, content, metadata_json) VALUES (?, ?, ?)",
                [
                    (
                        str(document.metadata["parent_id"]),
                        document.page_content,
                        json.dumps(document.metadata, ensure_ascii=False),
                    )
                    for document in documents
                ],
            )
            connection.commit()

    def get(self, parent_id: str) -> Document | None:
        with sqlite3.connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT content, metadata_json FROM parent_docs WHERE parent_id = ?",
                (parent_id,),
            ).fetchone()
        if row is None:
            return None
        content, metadata_json = row
        return Document(page_content=content, metadata=json.loads(metadata_json))
