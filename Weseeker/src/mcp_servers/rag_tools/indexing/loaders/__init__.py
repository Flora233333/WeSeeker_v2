from __future__ import annotations

from langchain_core.documents import Document

from mcp_servers.rag_tools.indexing.loaders.docx_loader import load_docx
from mcp_servers.rag_tools.indexing.loaders.markdown_loader import load_markdown
from mcp_servers.rag_tools.indexing.loaders.pdf_loader import load_pdf
from mcp_servers.rag_tools.indexing.loaders.txt_loader import load_text
from mcp_servers.rag_tools.indexing.loaders.xlsx_loader import load_xlsx
from mcp_servers.rag_tools.indexing.scanner import FileRecord


def load_file_documents(file_record: FileRecord) -> list[Document]:
    suffix = file_record.path.suffix.lower()
    if suffix == ".md":
        return load_markdown(file_record.path)
    if suffix == ".docx":
        return load_docx(file_record.path)
    if suffix == ".pdf":
        return load_pdf(file_record.path)
    if suffix == ".xlsx":
        return load_xlsx(file_record.path)
    if suffix == ".txt":
        return load_text(file_record.path)
    raise ValueError(f"Step 1 暂不支持该扩展名: {suffix}")
