from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient

from config.settings import get_settings


THREAD_ID_HEADER = "X-WeSeeker-Thread-Id"


def create_mcp_client(*, thread_id: str | None = None) -> MultiServerMCPClient:
    settings = get_settings()
    headers = {THREAD_ID_HEADER: thread_id} if thread_id else None
    return MultiServerMCPClient(
        {
            "file_tools": {
                "transport": "http",
                "url": f"http://127.0.0.1:{settings.mcp.file_tools_port}/mcp",
                "headers": headers,
            }
        }
    )
