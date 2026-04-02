from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient

from config.settings import get_settings


def create_mcp_client() -> MultiServerMCPClient:
    settings = get_settings()
    return MultiServerMCPClient(
        {
            "file_tools": {
                "transport": "http",
                "url": f"http://127.0.0.1:{settings.mcp.file_tools_port}/mcp",
            }
        }
    )
