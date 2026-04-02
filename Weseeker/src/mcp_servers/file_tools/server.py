from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from config.settings import get_settings
from mcp_servers.file_tools.search import execute_search


mcp = FastMCP(
    name="weseeker-file-tools",
    instructions="WeSeeker MVP 文件搜索工具集",
    host="127.0.0.1",
    port=get_settings().mcp.file_tools_port,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def search_files(
    keyword: str,
    path: str | None = None,
    max_results: int = 20,
) -> str:
    """在本地电脑中按文件名搜索文件。"""
    return await execute_search(keyword=keyword, path=path, max_results=max_results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
