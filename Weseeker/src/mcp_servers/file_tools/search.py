from __future__ import annotations

import json

from mcp_servers.file_tools.utils.everything_client import EverythingClient
from mcp_servers.file_tools.utils.path_resolver import resolve_search_path


async def execute_search(keyword: str, path: str | None = None, max_results: int = 20) -> str:
    cleaned_keyword = " ".join(keyword.split())
    if not cleaned_keyword:
        return json.dumps(
            {
                "ok": False,
                "error": "搜索关键词不能为空。",
            },
            ensure_ascii=False,
        )

    resolved_path = resolve_search_path(path)
    client = EverythingClient()
    try:
        results = await client.search(
            cleaned_keyword,
            search_path=resolved_path,
            max_results=max_results,
        )
    finally:
        await client.close()

    return json.dumps(
        {
            "ok": True,
            "keyword": cleaned_keyword,
            "path": resolved_path,
            "count": len(results),
            "results": [
                {
                    "index": index,
                    "name": item.name,
                    "path": item.path,
                    "full_path": item.full_path,
                    "size": item.size,
                    "modified": item.modified,
                    "is_dir": item.is_dir,
                }
                for index, item in enumerate(results, start=1)
            ],
        },
        ensure_ascii=False,
    )
