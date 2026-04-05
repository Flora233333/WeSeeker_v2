from __future__ import annotations

import json

import httpx

from config.settings import get_settings
from mcp_servers.file_tools.utils.candidate_registry import clear_candidates
from mcp_servers.file_tools.utils.candidate_registry import store_candidates
from mcp_servers.file_tools.utils.everything_client import EverythingClient
from mcp_servers.file_tools.utils.file_extractors import format_size_display
from mcp_servers.file_tools.utils.path_resolver import resolve_search_path
from mcp_servers.file_tools.utils.tool_response import build_error_response


async def execute_search(
    keyword: str,
    path: str | None = None,
    max_results: int = 20,
    *,
    client_id: str | None = None,
) -> str:
    cleaned_keyword = " ".join(keyword.split())
    if not cleaned_keyword:
        return build_error_response(
            "invalid_argument",
            "搜索关键词不能为空。",
            user_hint="请提供要搜索的文件名关键词。",
            operator_hint="调用 search_files 时传入非空 keyword。",
        )

    resolved_path = resolve_search_path(path) # 解析路径，如：去桌面找这种，模型会传一个desktop进来
    client = EverythingClient()
    try:
        results = await client.search(
            cleaned_keyword,
            search_path=resolved_path,
            max_results=max_results,
        )
    except httpx.ConnectError:
        clear_candidates(client_id)
        settings = get_settings()
        return build_error_response(
            "everything_unavailable",
            "Everything HTTP 服务不可用，请先确认 Everything 已启动 HTTP 服务，"
            f"当前配置地址为 http://{settings.everything.host}:{settings.everything.port}。",
            retryable=True,
            user_hint="文件搜索服务当前不可用，请稍后重试。",
            operator_hint=(
                "请确认 Everything 已启动 HTTP 服务，并监听 "
                f"http://{settings.everything.host}:{settings.everything.port}。"
            ),
        )
    except httpx.HTTPError as exc:
        clear_candidates(client_id)
        return build_error_response(
            "everything_request_failed",
            f"Everything HTTP 请求失败：{exc}",
            retryable=True,
            user_hint="搜索服务请求失败，请稍后重试。",
            operator_hint="请检查 Everything HTTP 服务状态与请求参数。",
        )
    finally:
        await client.close()

    store_candidates(client_id, results) # 直接存内存里了，按照client_id索引

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
                    "size_display": format_size_display(item.size),
                    "modified": item.modified,
                    "is_dir": item.is_dir,
                }
                for index, item in enumerate(results, start=1)
            ],
        },
        ensure_ascii=False,
    )
