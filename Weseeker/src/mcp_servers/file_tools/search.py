from __future__ import annotations

import json
from datetime import datetime

import httpx

from mcp_servers.file_tools.utils.candidate_registry import (
    clear_candidates,
    get_all_candidate_snapshots,
    normalize_candidate_source,
    store_candidates,
)
from mcp_servers.file_tools.utils.everything_client import EverythingClient, SearchItem
from mcp_servers.file_tools.utils.file_extractors import format_size_display
from mcp_servers.file_tools.utils.path_resolver import resolve_search_path
from mcp_servers.file_tools.utils.tool_response import build_error_response


def _serialize_size_fields(item: SearchItem) -> tuple[int | None, str]:
    # [改动 1]
    # 文件夹不再显示 0 B，避免模型把目录误读成空文件。
    # size 对目录返回 null，size_display 返回 "folder"。
    if item.is_dir:
        return None, "folder"
    return item.size, format_size_display(item.size)


def _serialize_results(
    results: tuple[SearchItem, ...] | list[SearchItem],
) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []

    for index, item in enumerate(results, start=1):
        size, size_display = _serialize_size_fields(item)
        serialized.append(
            {
                "index": index,
                "name": item.name,
                "path": item.path,
                "full_path": item.full_path,
                "size": size,
                "size_display": size_display,
                "modified": item.modified,
                "is_dir": item.is_dir,
            }
        )

    return serialized


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

    resolved_path = resolve_search_path(path)
    client = EverythingClient()
    try:
        results = await client.search(
            cleaned_keyword,
            search_path=resolved_path,
            max_results=max_results,
        )
    except httpx.ConnectError:
        clear_candidates(client_id, source="search_files")
        return build_error_response(
            "everything_unavailable",
            "Everything HTTP 服务不可用，请先确认 Everything 已启动 HTTP 服务。",
            retryable=True,
            user_hint="文件搜索服务当前不可用，请稍后重试。",
            operator_hint="请检查 Everything HTTP 服务监听状态。",
        )
    except httpx.HTTPError as exc:
        clear_candidates(client_id, source="search_files")
        return build_error_response(
            "everything_request_failed",
            f"Everything HTTP 请求失败：{exc}",
            retryable=True,
            user_hint="搜索服务请求失败，请稍后重试。",
            operator_hint="请检查 Everything HTTP 服务状态与请求参数。",
        )
    finally:
        await client.close()

    updated_at = datetime.now().isoformat(timespec="seconds")
    store_candidates(
        client_id,
        results,
        source="search_files",
        query=cleaned_keyword,
        path=resolved_path,
        updated_at=updated_at,
    )

    return json.dumps(
        {
            "ok": True,
            "keyword": cleaned_keyword,
            "path": resolved_path,
            "count": len(results),
            "results": _serialize_results(results),
        },
        ensure_ascii=False,
    )


async def execute_get_current_candidates(client_id: str | None = None) -> str:
    snapshots = get_all_candidate_snapshots(client_id)
    sources: dict[str, dict[str, object]] = {}

    for source_name in ("search_files", "list_folder_contents"):
        source = normalize_candidate_source(source_name)
        snapshot = snapshots.get(source)
        if snapshot is None:
            sources[source_name] = {
                "has_candidates": False,
                "message": "当前该来源下没有可用 candidates。",
                "count": 0,
                "results": [],
            }
            continue

        sources[source_name] = {
            "has_candidates": True,
            "updated_at": snapshot.updated_at,
            "query": snapshot.query,
            "path": snapshot.path,
            "count": len(snapshot.results),
            "results": _serialize_results(snapshot.results),
        }

    return json.dumps(
        {
            "ok": True,
            "sources": sources,
        },
        ensure_ascii=False,
    )
