from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx

from mcp_servers.file_tools.utils.candidate_registry import (
    get_candidate_by_index,
    normalize_candidate_source,
    store_candidates,
)
from mcp_servers.file_tools.utils.everything_client import EverythingClient
from mcp_servers.file_tools.utils.file_extractors import format_size_display
from mcp_servers.file_tools.utils.tool_response import build_error_response


def _serialize_size_fields(item_size: int, *, is_dir: bool) -> tuple[int | None, str]:
    if is_dir:
        return None, "folder"
    return item_size, format_size_display(item_size)


async def execute_list_folder_contents(
    file_index: int,
    max_results: int = 30,
    *,
    client_id: str | None = None,
    candidate_source: str = "search_files",
) -> str:
    try:
        normalized_source = normalize_candidate_source(candidate_source)
    except ValueError as exc:
        return build_error_response(
            "invalid_candidate_source",
            str(exc),
            user_hint="候选来源仅支持 search_files 或 list_folder_contents。",
            operator_hint="请传入合法 candidate_source。",
        )

    candidate = get_candidate_by_index(client_id, file_index, source=normalized_source)
    if candidate is None:
        return build_error_response(
            "candidate_not_found",
            "未找到对应的候选文件序号，请先重新搜索。",
            user_hint="当前候选序号无效，请先确认当前 candidates。",
            operator_hint="指定 source 下未找到对应 file_index。",
        )

    folder_path = Path(candidate.full_path)
    if not folder_path.exists():
        return build_error_response(
            "directory_not_found",
            "目标文件夹不存在或已被移动。",
            user_hint="目标文件夹当前不可访问，请重新搜索。",
            operator_hint="候选中的目录路径已失效。",
        )
    if not folder_path.is_dir():
        return build_error_response(
            "not_a_directory",
            "当前候选不是文件夹，无法列出目录内容。",
            user_hint="请选择文件夹候选后再查看目录内容。",
            operator_hint="list_folder_contents 只支持目录候选。",
        )

    client = EverythingClient()
    try:
        items = await client.list_children(str(folder_path), max_results=max_results)
    except httpx.ConnectError:
        return build_error_response(
            "folder_list_failed",
            "目录列出失败：Everything HTTP 服务不可用。",
            retryable=True,
            user_hint="目录列出服务当前不可用，请稍后重试。",
            operator_hint="请检查 Everything HTTP 服务监听状态。",
        )
    except httpx.HTTPError as exc:
        return build_error_response(
            "folder_list_failed",
            f"目录列出失败：{exc}",
            retryable=True,
            user_hint="目录列出失败，请稍后重试。",
            operator_hint="请检查 Everything HTTP 服务状态与目录查询参数。",
        )
    finally:
        await client.close()

    updated_at = datetime.now().isoformat(timespec="seconds")
    store_candidates(
        client_id,
        items,
        source="list_folder_contents",
        query=None,
        path=str(folder_path),
        updated_at=updated_at,
    )

    return json.dumps(
        {
            "ok": True,
            "source": "file_index",
            "candidate_source": normalized_source,
            "folder_name": folder_path.name,
            "folder_path": str(folder_path),
            "updated_at": updated_at,
            "count": len(items),
            "results": [
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
                for index, item in enumerate(items, start=1)
                for size, size_display in [_serialize_size_fields(item.size, is_dir=item.is_dir)]
            ],
            "notice": (
                "这些结果已写入 list_folder_contents candidates；"
                "后续若继续用 file_index，请指定正确的 candidate_source。"
            ),
        },
        ensure_ascii=False,
    )
