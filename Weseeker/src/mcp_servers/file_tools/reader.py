from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp_servers.file_tools.utils.candidate_registry import (
    get_candidate_by_index,
    normalize_candidate_source,
)
from mcp_servers.file_tools.utils.file_extractors import (
    StructuredPreviewError,
    UnsupportedFileTypeError,
    extract_preview,
    normalize_depth,
)
from mcp_servers.file_tools.utils.file_filter import should_include_path
from mcp_servers.file_tools.utils.tool_response import build_error_response


async def execute_read_content(
    file_index: int | None = None,
    file_path: str | None = None,
    depth: str = "L1",
    *,
    client_id: str | None = None,
    candidate_source: str = "search_files",
) -> str:
    try:
        normalized_depth = normalize_depth(depth)
        source, resolved_path = _resolve_target_path(
            file_index=file_index,
            file_path=file_path,
            client_id=client_id,
            candidate_source=candidate_source,
        )
    except ValueError as exc:
        return _map_argument_error(str(exc))

    if not resolved_path.exists():
        return build_error_response(
            "file_not_found",
            "目标文件不存在或已被移动。",
            user_hint="目标文件当前不可访问，请重新搜索或确认路径。",
            operator_hint="请确认候选缓存中的文件仍存在于本地磁盘。",
        )
    if resolved_path.is_dir():
        return build_error_response(
            "directory_not_supported",
            "当前目标是文件夹，不支持内容预览。",
            user_hint="当前目标是文件夹，请改为使用 list_folder_contents 查看目录内容。",
            operator_hint="目录预览应改走 list_folder_contents，而不是 read_file_content。",
        )
    if not should_include_path(str(resolved_path)):
        return build_error_response(
            "preview_not_allowed",
            "当前目标属于受限文件或临时文件，无法预览。",
            user_hint="当前文件不允许预览，请选择其他候选文件。",
            operator_hint="受限路径和临时文件由 file_filter.py 拦截。",
        )

    try:
        preview = await asyncio.to_thread(extract_preview, resolved_path, normalized_depth)
    except StructuredPreviewError as exc:
        return build_error_response(
            exc.error_type,
            exc.message,
            retryable=exc.retryable,
            user_hint=exc.user_hint,
            operator_hint=exc.operator_hint,
        )
    except UnsupportedFileTypeError as exc:
        return build_error_response(
            "unsupported_file_type",
            str(exc),
            user_hint="当前文件类型暂不支持预览。",
            operator_hint="如需支持该格式，请在 file_extractors.py 中补对应 extractor。",
        )
    except OSError as exc:
        return build_error_response(
            "preview_failed",
            f"读取文件失败：{exc}",
            retryable=True,
            user_hint="文件读取失败，请稍后重试或确认文件是否被占用。",
            operator_hint="请检查文件权限、占用状态和底层解析器报错。",
        )

    return json.dumps(
        {
            "ok": True,
            "source": source,
            "candidate_source": candidate_source,
            "file_name": resolved_path.name,
            "file_path": str(resolved_path),
            "file_type": preview.file_type,
            "depth": normalized_depth,
            "preview_text": preview.preview_text,
            "metadata": preview.metadata,
        },
        ensure_ascii=False,
    )


def _resolve_target_path(
    *,
    file_index: int | None,
    file_path: str | None,
    client_id: str | None,
    candidate_source: str,
) -> tuple[str, Path]:
    if file_index is None and not file_path:
        raise ValueError("必须提供 file_index 或 file_path。")

    if file_index is not None:
        normalized_source = normalize_candidate_source(candidate_source)
        candidate = get_candidate_by_index(client_id, file_index, source=normalized_source)
        if candidate is None:
            raise ValueError("未找到对应的候选文件序号，请先重新搜索。")
        return "file_index", Path(candidate.full_path)

    assert file_path is not None
    return "file_path", Path(file_path)


def _map_argument_error(message: str) -> str:
    if message == "必须提供 file_index 或 file_path。":
        return build_error_response(
            "invalid_argument",
            message,
            user_hint="请提供候选序号或完整文件路径。",
            operator_hint="read_file_content 需要 file_index 或 file_path 二选一。",
        )
    if message == "未找到对应的候选文件序号，请先重新搜索。":
        return build_error_response(
            "candidate_not_found",
            message,
            user_hint="当前候选序号无效，请先重新搜索再预览。",
            operator_hint="指定 source 下未找到对应 client_id 的最近候选结果。",
        )
    if message == "depth 仅支持 L1、L2、L3。":
        return build_error_response(
            "invalid_argument",
            message,
            user_hint="预览深度仅支持 L1、L2、L3。",
            operator_hint="调用 read_file_content 时请传入合法 depth。",
        )
    if message == "candidate_source 仅支持 search_files 或 list_folder_contents。":
        return build_error_response(
            "invalid_candidate_source",
            message,
            user_hint="候选来源仅支持 search_files 或 list_folder_contents。",
            operator_hint="请传入合法 candidate_source。",
        )

    return build_error_response(
        "invalid_argument",
        message,
        user_hint="工具参数有误，请调整后重试。",
        operator_hint="请检查 read_file_content 的参数组合是否合法。",
    )
