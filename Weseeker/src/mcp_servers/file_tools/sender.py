"""文件发送工具实现。

提供两个执行函数：
- execute_prepare_send：校验文件、提取预览、生成 send_token
- execute_confirm_send：消费 token、执行发送（目前 mock）
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp_servers.file_tools.utils.candidate_registry import (
    get_candidate_by_index,
    normalize_candidate_source,
)
from mcp_servers.file_tools.utils.file_extractors import (
    extract_preview,
    format_size_display,
)
from mcp_servers.file_tools.utils.file_filter import should_include_path
from mcp_servers.file_tools.utils.pending_send_registry import (
    SendFileItem,
    consume_pending_send,
    create_pending_send,
)
from mcp_servers.file_tools.utils.tool_response import build_error_response


async def execute_prepare_send(
    file_indices: list[int],
    *,
    client_id: str | None = None,
    candidate_source: str = "search_files",
) -> str:
    """校验文件有效性，提取概览，生成 send_token。"""

    if not file_indices:
        return build_error_response(
            "invalid_argument",
            "file_indices 不能为空。",
            user_hint="请指定要发送的文件序号。",
            operator_hint="prepare_send 需要至少一个 file_index。",
        )

    # 去重并保持顺序
    seen: set[int] = set()
    unique_indices: list[int] = []
    for idx in file_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    try:
        normalized_source = normalize_candidate_source(candidate_source)
    except ValueError as exc:
        return build_error_response(
            "invalid_candidate_source",
            str(exc),
            user_hint="候选来源仅支持 search_files 或 list_folder_contents。",
            operator_hint="请传入合法 candidate_source。",
        )

    files: list[SendFileItem] = []
    errors: list[dict[str, object]] = []

    for idx in unique_indices:
        candidate = get_candidate_by_index(client_id, idx, source=normalized_source)
        if candidate is None:
            errors.append({"file_index": idx, "error": "未找到对应候选序号"})
            continue

        path = Path(candidate.full_path)

        if not path.exists():
            errors.append({"file_index": idx, "error": "文件不存在或已被移动"})
            continue
        if path.is_dir():
            errors.append({"file_index": idx, "error": "不能发送文件夹"})
            continue
        if not should_include_path(str(path)):
            errors.append({"file_index": idx, "error": "受限文件，不允许发送"})
            continue

        # 提取 L1 预览用于概览
        try:
            preview = await asyncio.to_thread(extract_preview, path, "L1")
            preview_text = preview.preview_text or ""
            file_type = preview.file_type
        except Exception:
            preview_text = ""
            file_type = path.suffix.lstrip(".")

        size = candidate.size if not candidate.is_dir else 0

        files.append(
            SendFileItem(
                name=candidate.name,
                full_path=candidate.full_path,
                size=size,
                size_display=format_size_display(size),
                preview_text=preview_text,
                file_type=file_type,
            )
        )

    if not files:
        return build_error_response(
            "no_valid_files",
            "没有可发送的有效文件。",
            user_hint="所有指定文件均无法发送，请检查序号后重试。",
            operator_hint=json.dumps(errors, ensure_ascii=False),
        )

    token = create_pending_send(client_id, files)

    return json.dumps(
        {
            "ok": True,
            "send_token": token,
            "file_count": len(files),
            "files": [
                {
                    "name": f.name,
                    "full_path": f.full_path,
                    "size": f.size,
                    "size_display": f.size_display,
                    "file_type": f.file_type,
                    "preview_text": f.preview_text,
                }
                for f in files
            ],
            "errors": errors if errors else None,
            "notice": "请向用户展示文件概览并获取确认后，再调用 confirm_send。",
        },
        ensure_ascii=False,
    )


async def execute_confirm_send(
    send_token: str,
    *,
    client_id: str | None = None,
) -> str:
    """消费 token，执行文件发送（目前 mock 返回成功）。"""

    if not send_token:
        return build_error_response(
            "invalid_argument",
            "send_token 不能为空。",
            user_hint="缺少发送令牌，请重新准备发送。",
            operator_hint="confirm_send 需要 prepare_send 返回的 send_token。",
        )

    pending = consume_pending_send(send_token, client_id)
    if pending is None:
        return build_error_response(
            "token_expired_or_invalid",
            "发送令牌无效或已过期，请重新调用 prepare_send。",
            user_hint="发送令牌已失效，请重新准备发送。",
            operator_hint="token 可能已过期（TTL 300s）或 client_id 不匹配。",
        )

    # ===== 真正的发送逻辑（目前 mock）=====
    results = []
    for f in pending.files:
        # TODO: 接入实际发送系统（邮件、微信、IM 等）
        results.append(
            {
                "name": f.name,
                "full_path": f.full_path,
                "status": "sent",
                "message": "发送成功",
            }
        )

    return json.dumps(
        {
            "ok": True,
            "send_token": pending.token,
            "file_count": len(results),
            "results": results,
        },
        ensure_ascii=False,
    )

