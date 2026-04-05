from __future__ import annotations

import json


def build_error_response(
    error_type: str,
    message: str,
    *,
    retryable: bool = False,
    user_hint: str | None = None,
    operator_hint: str | None = None,
) -> str:
    payload = {
        "ok": False,
        "error_type": error_type,
        "retryable": retryable,
        "message": message,
        # 保留 error 作为兼容字段，避免旧摘要逻辑或调用方直接断裂。
        "error": message,
        "user_hint": user_hint or message,
        "operator_hint": operator_hint or "",
    }
    return json.dumps(payload, ensure_ascii=False)
