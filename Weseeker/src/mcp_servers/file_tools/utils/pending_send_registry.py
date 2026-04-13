"""待发送批次的临时存储与 token 管理。

每次 prepare_send 成功后，生成一个短期 token 绑定待发送文件列表；
confirm_send 使用该 token 取出并消费，用完即删。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

_registry_lock = Lock()
_pending_sends: dict[str, "PendingSend"] = {}

# token 有效期（秒）
_TOKEN_TTL = 300


def _is_expired(pending: "PendingSend", *, now: datetime) -> bool:
    return (now - pending.created_at).total_seconds() > _TOKEN_TTL


def _purge_expired_locked(*, now: datetime) -> None:
    expired = [
        token
        for token, pending in _pending_sends.items()
        if _is_expired(pending, now=now)
    ]
    for token in expired:
        del _pending_sends[token]


@dataclass(frozen=True)
class SendFileItem:
    """待发送文件的结构化信息。"""

    name: str
    full_path: str
    size: int
    size_display: str
    modified: str | None
    preview_text: str
    file_type: str


@dataclass
class PendingSend:
    """一次待发送批次。"""

    token: str
    client_id: str | None
    files: list[SendFileItem]
    created_at: datetime = field(default_factory=datetime.now)


def create_pending_send(
    client_id: str | None,
    files: list[SendFileItem],
) -> str:
    """创建待发送批次，返回 token。同时清理过期 token。"""
    token = uuid.uuid4().hex[:12]
    with _registry_lock:
        now = datetime.now()
        _purge_expired_locked(now=now)

        _pending_sends[token] = PendingSend(
            token=token,
            client_id=client_id,
            files=files,
        )
    return token


def consume_pending_send(token: str, client_id: str | None) -> PendingSend | None:
    """按 token 取出并删除待发送批次。校验 client_id 一致性。"""
    with _registry_lock:
        now = datetime.now()
        _purge_expired_locked(now=now)

        pending = _pending_sends.get(token)
        if pending is None:
            return None
        if pending.client_id != client_id:
            return None

        return _pending_sends.pop(token)


def clear_pending_sends(client_id: str | None) -> int:
    """清空当前 client_id 下的所有待发送批次，返回清理数量。"""
    with _registry_lock:
        now = datetime.now()
        _purge_expired_locked(now=now)

        cleared_tokens = [
            token for token, pending in _pending_sends.items() if pending.client_id == client_id
        ]
        for token in cleared_tokens:
            del _pending_sends[token]

        return len(cleared_tokens)

