from __future__ import annotations

from threading import Lock

from mcp_servers.file_tools.utils.everything_client import SearchItem


_DEFAULT_CLIENT_KEY = "default"
_registry_lock = Lock()
_candidate_registry: dict[str, tuple[SearchItem, ...]] = {}


def normalize_client_key(client_id: str | None) -> str:
    return client_id or _DEFAULT_CLIENT_KEY


def store_candidates(client_id: str | None, candidates: list[SearchItem]) -> None:
    with _registry_lock:
        _candidate_registry[normalize_client_key(client_id)] = tuple(candidates)


def get_candidate_by_index(client_id: str | None, file_index: int) -> SearchItem | None:
    if file_index < 1:
        return None

    with _registry_lock:
        candidates = _candidate_registry.get(normalize_client_key(client_id), ())

    candidate_offset = file_index - 1
    if candidate_offset >= len(candidates):
        return None
    return candidates[candidate_offset]


def clear_candidates(client_id: str | None) -> None:
    with _registry_lock:
        _candidate_registry.pop(normalize_client_key(client_id), None)
