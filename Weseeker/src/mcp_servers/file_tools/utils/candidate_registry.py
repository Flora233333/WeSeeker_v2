from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal

from mcp_servers.file_tools.utils.everything_client import SearchItem

CandidateSource = Literal["search_files", "list_folder_contents"]
_DEFAULT_CLIENT_KEY = "default"
_VALID_SOURCES: tuple[CandidateSource, ...] = ("search_files", "list_folder_contents")
_registry_lock = Lock()


@dataclass(frozen=True)
class CandidateSnapshot:
    source: CandidateSource
    query: str | None
    path: str | None
    updated_at: str | None
    results: tuple[SearchItem, ...]


_candidate_registry: dict[str, dict[CandidateSource, CandidateSnapshot]] = {}


def normalize_client_key(client_id: str | None) -> str:
    return client_id or _DEFAULT_CLIENT_KEY


def normalize_candidate_source(candidate_source: str) -> CandidateSource:
    if candidate_source not in _VALID_SOURCES:
        raise ValueError("candidate_source 仅支持 search_files 或 list_folder_contents。")
    return candidate_source


def store_candidates(
    client_id: str | None,
    candidates: list[SearchItem],
    *,
    source: CandidateSource = "search_files",
    query: str | None = None,
    path: str | None = None,
    updated_at: str | None = None,
) -> None:
    normalized_client = normalize_client_key(client_id)
    with _registry_lock:
        client_snapshots = _candidate_registry.setdefault(normalized_client, {})
        client_snapshots[source] = CandidateSnapshot(
            source=source,
            query=query,
            path=path,
            updated_at=updated_at,
            results=tuple(candidates),
        )


def get_candidate_by_index(
    client_id: str | None,
    file_index: int,
    *,
    source: CandidateSource = "search_files",
) -> SearchItem | None:
    if file_index < 1:
        return None

    snapshot = get_candidate_snapshot(client_id, source=source)
    if snapshot is None:
        return None

    candidate_offset = file_index - 1
    if candidate_offset >= len(snapshot.results):
        return None
    return snapshot.results[candidate_offset]


def get_candidate_snapshot(
    client_id: str | None,
    *,
    source: CandidateSource = "search_files",
) -> CandidateSnapshot | None:
    normalized_client = normalize_client_key(client_id)
    with _registry_lock:
        client_snapshots = _candidate_registry.get(normalized_client, {})
        return client_snapshots.get(source)


def get_all_candidate_snapshots(client_id: str | None) -> dict[CandidateSource, CandidateSnapshot]:
    normalized_client = normalize_client_key(client_id)
    with _registry_lock:
        client_snapshots = _candidate_registry.get(normalized_client, {})
        return dict(client_snapshots)


def clear_candidates(client_id: str | None, *, source: CandidateSource | None = None) -> None:
    normalized_client = normalize_client_key(client_id)
    with _registry_lock:
        if source is None:
            _candidate_registry.pop(normalized_client, None)
            return

        client_snapshots = _candidate_registry.get(normalized_client)
        if not client_snapshots:
            return
        client_snapshots.pop(source, None)
        if not client_snapshots:
            _candidate_registry.pop(normalized_client, None)
