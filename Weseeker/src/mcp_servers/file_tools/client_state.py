from __future__ import annotations

from dataclasses import dataclass

from mcp_servers.file_tools.utils.candidate_registry import (
    clear_candidates,
    get_all_candidate_snapshots,
)
from mcp_servers.file_tools.utils.pending_send_registry import clear_pending_sends


@dataclass(frozen=True)
class ClearClientStateResult:
    client_id: str | None
    cleared_candidate_source_count: int
    cleared_candidate_item_count: int
    cleared_pending_count: int


def clear_client_state(client_id: str | None) -> ClearClientStateResult:
    snapshots = get_all_candidate_snapshots(client_id)
    cleared_candidate_source_count = len(snapshots)
    cleared_candidate_item_count = sum(len(snapshot.results) for snapshot in snapshots.values())

    clear_candidates(client_id)
    cleared_pending_count = clear_pending_sends(client_id)

    return ClearClientStateResult(
        client_id=client_id,
        cleared_candidate_source_count=cleared_candidate_source_count,
        cleared_candidate_item_count=cleared_candidate_item_count,
        cleared_pending_count=cleared_pending_count,
    )
