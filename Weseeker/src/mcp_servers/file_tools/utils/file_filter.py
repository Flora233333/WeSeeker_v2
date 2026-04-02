from __future__ import annotations

from pathlib import Path


BLOCKED_PATH_PARTS = {
    "$Recycle.Bin",
    ".git",
    "node_modules",
    "Windows",
    "Program Files",
    "ProgramData",
}


def should_include_path(full_path: str) -> bool:
    path = Path(full_path)
    return not any(part in BLOCKED_PATH_PARTS for part in path.parts)
