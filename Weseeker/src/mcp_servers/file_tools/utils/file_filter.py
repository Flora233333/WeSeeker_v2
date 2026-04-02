from __future__ import annotations

import os
import stat
from pathlib import Path


BLOCKED_PATH_PARTS = {
    "$Recycle.Bin",
    ".git",
    "node_modules",
    "Windows",
    "Program Files",
    "ProgramData",
}

TEMPORARY_FILE_SUFFIXES = {
    ".tmp",
    ".temp",
}

TEMPORARY_FILE_PREFIXES = (
    "~$",
    ".~",
)

WINDOWS_BLOCKED_FILE_ATTRIBUTES = (
    stat.FILE_ATTRIBUTE_SYSTEM,
    stat.FILE_ATTRIBUTE_TEMPORARY,
)


def _is_temporary_name(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith(TEMPORARY_FILE_PREFIXES) or path.suffix.lower() in TEMPORARY_FILE_SUFFIXES


def _has_blocked_windows_attributes(path: Path) -> bool:
    try:
        file_attributes = os.stat(path, follow_symlinks=False).st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False

    return any(file_attributes & attribute for attribute in WINDOWS_BLOCKED_FILE_ATTRIBUTES)


def should_include_path(full_path: str) -> bool:
    path = Path(full_path)
    if any(part in BLOCKED_PATH_PARTS for part in path.parts):
        return False
    if _is_temporary_name(path):
        return False
    if _has_blocked_windows_attributes(path):
        return False
    return True
