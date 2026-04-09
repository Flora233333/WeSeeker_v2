from __future__ import annotations

import os
import stat
from pathlib import Path


BLOCKED_PATH_PARTS = {
    "$Recycle.Bin",
    ".cache",
    ".git",
    ".ssh",
    ".aws",
    ".vscode",
    "__pycache__",
    "node_modules",
    "AppData",
    "ProgramData",
    "Program Files",
    "Program Files (x86)",
    "System Volume Information",
    "Windows",
}

TEMPORARY_FILE_SUFFIXES = {
    ".tmp",
    ".temp",
    ".bak",
    ".swp",
}

TEMPORARY_FILE_PREFIXES = (
    "~$",
    ".~",
    "._",
)

WINDOWS_BLOCKED_FILE_ATTRIBUTES = (
    stat.FILE_ATTRIBUTE_SYSTEM,
    stat.FILE_ATTRIBUTE_TEMPORARY,
)


def _is_temporary_name(path: Path) -> bool:
    name = path.name
    name_lower = name.lower()
    if any(name.startswith(prefix) for prefix in TEMPORARY_FILE_PREFIXES):
        return True
    if path.suffix.lower() in TEMPORARY_FILE_SUFFIXES:
        return True
    # thumbs.db, desktop.ini 等 Windows 系统生成文件
    if name_lower in {"thumbs.db", "desktop.ini", "ntuser.dat"}:
        return True
    return False


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

