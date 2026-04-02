from __future__ import annotations

from pathlib import Path

from config.settings import get_settings


def resolve_search_path(path: str | None) -> str | None:
    if not path:
        return None

    settings = get_settings()
    key = path.strip().lower()
    aliases = {
        "desktop": settings.paths.desktop or str(Path.home() / "Desktop"),
        "桌面": settings.paths.desktop or str(Path.home() / "Desktop"),
        "downloads": settings.paths.downloads or str(Path.home() / "Downloads"),
        "下载": settings.paths.downloads or str(Path.home() / "Downloads"),
        "documents": settings.paths.documents or str(Path.home() / "Documents"),
        "文档": settings.paths.documents or str(Path.home() / "Documents"),
    }
    return aliases.get(key, path)
