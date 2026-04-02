from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from config.settings import get_settings
from mcp_servers.file_tools.utils.file_filter import should_include_path


@dataclass(frozen=True)
class SearchItem:
    name: str
    path: str
    size: int
    modified: str
    is_dir: bool = False

    @property
    def full_path(self) -> str:
        return str(Path(self.path) / self.name)


class EverythingClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = f"http://{settings.everything.host}:{settings.everything.port}"
        self._client = httpx.AsyncClient(timeout=10.0)

    async def search(
        self,
        keyword: str,
        *,
        search_path: str | None = None,
        max_results: int = 20,
    ) -> list[SearchItem]:
        params = {
            "search": keyword,
            "json": 1,
            "path_column": 1,
            "size_column": 1,
            "date_modified_column": 1,
            "count": min(max(max_results * 5, 50), 200),
        }
        response = await self._client.get(self._base_url, params=params)
        response.raise_for_status()

        data = response.json()
        raw_results = data.get("results", [])
        normalized: list[SearchItem] = []
        search_path_lower = search_path.lower() if search_path else None

        for item in raw_results:
            name = item.get("name") or ""
            parent = item.get("path") or ""
            full_path = str(Path(parent) / name)

            if not name or not parent:
                continue
            if search_path_lower and not full_path.lower().startswith(search_path_lower):
                continue
            if not should_include_path(full_path):
                continue

            normalized.append(
                SearchItem(
                    name=name,
                    path=parent,
                    size=int(item.get("size") or 0),
                    modified=str(item.get("date_modified") or ""),
                    is_dir=bool(item.get("is_folder") or item.get("is_dir") or False),
                )
            )
            if len(normalized) >= max_results:
                break

        return normalized

    async def close(self) -> None:
        await self._client.aclose()
