from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from config.settings import get_settings
from mcp_servers.file_tools.utils.file_filter import should_include_path


LOW_PRIORITY_NAME_PREFIXES = ("_", ".", "~$")


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


def _escape_everything_text(text: str) -> str:
    return text.replace('"', '""').strip()


def _scoped_query(search: str, search_path: str | None) -> str:
    if not search_path:
        return search
    base = str(Path(search_path)).rstrip("\\/")
    # [改动 1]
    # 把路径约束前置到 Everything 查询里，而不是先全局搜再 Python 里截断。
    return f'"{base}\\\" {search}'


def _is_simple_short_keyword(keyword: str) -> bool:
    kw = keyword.strip()
    return bool(kw) and len(kw) <= 8 and re.fullmatch(r"[A-Za-z0-9._-]+", kw) is not None


def _score_item(item: SearchItem, keyword: str) -> tuple[int, int, int, int, str]:
    q = keyword.lower().strip()
    name = item.name.lower()
    stem = item.name.lower() if item.is_dir else Path(item.name).stem.lower()

    score = 0

    # [改动 2]
    # 目录优先，避免用户找“文件夹”时目录被普通文件压下去。
    if item.is_dir:
        score += 300

    # 精确匹配 > whole word > 前缀 > 包含
    if stem == q or name == q:
        score += 5000
    elif re.search(rf"(?<![0-9a-z_]){re.escape(q)}(?![0-9a-z_])", stem):
        score += 3000
    elif stem.startswith(q):
        score += 2000
    elif q in stem:
        score += 1000
    elif q in name:
        score += 500

    # [改动 3]
    # _xx /.xx / ~$xx 这类结果降权，但不直接误杀。
    if name.startswith(LOW_PRIORITY_NAME_PREFIXES):
        score -= 400

    # 名字越短越靠前；路径越浅越靠前
    depth = item.full_path.count("\\") + item.full_path.count("/")
    return (-score, len(stem), depth, len(item.full_path), name)


def _detect_is_dir(item: dict, full_path: str) -> bool:
    # [改动 4]
    # 先信任 Everything 返回的显式目录字段；
    # 如果没有，再回退到本地文件系统判断，避免字段缺失时误判成 False。
    raw_is_folder = item.get("is_folder")
    if raw_is_folder is not None:
        return bool(raw_is_folder)

    raw_is_dir = item.get("is_dir")
    if raw_is_dir is not None:
        return bool(raw_is_dir)

    try:
        return Path(full_path).is_dir()
    except OSError:
        return False


class EverythingClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = f"http://{settings.everything.host}:{settings.everything.port}"
        self._client = httpx.AsyncClient(timeout=10.0)

    async def _fetch(self, query: str, count: int) -> list[dict]:
        params = {
            "search": query,
            "json": 1,
            "path_column": 1,
            "size_column": 1,
            "date_modified_column": 1,
            "count": count,
            "sort": "name",
            "ascending": 1,
        }
        response = await self._client.get(self._base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])

    async def search(
        self,
        keyword: str,
        *,
        search_path: str | None = None,
        max_results: int = 20,
    ) -> list[SearchItem]:
        keyword = keyword.strip()
        if not keyword:
            return []

        # [改动 5]
        # 提高短关键词召回量，避免 v2 这种词在前 50/100 条就被截断。
        fetch_count = min(max(max_results * 20, 200), 1000)

        queries: list[str] = []

        # [改动 6]
        # 短关键词优先用更精确的目录查询。
        if _is_simple_short_keyword(keyword):
            kw = _escape_everything_text(keyword)
            queries.append(_scoped_query(f'folder: wfn:"{kw}"', search_path))
            queries.append(_scoped_query(f'folder: ww:"{kw}"', search_path))
            queries.append(_scoped_query(f'ww:"{kw}"', search_path))

        # 最后再走普通召回
        queries.append(_scoped_query(keyword, search_path))

        merged: list[SearchItem] = []
        seen: set[str] = set()

        for query in queries:
            raw_results = await self._fetch(query, fetch_count)

            for item in raw_results:
                name = item.get("name") or ""
                parent = item.get("path") or ""
                if not name or not parent:
                    continue

                full_path = str(Path(parent) / name)
                full_path_key = full_path.lower()

                if full_path_key in seen:
                    continue
                if not should_include_path(full_path):
                    continue

                # [改动 7]
                # 用更稳的目录检测逻辑，不再依赖 HTTP 返回一定带 is_folder/is_dir。
                is_dir = _detect_is_dir(item, full_path)

                # [改动 8]
                # 过滤所有 0 B 普通文件；文件夹不参与这个过滤。
                size = 0 if is_dir else int(item.get("size") or 0)
                if not is_dir and size == 0:
                    continue

                merged.append(
                    SearchItem(
                        name=name,
                        path=parent,
                        size=size,
                        modified=str(item.get("date_modified") or ""),
                        is_dir=is_dir,
                    )
                )
                seen.add(full_path_key)

        merged.sort(key=lambda x: _score_item(x, keyword))
        return merged[:max_results]

    async def list_children(self, folder_path: str, *, max_results: int = 30) -> list[SearchItem]:
        query = f'parent:"{folder_path}"'
        return await self.search(query, max_results=max_results)

    async def close(self) -> None:
        await self._client.aclose()
