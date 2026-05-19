from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from config.settings import get_settings
from mcp_servers.rag_tools.search_kb import search_kb as execute_search_kb

mcp = FastMCP(
    name="weseeker-rag-tools",
    instructions="WeSeeker RAG 知识库检索工具集",
    host="127.0.0.1",
    port=get_settings().mcp.rag_tools_port,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def search_kb(
    query: str,
    kb_name: str,
    top_k: int = 5,
    candidate_k: int = 30,
) -> str:
    """按知识库内容检索相关笔记或文档片段，返回面向 Agent 的证据摘要。

    用途：
    - 当用户按内容、主题、知识点或自然语言问题查找自己写过的资料时
    - 当用户问“我之前有没有写过 X”“某个知识点在哪篇笔记里”时
    - 当文件名或路径不明确，需要从文档正文中找证据时

    Agent 使用规则：
    - 本工具只负责“按内容找证据”，不会创建 file_tools candidates
    - 返回结果中的 `rank` 只是 RAG 结果排序，不是 `file_index`
    - 如果用户追问“第几个预览一下”“打开看看”“发过来”，必须先调用
      `search_files(keyword=<source.file_name>)`，用 Everything 搜索同名文件并建立
      file_tools candidates
    - 建立 candidates 后，再用 `read_file_content(file_index=..., candidate_source="search_files")`
      或 `prepare_send(file_indices=[...], candidate_source="search_files")`
    - 不要把 RAG 的 `rank` 当作 file_tools 的 `file_index`
    - 本工具返回客观 `retrieval_signals`；是否重搜、是否回答、是否让用户补充关键词，
      由 Agent 自己根据结果判断

    不适用：
    - 如果用户明确给出文件名、路径或目录线索，优先使用 file_tools 的文件搜索工具
    - 如果用户只是要查看当前候选文件内容，优先使用 read_file_content

    Args:
        query: 用户原始检索问题，不能为空。当前只做首尾空白清理，不做 Query Rewrite 或 HyDE。
        kb_name: 已完成离线索引的知识库名称，例如 `test_kb_notes`。
        top_k: 最终返回的 parent evidence 数量，默认 5。
        candidate_k: Vector / BM25 两路各自召回的 child candidate 数量，默认 30。

    Returns:
        JSON 字符串，成功时是精简的 agent-facing 结构：
        - `ok`: 是否成功
        - `kb_name`: 查询的知识库
        - `query`: 实际检索 query
        - `result_count`: 返回结果数量
        - `results`: 证据列表，每项包含 `rank`、`source`、`evidence`、`retrieval_signals`
        - `retrieval_signals`: 顶层客观检索信号，供 Agent 自行判断是否需要重搜
        - `usage_notice`: 后续预览/发送必须先 `search_files(source.file_name)` 的提醒

        失败时包含 `ok=false`、`error_type`、`message`、`user_hint`、`operator_hint`。

    备注：
    - MCP 返回默认不包含 `file_path`、`diagnostics`、`scores`、`rank_score` 等内部调试字段。
    - 本工具只检索已有索引，不负责构建或更新知识库。
    """
    payload = execute_search_kb(
        query=query,
        kb_name=kb_name,
        top_k=top_k,
        candidate_k=candidate_k,
    )
    return json.dumps(_format_agent_payload(payload), ensure_ascii=False)


def _format_agent_payload(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("ok") is not True:
        return payload

    results = _list_of_dicts(payload.get("results"))
    projected_results = [_format_agent_result(result) for result in results]
    top_result = projected_results[0] if projected_results else {}
    top_signals = top_result.get("retrieval_signals") if isinstance(top_result, dict) else {}

    return {
        "ok": True,
        "kb_name": payload.get("kb_name"),
        "query": payload.get("query"),
        "result_count": len(projected_results),
        "results": projected_results,
        "retrieval_signals": {
            "result_count": len(projected_results),
            "top_sources": _string_list(
                top_signals.get("sources") if isinstance(top_signals, dict) else None
            ),
            "top_matched_snippet_count": _int_or_zero(
                top_signals.get("matched_snippet_count") if isinstance(top_signals, dict) else None
            ),
            "content_tokens": _content_tokens(payload),
        },
        "usage_notice": (
            "RAG results are evidence only. To preview or send a file, call "
            "search_files with source.file_name first, then use file_tools candidates."
        ),
    }


def _format_agent_result(result: dict[str, object]) -> dict[str, object]:
    matched_snippets = _matched_snippets(result)
    return {
        "rank": result.get("rank"),
        "source": {
            "file_name": result.get("file_name"),
            "doc_type": result.get("doc_type"),
            "heading_path": result.get("heading_path"),
            "section_title": result.get("section_title"),
            "page_number": result.get("page_number"),
        },
        "evidence": {
            "snippet": result.get("snippet") or "",
            "matched_snippets": matched_snippets,
        },
        "retrieval_signals": {
            "sources": _string_list(result.get("retrieval_sources")),
            "matched_snippet_count": len(matched_snippets),
        },
    }


def _matched_snippets(result: dict[str, object]) -> list[str]:
    matched_children = _list_of_dicts(result.get("matched_children"))
    snippets: list[str] = []
    for child in matched_children:
        snippet = child.get("snippet")
        if isinstance(snippet, str) and snippet:
            snippets.append(snippet)
    return snippets


def _content_tokens(payload: dict[str, object]) -> list[str]:
    query_profile = payload.get("query_profile")
    if not isinstance(query_profile, dict):
        return []
    return _string_list(query_profile.get("content_tokens"))


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
