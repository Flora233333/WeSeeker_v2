from __future__ import annotations

import json
from datetime import datetime

from mcp.server.fastmcp import Context, FastMCP

from config.settings import get_settings
from mcp_servers.file_tools.folder import execute_list_folder_contents
from mcp_servers.file_tools.reader import execute_read_content
from mcp_servers.file_tools.search import execute_get_current_candidates, execute_search

mcp = FastMCP(
    name="weseeker-file-tools",
    instructions="WeSeeker MVP 文件搜索工具集",
    host="127.0.0.1",
    port=get_settings().mcp.file_tools_port,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def get_current_time() -> str:
    """
    获取当前系统时间。

    用途：
    - 当用户直接询问当前时间时
    - 当模型需要确认“现在”这一时点，以便解释搜索结果、修改时间或快照时间时

    Returns:
        JSON 字符串，成功结构如下：
        - `ok`: 是否成功
        - `current_time`: 当前系统本地时间，格式为 ISO 8601，精确到秒
        - `timezone`: 固定返回 `local`
    """
    return json.dumps(
        {
            "ok": True,
            "current_time": datetime.now().isoformat(timespec="seconds"),
            "timezone": "local",
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def search_files(
    keyword: str,
    path: str | None = None,
    max_results: int = 20,
    ctx: Context | None = None,
) -> str:
    """
    在本地电脑中按文件名搜索文件，返回经过安全过滤后的候选列表。

    用途：
    - 当用户记得文件名关键词，但不确定具体位置时
    - 当用户给出了位置范围（如桌面、下载目录或某个绝对路径）时
    - 当后续需要继续使用 `file_index` 调用 `read_file_content` 或 `list_folder_contents` 时

    使用方法：
    - 这是生成 `search_files` candidates 的唯一入口
    - 每次新的 `search_files` 调用都会覆盖当前客户端下旧的 `search_files` candidates
    - 如果后续要调用 `read_file_content` 或 `list_folder_contents`，
      通常应先调用本工具拿到最新 `file_index`

    Args:
        keyword: 搜索关键词。应尽量提炼为文件名里可能真实出现的词，不能为空。
        path: 搜索范围，可为空。支持系统别名（用户已经自动定义了别名路径，由程序解析） `desktop`/`桌面`、`downloads`/`下载`
            、`documents`/`文档`，也支持用户直接提供的绝对路径。
        max_results: 最大返回结果数。默认 20；建议保持在 1 到 50 之间，避免候选过长。

    Returns:
        JSON 字符串，成功结构如下：
        - `ok`: 是否成功
        - `keyword`: 清洗后的搜索关键词
        - `path`: 实际使用的搜索范围；为空时表示全局搜索
        - `count`: 返回的候选数量
        - `results`: 候选列表，每项包含：
          - `index`: 候选序号，从 1 开始
          - `name`: 文件或文件夹名
          - `path`: 父目录路径
          - `full_path`: 完整路径
          - `size`: 文件大小（字节）。普通文件为整数；文件夹固定为 `null`
          - `size_display`: 文件大小的可读显示值。普通文件如 `12.3 KB`；文件夹固定为 `folder`
          - `modified`: 修改时间字符串
          - `is_dir`: 是否为文件夹（目录）。`true` 表示目录，`false` 表示普通文件

        失败时返回：
        - `ok: false`
        - `error_type`: 结构化错误类型
        - `message`: 可读错误信息
        - `user_hint`: 面向模型/用户的提示
        - `operator_hint`: 面向开发调试的提示

        备注：
        - 搜索结果会写入 `search_files` candidates
        - 后续若要继续依赖 `file_index`，请注意它属于 `search_files`
          candidates，而不是 `list_folder_contents` candidates
        - 搜索阶段会尽量识别真实目录，并过滤 0 B 的普通文件噪音；
          文件夹不会因为大小显示为 `0 B` 而与空文件混淆
    """
    client_id = ctx.client_id if ctx is not None else None
    return await execute_search(
        keyword=keyword,
        path=path,
        max_results=max_results,
        client_id=client_id,
    )


@mcp.tool()
async def get_current_candidates(ctx: Context | None = None) -> str:
    """
    查看当前客户端下最新的 candidates 快照。

    用途：
    - 当模型准备继续使用 `file_index`，但不确定它属于哪一类 candidates 时
    - 当上下文已经发生新搜索、目录展开或阶段切换，需要重新确认当前候选集时
    - 用于避免把旧阶段的 `file_index` 幻觉带到当前阶段

    使用方法：
    - 当前系统中有两类 candidates：
      - `search_files`
      - `list_folder_contents`
    - 这两类 candidates 各自维护独立的 `file_index`
    - 如果你拿不准后续工具应使用哪一类 `candidate_source`，先调用本工具确认

    Args:
        无显式业务参数。工具会自动基于当前客户端读取最新 candidates 快照。

    Returns:
        JSON 字符串，成功结构如下：
        - `ok`: 是否成功
        - `sources`: 当前 candidates 快照，按来源分组：
          - `search_files`
          - `list_folder_contents`

        每个来源下包含：
        - `has_candidates`: 当前该来源下是否存在候选
        - `updated_at`: 最近一次写入该来源 candidates 的时间
        - `query`: 该来源对应的搜索词（如有）
        - `path`: 该来源对应的路径范围（如有）
        - `count`: 当前候选数量
        - `results`: 当前候选列表，字段与 `search_files` 结果项一致
          - `size`: 普通文件为整数；文件夹固定为 `null`
          - `size_display`: 普通文件为可读大小；文件夹固定为 `folder`
          - `is_dir`: 是否为文件夹（目录）

        当某个来源当前没有可用 candidates 时：
        - `has_candidates: false`
        - `message`: 当前该来源下没有可用 candidates
        - `count: 0`
        - `results: []`

    备注：
    - 本工具只读，不会修改任何 candidates
    - 当你准备继续使用 `file_index` 时，优先用它确认当前 `candidate_source`
    - `search_files.path` 表示搜索范围；`list_folder_contents.path` 表示最近一次展开的文件夹路径
    """
    client_id = ctx.client_id if ctx is not None else None
    return await execute_get_current_candidates(client_id=client_id)


@mcp.tool()
async def list_folder_contents(
    file_index: int,
    max_results: int = 30,
    candidate_source: str = "search_files",
    ctx: Context | None = None,
) -> str:
    """
    列出指定文件夹的直属子项。

    用途：
    - 当用户想查看某个文件夹里有什么时
    - 当模型已经通过 `search_files` 找到了目标文件夹，想进一步展开其直属内容时
    - 只列出直属子项，不做递归遍历

    使用方法：
    - 调用本工具前，必须先通过 `search_files` 找到目标文件夹
    - 再使用该文件夹在 `search_files` candidates 中对应的 `file_index`
    - 默认 `candidate_source="search_files"`
    - 调用成功后，返回的直属子项会写入 `list_folder_contents` candidates
    - 如果之后要继续使用这些子项的 `file_index`，应改用 `candidate_source="list_folder_contents"`

    Args:
        file_index: 目标文件夹在当前候选列表中的序号。
            默认应来自 `search_files` candidates。
        max_results: 最大返回子项数。默认 30。
        candidate_source: 当前 `file_index` 所属的 candidates 来源。
            支持：`search_files`、`list_folder_contents`。
            默认值为 `search_files`。

    Returns:
        JSON 字符串，成功结构如下：
        - `ok`: 是否成功
        - `source`: 固定为 `file_index`
        - `candidate_source`: 本次解析 `file_index` 所使用的来源
        - `folder_name`: 文件夹名
        - `folder_path`: 文件夹完整路径
        - `updated_at`: 本次目录展开结果写入时间
        - `count`: 返回的直属子项数量
        - `results`: 子项列表，每项包含：
          - `index`: 子项在 `list_folder_contents` candidates 中的序号
          - `name`: 子项名称
          - `path`: 子项父目录路径
          - `full_path`: 子项完整路径
          - `size`: 文件大小（字节）。普通文件为整数；文件夹固定为 `null`
          - `size_display`: 文件大小的可读显示值。普通文件如 `12.3 KB`；文件夹固定为 `folder`
          - `modified`: 修改时间字符串
          - `is_dir`: 是否为文件夹（目录）
        - `notice`: 关于 candidates/source 使用方式的提醒

        失败时返回：
        - `ok: false`
        - `error_type`: 结构化错误类型
        - `message`: 可读错误信息
        - `user_hint`: 面向模型/用户的提示
        - `operator_hint`: 面向开发调试的提示

    备注：
    - `list_folder_contents` 的结果会写入独立的 `list_folder_contents` candidates
    - 这些结果与 `search_files` candidates 的 `file_index` 不能混用
    """
    client_id = ctx.client_id if ctx is not None else None
    return await execute_list_folder_contents(
        file_index=file_index,
        max_results=max_results,
        client_id=client_id,
        candidate_source=candidate_source,
    )


@mcp.tool()
async def read_file_content(
    file_index: int | None = None,
    file_path: str | None = None,
    depth: str = "L1",
    candidate_source: str = "search_files",
    ctx: Context | None = None,
) -> str:
    """
    读取文件内容或基础元信息，返回结构化预览结果。

    用途：
    - 已通过 `search_files` 或 `list_folder_contents` 找到候选文件，需要进一步确认内容时
    - 用户需要快速预览文本、文档、表格、PPT、PDF 或图片的基础信息时
    - 需要避免模型自行拼接路径，优先通过 `file_index` 读取当前候选结果时

    使用方法：
    - 如果使用 `file_index`，必须同时确认它来自哪一类 candidates
    - 默认 `candidate_source="search_files"`
    - 如果文件来自目录展开结果，必须显式使用 `candidate_source="list_folder_contents"`
    - 如果不确定当前 `file_index` 属于哪类 candidates，先调用 `get_current_candidates`
    - `file_path` 仅作为兜底输入，在没有可用 candidates 时使用

    Args:
        file_index: 候选文件在当前候选列表中的序号。
            与 `candidate_source` 一起使用。
        file_path: 文件绝对路径。仅在无法使用候选序号时作为兜底输入。
        depth: 预览深度，支持 `L1`、`L2`、`L3`。
        candidate_source: 当前 `file_index` 所属的 candidates 来源。
            支持：`search_files`、`list_folder_contents`。
            默认值为 `search_files`。

    Returns:
        JSON 字符串，成功结构如下：
        - `ok`: 是否成功
        - `source`: 路径来源，`file_index` 或 `file_path`
        - `candidate_source`: 本次解析 `file_index` 所使用的来源
        - `file_name`: 文件名
        - `file_path`: 文件完整路径
        - `file_type`: 文件类型
        - `depth`: 实际使用的预览深度
        - `preview_text`: 文本预览或图片/页面预览摘要
        - `metadata`: 文件大小（`size`）、可读文件大小（`size_display`）、
          修改时间以及类型专属 metadata

        失败时返回：
        - `ok: false`
        - `error_type`: 结构化错误类型
        - `message`: 可读错误信息
        - `user_hint`: 面向模型/用户的提示
        - `operator_hint`: 面向开发调试的提示

    备注：
    - 使用 `file_index` 时，`candidate_source` 与 `file_index` 必须匹配
    - 相同数字的 `file_index` 在不同 candidates 来源下可能指向完全不同的文件
    """
    client_id = ctx.client_id if ctx is not None else None
    return await execute_read_content(
        file_index=file_index,
        file_path=file_path,
        depth=depth,
        client_id=client_id,
        candidate_source=candidate_source,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
