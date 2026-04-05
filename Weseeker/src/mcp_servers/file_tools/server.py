from __future__ import annotations

from mcp.server.fastmcp import Context, FastMCP

from config.settings import get_settings
from mcp_servers.file_tools.reader import execute_read_content
from mcp_servers.file_tools.search import execute_search


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
async def search_files(
    keyword: str,
    path: str | None = None,
    max_results: int = 20,
    ctx: Context | None = None,
) -> str:
    """
    在本地电脑中按文件名搜索文件，返回经过安全过滤后的候选列表。

    适用场景：
    - 用户记得文件名关键词，但不确定具体位置
    - 用户给出了范围线索，例如桌面、下载目录或某个磁盘路径
    - 需要先拿到候选列表，再由 Agent 继续做确认或追问

    搜索规则：
    - 使用 Everything HTTP API 做文件名检索
    - 会过滤系统目录、Windows 系统文件、临时文件和常见开发噪音目录
    - 当前阶段只做文件名级搜索，不读取文件内容

    Args:
        keyword: 搜索关键词。应尽量提炼为文件名里可能真实出现的词，不能为空。
        path: 搜索范围，可为空。支持系统别名 `desktop`/`桌面`、`downloads`/`下载`
            、`documents`/`文档`，也支持用户直接提供的绝对路径。
        max_results: 最大返回结果数。默认 20；建议保持在 1 到 50 之间，避免候选过长。

    Returns:
        JSON 字符串，结构如下：
        - `ok`: 是否成功
        - `keyword`: 清洗后的搜索关键词
        - `path`: 实际使用的搜索范围；为空时表示全局搜索
        - `count`: 返回的候选数量
        - `results`: 候选列表，每项包含：
          - `index`: 候选序号，从 1 开始
          - `name`: 文件或文件夹名
          - `path`: 父目录路径
          - `full_path`: 完整路径
          - `size`: 文件大小（字节）
          - `size_display`: 文件大小的可读显示值，例如 `12.4 KB`、`3.1 MB`
          - `modified`: 修改时间字符串
          - `is_dir`: 是否为文件夹（目录）

        失败时返回：
        - `ok: false`
        - `error`: 可读错误信息
    """
    client_id = ctx.client_id if ctx is not None else None
    return await execute_search(
        keyword=keyword,
        path=path,
        max_results=max_results,
        client_id=client_id,
    )


@mcp.tool()
async def read_file_content(
    file_index: int | None = None,
    file_path: str | None = None,
    depth: str = "L1",
    ctx: Context | None = None,
) -> str:
    """
    读取文件内容或基础元信息，返回结构化预览结果。

    适用场景：
    - 已通过 `search_files` 找到候选文件，需要进一步确认内容
    - 用户需要快速预览文本、文档、表格、PPT、PDF 或图片的基础信息
    - 需要避免模型自行拼接路径，优先通过 `file_index` 读取最近一次搜索结果

    预览规则：
    - 优先使用 `file_index`，从当前客户端最近一次 `search_files` 的候选结果中解析真实路径
    - `file_path` 仅作为兜底输入，在没有候选列表时使用
    - 当前阶段只提供文本提取和 metadata 预览，不做 OCR 或多模态视觉摘要
    - 支持 `txt/md/py/json/yaml/yml/log/csv/docx/xlsx/pptx/pdf` 和常见图片格式

    Args:
        file_index: 最近一次搜索结果中的候选序号，推荐优先使用。
        file_path: 文件绝对路径。仅在无法使用候选序号时使用。
        depth: 预览深度，支持 `L1`、`L2`、`L3`。

    Returns:
        JSON 字符串，结构如下：
        - `ok`: 是否成功
        - `source`: 路径来源，`file_index` 或 `file_path`
        - `file_name`: 文件名
        - `file_path`: 文件完整路径
        - `file_type`: 文件类型
        - `depth`: 实际使用的预览深度
        - `preview_text`: 文本预览；图片等非文本文件返回说明文字
        - `metadata`: 文件大小（`size`）、可读文件大小（`size_display`）、修改时间以及类型专属 metadata

        失败时返回：
        - `ok: false`
        - `error`: 可读错误信息
    """
    client_id = ctx.client_id if ctx is not None else None
    return await execute_read_content(
        file_index=file_index,
        file_path=file_path,
        depth=depth,
        client_id=client_id,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
