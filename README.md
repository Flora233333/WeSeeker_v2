# WeSeeker

WeSeeker 是一个运行在 Windows PC 上的智能文件管家 Agent。用户通过自然语言描述文件需求，系统在本地完成文件搜索、内容预览、后续可扩展到知识库检索与安全发送。

当前状态：开发中，已完成文件搜索 / 文件预览 MVP 主链路。

## 项目目标

- 使用 LangChain + LangGraph 组织 Agent 主流程
- 使用 FastMCP 提供标准化工具服务
- 使用 `interrupt()` + `Command(resume=...)` 实现框架级人工确认
- 在 Windows 本地环境中提供稳定、可恢复的文件管家能力

## 运行环境

- 操作系统：Windows
- Python 环境：conda `lang_agent`
- 项目代码目录：`Weseeker/`

## 快速开始

以下命令在 `Weseeker/` 目录执行：

```bash
conda activate lang_agent
cd Weseeker
pip install -e .[dev]
```

## 常用命令

```bash
# 启动文件工具 MCP Server
python scripts/start_mcp_servers.py

# 启动 CLI Agent
python scripts/run_agent.py

# 运行测试
pytest
```

## 当前已可用能力

- `search_files`：文件名搜索
- `read_file_content`：文件内容与基础预览
- CLI 调试入口：展示 `[Tool Trace]`

## 开发说明

当前仓库已从工程骨架进入 MVP 联调阶段。

- `scripts/start_mcp_servers.py` 会先探测 `9100` 端口；若目标 `file_tools` MCP 服务已在运行，会直接给出友好提示，不再抛出难看的端口绑定异常
- `src/listeners/cli.py` 默认展示 `[Tool Trace]`，方便观察 Agent 是否真实触发工具调用
- 调试前应先检查 MCP 端口是否已开启
- 如果修改了 MCP 相关文件，应先关闭原有端口监听，再重新运行对应服务后继续调试

后续将继续补自动化测试，并推进 `send_file`、HITL、RAG 等能力。
