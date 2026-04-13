# WeSeeker

WeSeeker 是一个运行在 Windows PC 上的智能文件管家 Agent。用户通过自然语言描述文件需求，系统在本地完成文件搜索、候选管理、目录展开、内容预览和发送前确认；当前真实入口包括 CLI 和 Web 调试前端。

当前状态：开发中，已完成“文件搜索 -> 候选快照 -> 目录展开 -> 文件预览 -> `prepare_send` / `confirm_send`”MVP 主链路，并已接出 FastAPI + React/Vite 的 Web 调试台。

## 项目目标

- 使用 LangChain + LangGraph 组织 Agent 主流程
- 使用 FastMCP 提供标准化工具服务
- 当前发送链路先落地 `prepare_send -> 用户确认 -> confirm_send` 两阶段确认；`interrupt()` + `Command(resume=...)` 继续保留为后续增强方向
- 在 Windows 本地环境中提供稳定、可恢复的文件管家能力

## 运行环境

- 操作系统：Windows
- Python 环境：conda `lang_agent`
- 项目代码目录：`Weseeker/`
- Web 调试前端：`Weseeker/frontend/`（React 18 + Vite 6 + Tailwind CSS 3）

## 快速开始

以下命令在 `Weseeker/` 目录执行：

```bash
conda activate lang_agent
cd Weseeker
pip install -e .[dev]
```

如果要跑 Web 调试前端，还需要：

```bash
cd frontend
npm install
```

## 常用命令

```bash
# 启动文件工具 MCP Server
python scripts/start_mcp_servers.py

# 启动 CLI Agent
python scripts/run_agent.py

# 启动 FastAPI Web API
python scripts/start_web.py

# 启动 Web 调试前端（另一个终端）
cd frontend
npm run dev
cd ..

# 运行测试
pytest

# 构建 Web 调试前端
cd frontend
npm run build
cd ..
```

## 当前已可用能力

- `search_files`：文件名搜索
- `get_current_candidates`：查看当前双 source candidates 快照
- `list_folder_contents`：展开目录候选
- `read_file_content`：文件内容与基础预览
- `prepare_send` / `confirm_send`：发送前校验、概览展示与确认执行（当前真实外发仍为 mock）
- CLI 调试入口：展示 `[Tool Trace]`
- Web 调试前端：支持 SSE 流式查看 reasoning、assistant、tool、interrupt、usage 与事件时间线

## 当前真实接线

当前仓库的最小可运行主链路是：

```text
CLI: scripts/run_agent.py -> src/listeners/cli.py -> src/agent/runner.py -> src/agent/factory.py -> src/adapters/model_provider.py + src/adapters/mcp_client.py -> src/mcp_servers/file_tools/server.py

Web: scripts/start_web.py -> src/listeners/web.py -> src/listeners/web_event_normalizer.py -> src/agent/factory.py -> src/adapters/model_provider.py + src/adapters/mcp_client.py -> src/mcp_servers/file_tools/server.py
```

注意：微信 / 飞书监听、RAG 主链路、框架级 `interrupt()/resume()` 当前仍未接入真实运行主链路。

## 依赖说明

### Python 依赖（`Weseeker/pyproject.toml`）

- Agent / Graph：`langchain`、`langgraph`
- MCP / 工具链：`mcp`、`langchain-mcp-adapters`
- Web API：`fastapi`、`uvicorn`
- 文件预览：`python-docx`、`python-pptx`、`PyMuPDF`、`openpyxl`、`Pillow`
- 配置与基础设施：`httpx`、`pydantic-settings`、`pyyaml`、`loguru`

### 前端依赖（`Weseeker/frontend/package.json`）

- 运行时：`react`、`react-dom`、`react-markdown`、`remark-gfm`
- 构建：`vite`、`@vitejs/plugin-react`、`tailwindcss`、`postcss`、`autoprefixer`

## 开发说明

当前仓库已从工程骨架进入 MVP 联调阶段。

- `scripts/start_mcp_servers.py` 会先探测 `9100` 端口；若目标 `file_tools` MCP 服务已在运行，会直接给出友好提示，不再抛出难看的端口绑定异常
- `src/listeners/cli.py` 默认展示 `[Tool Trace]`，方便观察 Agent 是否真实触发工具调用
- `src/listeners/web.py` 当前通过 POST + SSE 向前端输出结构化流式事件；`src/listeners/web_event_normalizer.py` 负责把 LangChain 1.2+ 的 v2 stream 归一化成前端语义事件
- `Weseeker/frontend/` 是当前真实的 Web 调试前端工程，不要和根目录 `frontend_design/` 设计稿混淆
- 搜索结果当前已约束优先使用 Markdown 列表展示，必要时允许使用 Markdown 表格；前端支持对应渲染
- 调试前应先检查 MCP 端口是否已开启
- 如果修改了 MCP 相关文件，应先关闭原有端口监听，再重新运行对应服务后继续调试

后续将继续补自动化测试，并推进真实发送通道、`interrupt()/resume()`、RAG 与更多渠道监听能力。
