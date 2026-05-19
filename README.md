# WeSeeker

WeSeeker 是一个运行在 Windows PC 上的智能文件管家 Agent。用户通过自然语言描述文件需求，系统在本地完成文件搜索、候选管理、目录展开、内容预览和发送前确认；当前真实入口包括 CLI 和 Web 调试前端。

当前状态：开发中，已完成“文件搜索 -> 候选快照 -> 目录展开 -> 文件预览 -> `prepare_send` / `confirm_send`”MVP 主链路，并已接出 FastAPI + React/Vite 的 Web 调试台。Web 端当前还支持按 `thread_id` 隐式隔离工具态，并可通过 `clear_thread` 同步清理当前线程的消息历史、candidates 与 pending send。RAG 已具备条件启用的 Agent 试用链路，Web 端 RAG 问答已跑通，但默认仍关闭。

## 项目目标

- 使用 LangChain + LangGraph 组织 Agent 主流程
- 使用 FastMCP 提供标准化工具服务
- 当前发送链路先落地 `prepare_send -> 用户确认 -> confirm_send` 两阶段确认；`interrupt()` + `Command(resume=...)` 继续保留为后续增强方向
- 在 Windows 本地环境中提供稳定、可恢复的文件管家能力

## 当前真实状态

- 当前真实可运行入口：CLI 与 Web 调试前端
- 当前真实工具链：`file_tools` MCP Server；RAG 试用时额外启用 `rag_tools` MCP Server
- 当前真实发送链：`prepare_send -> 用户确认 -> confirm_send`
- 当前 Web 状态管理：按 `thread_id` 维护 thread 专属消息、agent、MCP client，并通过隐藏 header `X-WeSeeker-Thread-Id` 隔离工具态
- 当前 `clear_thread` 行为：清空当前 thread 的消息历史，并通过 `file_tools` 的内部维护入口同步清理该 thread 的 candidates / pending send
- 当前 `confirm_send` 仍为 mock 发送，未接入真实外发通道
- 当前 RAG 状态：`search_kb` MCP Tool 已可在 `rag.enabled=true` 时接入 Agent；RAG 结果只作为内容证据，后续预览 / 发送仍需先用 `search_files(source.file_name)` 建立 file_tools candidates
- 当前尚未进入真实主链路：微信监听、飞书监听、框架级 `interrupt()/resume()`；RAG 仍是默认关闭的试用能力

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
# 一键启动本地开发环境（默认启用 RAG）
python scripts/start_dev.py

# 一键启动但关闭 RAG
python scripts/start_dev.py --no-rag

# 只启动后端服务，不启动 Vite 前端
python scripts/start_dev.py --no-frontend

# 启动文件工具 MCP Server
python scripts/start_mcp_servers.py

# 启动 CLI Agent
python scripts/run_agent.py

# 启动 FastAPI Web API
python scripts/start_web.py

# 健康检查
curl http://127.0.0.1:8787/api/health

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

`scripts/start_dev.py` 会在启动任何子进程前检查必要外部服务：

- Everything HTTP：由 `settings.everything.host/port` 指定，默认 `127.0.0.1:8080`
- LM Studio embedding：当 RAG 启用且 `rag.embedding_provider=lmstudio` 时检查 `rag.lmstudio_embedding_base_url` 和 `rag.embedding_model`

上述检查任一失败，脚本会直接报错退出，不会拉起 MCP / Web / 前端子进程，也不会留下端口占用。若目标开发端口 `5173`、`8787`、`9100`、`9200` 已被占用，脚本也会 fail-fast。

## 当前已可用能力

- `search_files`：文件名搜索
- `get_current_candidates`：查看当前双 source candidates 快照
- `list_folder_contents`：展开目录候选
- `read_file_content`：文件内容与基础预览
- `prepare_send` / `confirm_send`：发送前校验、概览展示与确认执行（当前真实外发仍为 mock）
- `search_kb`：RAG 知识库内容检索。仅在 RAG 启用时可用，当前默认知识库为 `test_kb_notes`
- CLI 调试入口：展示 `[Tool Trace]`
- Web 调试前端：支持 SSE 流式查看 reasoning、assistant、tool、interrupt、usage 与事件时间线
- Web thread 隔离：不同 `thread_id` 会使用各自的 agent / MCP client / candidates / pending send bucket
- `clear_thread`：保留当前 `thread_id`，但清空该线程消息历史与远端工具态

## 当前真实接线

当前仓库的最小可运行主链路是：

```text
CLI: scripts/run_agent.py -> src/listeners/cli.py -> src/agent/runner.py -> src/agent/factory.py -> src/adapters/model_provider.py + src/adapters/mcp_client.py -> src/mcp_servers/file_tools/server.py

Web: scripts/start_web.py -> src/listeners/web.py -> src/listeners/web_event_normalizer.py -> src/agent/factory.py -> src/adapters/model_provider.py + src/adapters/mcp_client.py -> src/mcp_servers/file_tools/server.py
```

RAG 试用链路在 `rag.enabled=true` 或 `WESEEKER_RAG__ENABLED=true` 时额外接入：

```text
Agent -> src/adapters/mcp_client.py -> src/mcp_servers/rag_tools/server.py -> search_kb -> Hybrid RAG index
```

注意：微信 / 飞书监听、框架级 `interrupt()/resume()` 当前仍未接入真实运行主链路。RAG 当前是默认关闭的试用能力，不是强制默认链路。

Web 调试前端当前使用的后端接口包括：

- `POST /api/chat`
- `POST /api/resume`
- `POST /api/new_thread`
- `POST /api/clear_thread`
- `GET /api/health`

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
- Web 调试前端的工具态隔离依赖隐藏 header `X-WeSeeker-Thread-Id`；`file_tools` 侧优先用它作为 bucket key，再 fallback 到 `ctx.client_id`
- `clear_thread` 不能在 Web 进程本地直接清 registry；必须通过当前 thread 的 MCP client 调 `clear_client_state`，在远端 `file_tools` 进程里清真实 bucket
- 搜索结果当前已约束优先使用 Markdown 列表展示，必要时允许使用 Markdown 表格；前端支持对应渲染
- 调试前应先检查 MCP 端口是否已开启
- 如果修改了 MCP 相关文件，应先关闭原有端口监听，再重新运行对应服务后继续调试
- RAG 结果不是 file_tools candidates；如果用户基于 RAG 结果要求“第几个预览 / 发送”，Agent 必须先用 `search_files(source.file_name)` 建立正式 candidates，再调用 `read_file_content` 或 `prepare_send`

## 当前限制

- `confirm_send` 当前仍为 mock 发送，不能代表真实外发能力
- 全量 `pytest`、CLI 端到端与 Web 真实联调还未完全收敛
- SQLite checkpointer / `storage/` 仍是设计保留，不是当前主链路事实
- RAG 默认关闭；当前已通过 Web 端问答试用验证，但仍未做默认启用、rerank、HyDE 或独立 Query Rewrite 模块

后续将继续补自动化测试，并推进真实发送通道、`interrupt()/resume()`、RAG 检索增强与更多渠道监听能力。
