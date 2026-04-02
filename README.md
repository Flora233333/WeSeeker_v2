# WeSeeker

WeSeeker 是一个运行在 Windows PC 上的智能文件管家 Agent。用户通过自然语言描述文件需求，系统在本地完成文件搜索、目录展开、内容预览、知识库检索与安全发送。

当前状态：开发中。

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

## 计划中的常用命令

```bash
# 启动全部 MCP Server
python scripts/start_mcp_servers.py

# 启动 CLI Agent
python scripts/run_agent.py

# 运行测试
pytest
```

## 开发说明

当前仓库已完成工程骨架初始化。

- `Weseeker/src/` 已建立模块占位
- `Weseeker/scripts/` 已建立脚本占位
- `Weseeker/tests/` 已建立测试目录占位，但默认不纳入版本控制

后续将按技术大纲逐步填充最小可运行链路。
