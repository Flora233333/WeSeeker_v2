"""WeSeeker Web Listener — 纯 API，前端由 Vite/Nginx 单独提供。

启动:  python -m src.listeners.web
端口:  127.0.0.1:8787
"""
from __future__ import annotations
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import TextContent
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage

from agent.factory import create_weseeker_agent
from agent.runner import AgentRunner
from listeners.web_event_normalizer import WebEventNormalizer
from mcp_servers.file_tools.client_state import ClearClientStateResult


def _now_ms() -> int:
    return int(time.time() * 1000)


def _event(type_: str, **kwargs) -> dict:
    return {"type": type_, "ts": _now_ms(), **kwargs}


def _extract_resume_error_message(tool_result: dict | None) -> str:
    if not isinstance(tool_result, dict):
        return "本轮未实际执行 confirm_send。"

    raw = tool_result.get("raw")
    if isinstance(raw, dict):
        for key in ("message", "user_hint", "operator_hint"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    summary = tool_result.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()

    return "confirm_send 执行失败。"


def _parse_clear_client_state_result(tool_result: object, *, thread_id: str) -> ClearClientStateResult:
    if tool_result is None:
        return ClearClientStateResult(
            client_id=thread_id,
            cleared_candidate_source_count=0,
            cleared_candidate_item_count=0,
            cleared_pending_count=0,
        )

    content = getattr(tool_result, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, TextContent):
                try:
                    payload = json.loads(block.text)
                except json.JSONDecodeError:
                    continue
                return ClearClientStateResult(
                    client_id=payload.get("client_id", thread_id),
                    cleared_candidate_source_count=int(payload.get("cleared_candidate_source_count", 0) or 0),
                    cleared_candidate_item_count=int(payload.get("cleared_candidate_item_count", 0) or 0),
                    cleared_pending_count=int(payload.get("cleared_pending_count", 0) or 0),
                )

    return ClearClientStateResult(
        client_id=thread_id,
        cleared_candidate_source_count=0,
        cleared_candidate_item_count=0,
        cleared_pending_count=0,
    )


@dataclass
class WebThreadState:
    messages: list[BaseMessage] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    agent: object | None = None
    mcp_client: MultiServerMCPClient | None = None


class WebRunner(AgentRunner):
    """在 AgentRunner 基础上加 v2 流式事件接口。CLI 路径完全不动。"""

    def __init__(self) -> None:
        super().__init__()
        self._thread_states: dict[str, WebThreadState] = {}

    def create_thread(self) -> str:
        thread_id = uuid.uuid4().hex
        self._get_thread_state(thread_id)
        return thread_id

    async def initialize(self) -> None:
        """Web 路径按 thread 懒加载 agent runtime，不预创建共享 agent。"""
        self._agent = None
        self._mcp_client = None

    def _get_thread_state(self, thread_id: str) -> WebThreadState:
        state = self._thread_states.get(thread_id)
        if state is None:
            state = WebThreadState()
            self._thread_states[thread_id] = state
        return state

    async def _ensure_thread_runtime(self, thread_state: WebThreadState, thread_id: str) -> WebThreadState:
        if thread_state.agent is not None and thread_state.mcp_client is not None:
            return thread_state

        agent, mcp_client = await create_weseeker_agent(thread_id=thread_id)
        thread_state.agent = agent
        thread_state.mcp_client = mcp_client
        return thread_state

    async def clear_thread(self, thread_id: str) -> ClearClientStateResult:
        thread_state = self._get_thread_state(thread_id)
        async with thread_state.lock:
            if thread_state.mcp_client is None:
                result = ClearClientStateResult(
                    client_id=thread_id,
                    cleared_candidate_source_count=0,
                    cleared_candidate_item_count=0,
                    cleared_pending_count=0,
                )
            else:
                async with thread_state.mcp_client.session("file_tools") as session:
                    tool_result = await session.call_tool("clear_client_state")
                result = _parse_clear_client_state_result(tool_result, thread_id=thread_id)
            thread_state.messages = []
            return result

    @staticmethod
    def _collect_completed_messages(data: object) -> list[BaseMessage]:
        if not isinstance(data, dict):
            return []

        completed: list[BaseMessage] = []
        for source, payload in data.items():
            if source.startswith("__") or not isinstance(payload, dict):
                continue
            for message in payload.get("messages", []) or []:
                if isinstance(message, BaseMessage):
                    completed.append(message)
        return completed

    async def astream_events(self, text: str, thread_id: str) -> AsyncIterator[dict]:
        cfg = {"configurable": {"thread_id": thread_id}}
        thread_state = self._get_thread_state(thread_id)

        async with thread_state.lock:
            thread_state = await self._ensure_thread_runtime(thread_state, thread_id)
            agent = thread_state.agent
            norm = WebEventNormalizer()
            run_id = uuid.uuid4().hex[:12]
            history = list(thread_state.messages)
            current_messages = [*history, HumanMessage(content=text)]
            completed_messages: list[BaseMessage] = []

            yield _event("run_started", run_id=run_id, thread_id=thread_id)
            try:
                async for chunk in agent.astream(
                    {"messages": current_messages},
                    config=cfg,
                    stream_mode=["messages", "updates"],
                    version="v2",
                ):
                    if chunk.get("type") == "updates":
                        completed_messages.extend(
                            self._collect_completed_messages(chunk.get("data"))
                        )
                    for ev in norm.feed(chunk):
                        yield ev
                for ev in norm.flush():
                    yield ev
                thread_state.messages = [*current_messages, *completed_messages]
                yield _event("run_finished", run_id=run_id, reason="ok")
            except Exception as e:
                yield _event("error", code=e.__class__.__name__, message=str(e))
                yield _event("run_finished", run_id=run_id, reason="error")


runner = WebRunner()
app = FastAPI(title="WeSeeker Web API")

# 仅本地源
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost", "http://127.0.0.1",
    ],
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    await runner.initialize() # 创建 AgentRunner 对象


class ChatReq(BaseModel):
    text: str
    thread_id: str | None = None


class ResumeReq(BaseModel):
    thread_id: str
    send_token: str
    approved: bool


class ThreadReq(BaseModel):
    thread_id: str


def _sse(events: AsyncIterator[dict]) -> StreamingResponse:
    async def gen():
        async for ev in events:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _resume_events(
    *,
    thread_id: str,
    send_token: str,
    approved: bool,
) -> AsyncIterator[dict]:
    if approved:
        text = f'我确认发送，请立即调用 confirm_send，参数 send_token="{send_token}"。'
    else:
        text = f'我拒绝发送 send_token="{send_token}"，请放弃此次发送并简短告知用户。'

    confirm_send_seen = False
    confirm_send_result: dict | None = None
    stream_error: str | None = None

    async for event in runner.astream_events(text, thread_id):
        if event.get("type") == "tool_call" and event.get("name") == "confirm_send":
            confirm_send_seen = True

        if event.get("type") == "tool_result" and event.get("name") == "confirm_send":
            confirm_send_seen = True
            confirm_send_result = event

        if event.get("type") == "error" and stream_error is None:
            message = event.get("message")
            if isinstance(message, str) and message.strip():
                stream_error = message.strip()

        yield event

    if stream_error:
        yield _event(
            "interrupt_resolved",
            send_token=send_token,
            approved=approved,
            status="error",
            message=stream_error,
        )
        return

    if not approved:
        if confirm_send_seen:
            yield _event(
                "interrupt_resolved",
                send_token=send_token,
                approved=False,
                status="error",
                message="用户已拒绝，但本轮仍触发了 confirm_send。",
            )
            return

        yield _event(
            "interrupt_resolved",
            send_token=send_token,
            approved=False,
            status="rejected",
            message="已取消本次发送。",
        )
        return

    if isinstance(confirm_send_result, dict):
        if bool(confirm_send_result.get("ok")):
            yield _event(
                "interrupt_resolved",
                send_token=send_token,
                approved=True,
                status="confirmed",
                message="已确认发送并完成 confirm_send。",
            )
            return

        yield _event(
            "interrupt_resolved",
            send_token=send_token,
            approved=True,
            status="error",
            message=_extract_resume_error_message(confirm_send_result),
        )
        return

    yield _event(
        "interrupt_resolved",
        send_token=send_token,
        approved=True,
        status="error",
        message="本轮未实际执行 confirm_send。",
    )


@app.post("/api/chat")
async def chat(req: ChatReq):
    tid = req.thread_id or runner.create_thread()
    return _sse(runner.astream_events(req.text, tid))


@app.post("/api/resume")
async def resume(req: ResumeReq):
    return _sse(
        _resume_events(
            thread_id=req.thread_id,
            send_token=req.send_token,
            approved=req.approved,
        )
    )


@app.post("/api/new_thread")
async def new_thread():
    return {"thread_id": runner.create_thread()}


@app.post("/api/clear_thread")
async def clear_thread(req: ThreadReq):
    result = await runner.clear_thread(req.thread_id)
    return {
        "ok": True,
        "thread_id": req.thread_id,
        "cleared_candidate_source_count": result.cleared_candidate_source_count,
        "cleared_candidate_item_count": result.cleared_candidate_item_count,
        "cleared_pending_count": result.cleared_pending_count,
    }


@app.get("/api/health")
async def health():
    return {"ok": True}


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8787, log_level="info")


if __name__ == "__main__":
    main()
