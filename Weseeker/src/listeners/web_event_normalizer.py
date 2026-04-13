"""把 LangChain 1.2+ create_agent 的 v2 stream chunks 翻译成前端事件。

依赖：langchain >= 1.2, langgraph >= 1.1
用法：
    norm = WebEventNormalizer()
    async for chunk in agent.astream(inp, config=cfg,
                                     stream_mode=["messages","updates"],
                                     version="v2"):
        for ev in norm.feed(chunk):
            yield ev
    for ev in norm.flush():
        yield ev
"""
from __future__ import annotations
import json, time, uuid
from typing import Any, Iterator
from langchain_core.messages import AIMessageChunk, ToolMessage


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ev(type_: str, **kw) -> dict:
    return {"type": type_, "ts": _now_ms(), **kw}


class WebEventNormalizer:
    """状态机：聚合 token 流 → 离散语义事件。"""

    def __init__(self) -> None:
        # 当前正在累积的 AI 消息（按 message id 隔离）
        self._cur_msg_id: str | None = None
        self._cur_text: str = ""
        self._cur_reason: str = ""
        # tool_calls 的增量聚合：index -> {id, name, args_str}
        self._partial_tcs: dict[int, dict] = {}
        # 已 emit 过 tool_call 的 id 集合，避免重复
        self._emitted_tc_ids: set[str] = set()

    # ----- 主入口 -----
    def feed(self, chunk: dict) -> Iterator[dict]:
        ctype = chunk.get("type")
        data = chunk.get("data")
        if ctype == "messages":
            yield from self._on_messages(data)
        elif ctype == "updates":
            yield from self._on_updates(data)

    def flush(self) -> Iterator[dict]:
        if self._cur_msg_id and (self._cur_text or self._cur_reason):
            yield _ev("message_end", message_id=self._cur_msg_id)
        self._reset_cur()

    # ----- messages 模式：token 流 -----
    def _on_messages(self, data: Any) -> Iterator[dict]:
        token, metadata = data  # (BaseMessageChunk, dict)
        node = (metadata or {}).get("langgraph_node")

        # ToolMessage 出现 → 工具结果
        if isinstance(token, ToolMessage):
            yield from self._emit_tool_result(token)
            return

        if not isinstance(token, AIMessageChunk):
            return

        # 新消息开始
        if token.id and token.id != self._cur_msg_id:
            if self._cur_msg_id:
                yield _ev("message_end", message_id=self._cur_msg_id)
            self._reset_cur()
            self._cur_msg_id = token.id
            yield _ev("message_start", message_id=token.id, node=node)

        # 拆 content_blocks（LangChain 1.2 统一格式）
        # 注意：_split_blocks 返回的是「当前 chunk 内出现的 text/reason」，
        # 但 LangChain 不同 provider 可能发增量(delta) 也可能发累积(cumulative)
        text_in_chunk, reason_in_chunk = self._split_blocks(token)

        # 兜底：旧版 additional_kwargs.reasoning_content（通常是累积型）
        extra = getattr(token, "additional_kwargs", {}) or {}
        if not reason_in_chunk and isinstance(extra.get("reasoning_content"), str):
            reason_in_chunk = extra["reasoning_content"]

        # 统一用 suffix-prefix overlap 算出真正的 delta
        reason_delta = self._diff(self._cur_reason, reason_in_chunk)
        if reason_delta:
            self._cur_reason += reason_delta
            yield _ev("reasoning_delta", message_id=self._cur_msg_id, text=reason_delta)

        text_delta = self._diff(self._cur_text, text_in_chunk)
        if text_delta:
            self._cur_text += text_delta
            yield _ev("assistant_delta", message_id=self._cur_msg_id, text=text_delta)

        # tool_calls 增量聚合
        for tcc in (getattr(token, "tool_call_chunks", []) or []):
            idx = tcc.get("index", 0)
            slot = self._partial_tcs.setdefault(idx, {"id": None, "name": "", "args": ""})
            if tcc.get("id"):
                slot["id"] = tcc["id"]
            if tcc.get("name"):
                slot["name"] += tcc["name"]
            if tcc.get("args"):
                slot["args"] += tcc["args"]

        # 消息结束 → 把已聚合完整的 tool_call emit 出去
        if getattr(token, "chunk_position", None) == "last":
            for slot in self._partial_tcs.values():
                if slot["id"] and slot["id"] not in self._emitted_tc_ids:
                    try:
                        args = json.loads(slot["args"]) if slot["args"] else {}
                    except Exception:
                        args = {"_raw": slot["args"]}
                    self._emitted_tc_ids.add(slot["id"])
                    yield _ev("tool_call",
                              call_id=slot["id"],
                              name=slot["name"],
                              args=args)
            self._partial_tcs.clear()
            # usage
            usage = getattr(token, "usage_metadata", None)
            if usage:
                od = usage.get("output_token_details") or {}
                yield _ev("usage",
                          input=usage.get("input_tokens", 0),
                          output=usage.get("output_tokens", 0),
                          reasoning=od.get("reasoning", 0))
            yield _ev("message_end", message_id=self._cur_msg_id)
            self._reset_cur()

    # ----- updates 模式：节点边界（兜底捕获 ToolMessage）-----
    def _on_updates(self, data: Any) -> Iterator[dict]:
        if not isinstance(data, dict):
            return
        for node, payload in data.items():
            if not isinstance(payload, dict):
                continue
            for msg in payload.get("messages", []) or []:
                if isinstance(msg, ToolMessage):
                    # messages 流可能已经 emit 过；用 call_id 去重
                    yield from self._emit_tool_result(msg)

    # ----- 工具结果 + interrupt 合成 -----
    _emitted_tool_results: set[str] = set()  # 类级别也行；这里用实例

    def _emit_tool_result(self, msg: ToolMessage) -> Iterator[dict]:
        call_id = getattr(msg, "tool_call_id", None) or ""
        if call_id in getattr(self, "_seen_tr", set()):
            return
        if not hasattr(self, "_seen_tr"):
            self._seen_tr = set()
        self._seen_tr.add(call_id)

        name = getattr(msg, "name", "") or "?"
        # ToolMessage.content 在 LangChain 1.2 可能是：
        #   - str: 直接 JSON 字符串
        #   - list[dict]: content_blocks 格式 [{type:"text", text:"..."}]
        raw_content = msg.content
        if isinstance(raw_content, list):
            # 抽出所有 text 块拼起来
            parts = []
            for blk in raw_content:
                if isinstance(blk, dict):
                    if blk.get("type") == "text":
                        parts.append(blk.get("text", "") or "")
                    elif "text" in blk:
                        parts.append(blk["text"])
                elif isinstance(blk, str):
                    parts.append(blk)
            raw = "".join(parts)
        elif isinstance(raw_content, str):
            raw = raw_content
        else:
            raw = str(raw_content)

        parsed: Any
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"_raw": raw}
        ok = bool(isinstance(parsed, dict) and parsed.get("ok", True))
        summary = self._summarize(name, parsed)
        yield _ev("tool_result",
                  call_id=call_id, name=name, ok=ok,
                  summary=summary, raw=parsed)

        # 逻辑层中断：prepare_send 成功 → 合成 interrupt
        if name == "prepare_send" and ok and isinstance(parsed, dict):
            send_token = parsed.get("send_token")
            files = parsed.get("files") or []
            if send_token:
                yield _ev("interrupt",
                          action="prepare_send",
                          send_token=send_token,
                          files=files,
                          expires_in=300)

    @staticmethod
    def _summarize(name: str, parsed: Any) -> str:
        if not isinstance(parsed, dict):
            return ""
        if name == "search_files":
            return f"count={parsed.get('count', 0)}"
        if name == "list_folder_contents":
            return f"items={parsed.get('count', 0)}"
        if name == "read_file_content":
            md = parsed.get("metadata") or {}
            return f"{parsed.get('file_type', '')} · {md.get('size_display', '')}"
        if name == "prepare_send":
            return f"files={parsed.get('file_count', 0)}"
        if name == "confirm_send":
            return f"sent={parsed.get('file_count', 0)}"
        return "ok" if parsed.get("ok") else "fail"

    # ----- helpers -----
    @staticmethod
    def _diff(prev: str, incoming: str) -> str:
        """计算 incoming 真正新增的部分，处理混合 cumulative/incremental chunk 模式。

        算法：找 prev 的最长后缀，使其同时是 incoming 的前缀（重叠区），
        然后返回 incoming 去掉重叠区后的部分。

        这能同时处理三种 chunk 模式，且能在它们之间任意切换：
          - 累积型 (cumulative)：incoming 完全包含 prev → 重叠 = 整个 prev
          - 增量型 (incremental)：incoming 与 prev 无重叠 → 重叠 = ""
          - 滑窗型 (sliding overlap)：incoming 以 prev 的最后几字符开头 → 重叠 = 那几字符
        """
        if not incoming:
            return ""
        if not prev:
            return incoming
        # 限制最大重叠长度，保护性能（中文 token 通常 1-3 字，留 200 字符余量足够）
        max_k = min(len(prev), len(incoming), 200)
        for k in range(max_k, 0, -1):
            if prev.endswith(incoming[:k]):
                return incoming[k:]
        return incoming

    @classmethod
    def _merge_stream_segments(cls, segments: list[str]) -> str:
        """把同一 chunk 内的多段文本按重叠关系合并成单段。

        LangChain 1.2+ 在不同 provider 下，单个 chunk 里可能同时出现：
        - 增量片段
        - 累积片段
        - 带滑窗重叠的片段

        直接 `"".join(...)` 会把这些片段重复拼接，前端就会看到
        "找到了找到了 7 7 个" 这类重复文本。这里沿用同一套 overlap
        逻辑，把 chunk 内部先合并成「当前最新文本」。
        """
        merged = ""
        for segment in segments:
            if not segment:
                continue
            merged += cls._diff(merged, segment)
        return merged

    @staticmethod
    def _split_blocks(token: AIMessageChunk) -> tuple[str, str]:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        blocks = getattr(token, "content_blocks", None)
        if blocks:
            for b in blocks:
                t = b.get("type") if isinstance(b, dict) else None
                if t == "text":
                    text_parts.append(b.get("text", "") or "")
                elif t in ("reasoning", "thinking"):
                    reasoning_parts.append(
                        b.get("text") or b.get("reasoning") or b.get("thinking") or ""
                    )
            return (
                WebEventNormalizer._merge_stream_segments(text_parts),
                WebEventNormalizer._merge_stream_segments(reasoning_parts),
            )
        # 退化：content 是 str 或 list
        c = token.content
        if isinstance(c, str):
            return c, ""
        if isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict):
                    if blk.get("type") in ("reasoning_content", "reasoning", "thinking"):
                        reasoning_parts.append(
                            blk.get("text") or blk.get("reasoning") or blk.get("thinking") or ""
                        )
                    else:
                        text_parts.append(blk.get("text", "") or "")
                elif isinstance(blk, str):
                    text_parts.append(blk)
        return (
            WebEventNormalizer._merge_stream_segments(text_parts),
            WebEventNormalizer._merge_stream_segments(reasoning_parts),
        )

    def _reset_cur(self) -> None:
        self._cur_msg_id = None
        self._cur_text = ""
        self._cur_reason = ""
