"""消息归一化中间件。

提供两个独立的中间件，按需组合：
- ToolContentNormalizerMiddleware：通用，把 ToolMessage 的 list content 拍平为 str
- DeepSeekReasoningMiddleware：DeepSeek 专用，管理 reasoning_content 的保留与剥离

参考文档：https://api-docs.deepseek.com/guides/thinking_mode#tool-calls
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

# ANSI 终端颜色
_CYAN    = "\033[96m"
_YELLOW  = "\033[93m"
_GREEN   = "\033[92m"
_MAGENTA = "\033[95m"
_DIM     = "\033[2m"
_BOLD    = "\033[1m"
_RESET   = "\033[0m"


# ============================================================
# Debug 打印工具
# ============================================================

def _debug_print_messages(title: str, messages: list[BaseMessage], max_len: int = 100) -> None:
    """打印消息列表的摘要信息，用于调试。"""
    print(f"\n{'='*65}")
    print(f"  {title} ({len(messages)} messages)")
    print(f"{'='*65}")
    for i, msg in enumerate(messages):
        msg_type = msg.__class__.__name__

        # content 预览
        content = msg.content
        if isinstance(content, str):
            content_preview = content.replace("\n", "\\n")[:max_len] if content else "<empty>"
            content_type = "str"
        elif isinstance(content, list):
            content_type = f"list[{len(content)}]"
            try:
                content_preview = json.dumps(content[:2], ensure_ascii=False)[:max_len]
            except Exception:
                content_preview = str(content)[:max_len]
        else:
            content_type = type(content).__name__
            content_preview = str(content)[:max_len]

        # additional_kwargs 关键信息
        extra = getattr(msg, "additional_kwargs", {}) or {}
        has_reasoning = "reasoning_content" in extra
        reasoning_preview = ""
        if has_reasoning:
            rc = extra["reasoning_content"]
            if rc:
                reasoning_preview = str(rc).replace("\n", "\\n")[:80]
            else:
                reasoning_preview = "<empty>"

        # tool_calls（AIMessage 发起的调用）
        tool_calls = getattr(msg, "tool_calls", []) or []
        tool_calls_count = len(tool_calls)

        # reasoning tokens
        usage = getattr(msg, "usage_metadata", {}) or {}
        reasoning_tokens = ""
        output_details = usage.get("output_token_details") or {}
        if isinstance(output_details, dict) and output_details.get("reasoning"):
            reasoning_tokens = f"  r_tokens={output_details['reasoning']}"

        # --- 主行 ---
        rc_flag = "🧠" if has_reasoning else "  "
        print(
            f"  [{i:2d}] {rc_flag} {msg_type:<14s} "
            f"content={content_type:<9s} "
            f"tc={tool_calls_count}"
            f"{reasoning_tokens}"
        )

        # --- content 行（始终显示）---
        print(f"        content  : {content_preview!r}")

        # --- reasoning 行 ---
        if reasoning_preview:
            print(f"        {_BOLD}{_YELLOW}reasoning : {reasoning_preview!r}{_RESET}")

        # --- AIMessage 的 tool_calls：显示每个调用 ---
        if tool_calls_count > 0:
            for tc in tool_calls:
                tc_name = tc.get("name", "?")
                tc_args = tc.get("args", {})
                try:
                    args_str = json.dumps(tc_args, ensure_ascii=False)
                    if len(args_str) > 80:
                        args_str = args_str[:80] + "..."
                except Exception:
                    args_str = str(tc_args)[:80]
                print(f"        {_BOLD}{_CYAN}🔧 {tc_name}({args_str}){_RESET}")

        # --- ToolMessage：显示 工具名(tool_call_id) ---
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None) or "?"
            tool_call_id = getattr(msg, "tool_call_id", None) or "?"
            # tool_call_id 只显示前 8 位
            short_id = tool_call_id[:8] if len(tool_call_id) > 8 else tool_call_id
            print(f"        {_GREEN}📨 {tool_name}(call_id={short_id}...){_RESET}")

    print(f"{'='*65}\n")


def _debug_diff(label: str, before: list[BaseMessage], after: list[BaseMessage], max_len: int = 80) -> None:
    """对比 BEFORE/AFTER，reasoning 被剥掉的标 ★ STRIPPED，保留的标 ✓ KEPT。"""
    _RED = "\033[91m"
    print(f"\n{'='*65}")
    print(f"  {label} DIFF ({len(before)} messages)")
    print(f"{'='*65}")
    for i in range(max(len(before), len(after))):
        b = before[i] if i < len(before) else None
        a = after[i] if i < len(after) else None
        msg = a or b

        cls_name = msg.__class__.__name__
        b_extra = (getattr(b, "additional_kwargs", {}) or {}) if b else {}
        a_extra = (getattr(a, "additional_kwargs", {}) or {}) if a else {}
        b_has_rc = "reasoning_content" in b_extra
        a_has_rc = "reasoning_content" in a_extra

        # 判断 reasoning 变化
        if b_has_rc and not a_has_rc:
            tag = f"{_RED}{_BOLD}★ STRIPPED{_RESET}"
        elif b_has_rc and a_has_rc:
            tag = f"{_GREEN}✓ KEPT{_RESET}"
        else:
            tag = ""

        tc = len(getattr(msg, "tool_calls", []) or [])
        rc_flag = "🧠" if a_has_rc else ("  " if not b_has_rc else "💀")

        print(f"  [{i:2d}] {rc_flag} {cls_name:<14s} tc={tc}  {tag}")

    print(f"{'='*65}\n")


# ============================================================
# 通用中间件：ToolMessage list content → str
# ============================================================

class ToolContentNormalizerMiddleware(AgentMiddleware):
    """把 ToolMessage 的 list 形态 content 拍平为 str。

    部分 MCP Server 返回的 ToolMessage.content 是 list[dict]，
    而很多 LLM API（DeepSeek、Kimi、GLM 等）要求 content 必须是 str。
    本中间件在每次 model 调用前统一处理，与具体模型无关。
    """

    def __init__(self, *, debug: bool = False) -> None:
        super().__init__()
        self._debug = debug

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        if self._debug:
            _debug_print_messages("[ToolNorm] BEFORE", request.messages)
        cleaned = [self._flatten(m) for m in request.messages]
        if self._debug:
            _debug_print_messages("[ToolNorm] AFTER", cleaned)
        return handler(request.override(messages=cleaned))

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        if self._debug:
            _debug_print_messages("[ToolNorm] BEFORE", request.messages)
        cleaned = [self._flatten(m) for m in request.messages]
        if self._debug:
            _debug_print_messages("[ToolNorm] AFTER", cleaned)
        return await handler(request.override(messages=cleaned))

    @staticmethod
    def _flatten(msg: BaseMessage) -> BaseMessage:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
            text = "\n".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b)
                for b in msg.content
            )
            return ToolMessage(
                content=text,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )
        return msg


# ============================================================
# DeepSeek 专用中间件：reasoning_content 管理
# ============================================================

class DeepSeekReasoningMiddleware(AgentMiddleware):
    """管理 DeepSeek 推理模型的 reasoning_content，严格遵循官方文档规则。

    官方规则（https://api-docs.deepseek.com/guides/thinking_mode#tool-calls）：
    1. 同一轮用户提问内的 tool call 子回合：
       必须把 reasoning_content 回传给 API，让模型继续推理。
    2. 新一轮用户提问开始时：
       必须清除之前所有 assistant 消息的 reasoning_content。

    存储策略：
    - reasoning_content 始终保存在 AIMessage.additional_kwargs["reasoning_content"]
    - state 中的消息始终保留完整的 reasoning_content（用于 UI 展示）
    - 发给 API 时，根据上述规则动态决定保留或剥离
    """

    def __init__(self, *, debug: bool = False) -> None:
        super().__init__()
        self._debug = debug

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        if self._debug:
            _debug_print_messages("[DSReasoning] BEFORE", request.messages)
        cleaned = self._prepare_for_api(request.messages)
        if self._debug:
            _debug_diff("[DSReasoning]", request.messages, cleaned)
        return handler(request.override(messages=cleaned))

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        if self._debug:
            _debug_print_messages("[DSReasoning] BEFORE", request.messages)
        cleaned = self._prepare_for_api(request.messages)
        if self._debug:
            _debug_diff("[DSReasoning]", request.messages, cleaned)
        return await handler(request.override(messages=cleaned))

    @classmethod
    def _prepare_for_api(cls, messages: list[BaseMessage]) -> list[BaseMessage]:
        """根据 DeepSeek 官方规则，处理每条消息的 reasoning_content。"""

        # 找到最后一条 HumanMessage 的位置 → 「当前轮」的起点
        last_human_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break

        result: list[BaseMessage] = []
        for i, msg in enumerate(messages):
            # 兜底：AIMessage list content 拍平
            msg = cls._normalize_ai_content(msg)

            if isinstance(msg, AIMessage):
                if i < last_human_idx:
                    # 旧轮 → 剥掉 reasoning_content
                    msg = cls._strip_reasoning(msg)
                # 当前轮 → 保留 reasoning_content（同轮 tool call 子回合必须回传）

            result.append(msg)
        return result

    @staticmethod
    def _normalize_ai_content(msg: BaseMessage) -> BaseMessage:
        """兜底：如果 AIMessage.content 是 list，拍平并抽出 reasoning。"""
        if not isinstance(msg, AIMessage) or not isinstance(msg.content, list):
            return msg

        text_parts: list[str] = []
        reasoning_parts: list[str] = []

        for block in msg.content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            btype = block.get("type", "")
            if btype in ("text", "output_text"):
                text_parts.append(block.get("text", ""))
            elif btype in ("reasoning", "thinking", "reasoning_content"):
                reasoning_parts.append(
                    block.get("text")
                    or block.get("reasoning")
                    or block.get("thinking")
                    or ""
                )

        extra = dict(msg.additional_kwargs or {})
        if reasoning_parts:
            existing = extra.get("reasoning_content", "")
            extra["reasoning_content"] = "\n".join(
                p for p in [existing, *reasoning_parts] if p
            )

        return AIMessage(
            content="\n".join(p for p in text_parts if p),
            tool_calls=msg.tool_calls,
            additional_kwargs=extra,
            response_metadata=msg.response_metadata,
            id=msg.id,
            name=msg.name,
        )

    @staticmethod
    def _strip_reasoning(msg: AIMessage) -> AIMessage:
        """从 additional_kwargs 中移除 reasoning_content。"""
        extra = msg.additional_kwargs
        if not extra or "reasoning_content" not in extra:
            return msg

        clean_extra = {k: v for k, v in extra.items() if k != "reasoning_content"}
        return AIMessage(
            content=msg.content,
            tool_calls=msg.tool_calls,
            additional_kwargs=clean_extra,
            response_metadata=msg.response_metadata,
            id=msg.id,
            name=msg.name,
        )

