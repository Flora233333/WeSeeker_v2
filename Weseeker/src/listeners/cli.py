from __future__ import annotations

import asyncio
import sys

from loguru import logger

from agent.runner import AgentResponse, AgentRunner

'''
- 这程序本质上是“单线程异步程序为主”
- 不是传统那种多线程并发程序
- 只有 input() 被显式丢到了一个工作线程里
- await 不是“开线程”，而是“暂停这里，等结果回来再继续”
'''


def configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def print_welcome() -> None:
    print(
        """
==============================
WeSeeker MVP CLI
- 输入 quit / exit 退出
- 输入 clear 清空会话
- 每轮会显示工具调用过程
==============================
""".strip()
    )


def print_tool_trace(response: AgentResponse) -> None:
    print("\n-----")
    if not response.tool_traces:
        print("[Tool Trace] 本轮未触发工具调用")
        print("-----")
        return

    print("[Tool Trace]")
    for index, trace in enumerate(response.tool_traces, start=1):
        print(f"{index}. tool={trace.tool_name}")
        print(f"   args={trace.args}")
        preview_lines = trace.result_preview.splitlines() or [trace.result_preview]
        if preview_lines:
            print(f"   result={preview_lines[0]}")
            for line in preview_lines[1:]:
                print(f"          {line}")
    print("-----")


async def main() -> None:
    configure_stdio()
    print_welcome()
    runner = AgentRunner()
    await runner.initialize()
    logger.info("WeSeeker Agent 初始化完成")

    try:
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n你: ") # 把一个同步、阻塞的函数丢到线程里执行，然后在异步代码里 await 它的结果
            except EOFError:
                print("输入流已结束，退出程序。")
                break
            user_input = user_input.strip()

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q", "退出"}:
                print("再见。")
                break
            if user_input.lower() in {"clear", "清空", "cls"}:
                await runner.new_conversation()
                print("会话已清空。")
                continue

            response = await runner.process_message_with_trace(user_input)
            print_tool_trace(response)
            print(f"\nWeSeeker: {response.reply}")
    finally:
        await runner.cleanup()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
