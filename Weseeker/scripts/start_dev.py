from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
FRONTEND_DIR = ROOT_DIR / "frontend"


@dataclass(frozen=True)
class ManagedProcess:
    name: str
    process: subprocess.Popen


def _build_env(*, rag_enabled: bool) -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(SRC_DIR) if not existing_pythonpath else f"{SRC_DIR};{existing_pythonpath}"
    )
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env["WESEEKER_RAG__ENABLED"] = "true" if rag_enabled else "false"
    return env


def _start_process(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
) -> ManagedProcess:
    print(f"[start_dev] starting {name}: {' '.join(command)}")
    process = subprocess.Popen(command, cwd=cwd, env=env)
    return ManagedProcess(name=name, process=process)


def _resolve_npm_command() -> str:
    npm_command = shutil.which("npm.cmd") or shutil.which("npm")
    if npm_command is None:
        raise FileNotFoundError("未找到 npm，请先安装 Node.js 或确认 npm 已加入 PATH。")
    return npm_command


def _stop_processes(processes: list[ManagedProcess]) -> None:
    for item in reversed(processes):
        if item.process.poll() is not None:
            continue
        print(f"[start_dev] stopping {item.name}...")
        item.process.terminate()

    deadline = time.monotonic() + 8
    for item in reversed(processes):
        if item.process.poll() is not None:
            continue
        timeout = max(0.1, deadline - time.monotonic())
        try:
            item.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[start_dev] killing {item.name}...")
            item.process.kill()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start WeSeeker MCP servers, Web API, and Vite frontend for local dev."
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Do not start rag_tools and set WESEEKER_RAG__ENABLED=false.",
    )
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="Start backend services only; skip Vite frontend.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rag_enabled = not args.no_rag
    env = _build_env(rag_enabled=rag_enabled)
    processes: list[ManagedProcess] = []

    def handle_stop(signum, frame) -> None:
        del signum, frame
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_stop)

    try:
        processes.append(
            _start_process(
                name="file_tools",
                command=[sys.executable, "-m", "mcp_servers.file_tools.server"],
                cwd=ROOT_DIR,
                env=env,
            )
        )
        if rag_enabled:
            processes.append(
                _start_process(
                    name="rag_tools",
                    command=[sys.executable, "-m", "mcp_servers.rag_tools.server"],
                    cwd=ROOT_DIR,
                    env=env,
                )
            )
        processes.append(
            _start_process(
                name="web_api",
                command=[sys.executable, "scripts/start_web.py"],
                cwd=ROOT_DIR,
                env=env,
            )
        )
        if not args.no_frontend:
            npm_command = _resolve_npm_command()
            processes.append(
                _start_process(
                    name="frontend",
                    command=[npm_command, "run", "dev"],
                    cwd=FRONTEND_DIR,
                    env=env,
                )
            )

        print("[start_dev] ready:")
        print("  Web API:  http://127.0.0.1:8787")
        if not args.no_frontend:
            print("  Frontend: http://127.0.0.1:5173")
        print(f"  RAG:      {'enabled' if rag_enabled else 'disabled'}")
        print("[start_dev] press Ctrl+C to stop all child processes.")

        while True:
            failed = [item for item in processes if item.process.poll() not in (None, 0)]
            if failed:
                names = ", ".join(item.name for item in failed)
                raise RuntimeError(f"child process failed: {names}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("[start_dev] received stop signal.")
    finally:
        _stop_processes(processes)


if __name__ == "__main__":
    main()
