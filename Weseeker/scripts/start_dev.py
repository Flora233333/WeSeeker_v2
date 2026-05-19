from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
FRONTEND_DIR = ROOT_DIR / "frontend"
DEPENDENCY_TIMEOUT_SECONDS = 2.0

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEPENDENCY_ERRORS = (
    OSError,
    TimeoutError,
    urllib.error.URLError,
    urllib.error.HTTPError,
    json.JSONDecodeError,
)


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


def _request_json(
    *,
    url: str,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=DEPENDENCY_TIMEOUT_SECONDS) as response:
        response_body = response.read().decode("utf-8", "replace")
    parsed = json.loads(response_body)
    if not isinstance(parsed, dict):
        raise RuntimeError("response is not a JSON object")
    return parsed


def _check_everything(settings) -> None:
    url = f"http://{settings.everything.host}:{settings.everything.port}"
    probe_url = f"{url}/?search=test&json=1&count=1"
    try:
        _request_json(url=probe_url)
    except DEPENDENCY_ERRORS as exc:
        raise RuntimeError(
            "Everything HTTP is not reachable. Start Everything and enable its HTTP server "
            f"at {url}; otherwise search_files will fail. Detail: {exc}"
        ) from exc
    print(f"[start_dev] dependency ok: Everything HTTP {url}")


def _check_lmstudio_embedding(settings) -> None:
    base_url = settings.rag.lmstudio_embedding_base_url.rstrip("/")
    model = settings.rag.embedding_model
    try:
        payload = _request_json(
            url=f"{base_url}/embeddings",
            method="POST",
            payload={"model": model, "input": "ping"},
        )
    except DEPENDENCY_ERRORS as exc:
        raise RuntimeError(
            "LM Studio embedding endpoint is not reachable. Start LM Studio, load the "
            f"embedding model {model}, and expose {base_url}; otherwise search_kb will fail. "
            f"Detail: {exc}"
        ) from exc
    if "data" not in payload:
        raise RuntimeError(
            "LM Studio embedding endpoint responded, but the response does not contain "
            f"embedding data. Check model={model} at {base_url}."
        )
    print(f"[start_dev] dependency ok: LM Studio embeddings {base_url} model={model}")


def _is_port_open(port: int) -> bool:
    for host in ("127.0.0.1", "localhost", "::1"):
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            continue
    return False


def _check_required_ports(settings, *, rag_enabled: bool, frontend_enabled: bool) -> None:
    ports = {
        "file_tools": settings.mcp.file_tools_port,
        "web_api": 8787,
    }
    if rag_enabled:
        ports["rag_tools"] = settings.mcp.rag_tools_port
    if frontend_enabled:
        ports["frontend"] = 5173

    occupied = [
        f"{name}=127.0.0.1:{port}"
        for name, port in ports.items()
        if _is_port_open(port)
    ]
    if occupied:
        raise RuntimeError(
            "Required dev ports are already in use: "
            + ", ".join(occupied)
            + ". Stop those processes before running start_dev.py."
        )


def _check_dependencies(*, rag_enabled: bool, frontend_enabled: bool) -> None:
    from config.settings import get_settings

    settings = get_settings()
    _check_everything(settings)
    if rag_enabled and settings.rag.embedding_provider.lower().strip() == "lmstudio":
        _check_lmstudio_embedding(settings)
    _check_required_ports(settings, rag_enabled=rag_enabled, frontend_enabled=frontend_enabled)


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
        _check_dependencies(rag_enabled=rag_enabled, frontend_enabled=not args.no_frontend)
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
