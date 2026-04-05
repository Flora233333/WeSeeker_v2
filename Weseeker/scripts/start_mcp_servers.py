from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.settings import get_settings
from loguru import logger


SERVER_NAME = "weseeker-file-tools"
MCP_PATH = "/mcp"


def _build_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) if not existing else f"{SRC_DIR};{existing}"
    return env


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _probe_mcp_server(host: str, port: int) -> tuple[bool, str]:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "weseeker-startup-probe", "version": "0.1.0"},
        },
    }
    request = urllib.request.Request(
        url=f"http://{host}:{port}{MCP_PATH}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            body = response.read().decode("utf-8", "replace").strip()
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, str(exc.reason) or "连接失败"
    except TimeoutError:
        return False, "探测超时"

    if not body:
        return False, "空响应"

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for line in lines:
        candidate = line[5:].strip() if line.startswith("data:") else line
        try:
            response_payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        result = response_payload.get("result") if isinstance(response_payload, dict) else None
        if not isinstance(result, dict):
            continue
        server_info = result.get("serverInfo")
        if isinstance(server_info, dict) and server_info.get("name") == SERVER_NAME:
            return True, "matched"

    return False, "响应不是目标 MCP 服务"


def _log_starting(endpoint: str) -> None:
    logger.info("[{}] status=starting endpoint={}", SERVER_NAME, endpoint)


def _log_running(endpoint: str) -> None:
    logger.info("[{}] status=already_running endpoint={}", SERVER_NAME, endpoint)


def _exit_port_conflict(port: int, detail: str) -> None:
    logger.error("[{}] status=port_conflict port={} detail={}", SERVER_NAME, port, detail)
    raise SystemExit(1)


def main() -> None:
    settings = get_settings()
    host = "127.0.0.1"
    port = settings.mcp.file_tools_port
    endpoint = f"http://{host}:{port}{MCP_PATH}"

    if _is_port_open(host, port):
        matched, detail = _probe_mcp_server(host, port)
        if matched:
            _log_running(endpoint)
            return
        _exit_port_conflict(port, detail)

    env = _build_env()
    command = [sys.executable, "-m", "mcp_servers.file_tools.server"]
    _log_starting(endpoint)

    try:
        subprocess.run(command, check=True, cwd=ROOT_DIR, env=env)
    except subprocess.CalledProcessError:
        if _is_port_open(host, port):
            matched, detail = _probe_mcp_server(host, port)
            if matched:
                _log_running(endpoint)
                return
            _exit_port_conflict(port, detail)
        raise


if __name__ == "__main__":
    main()
