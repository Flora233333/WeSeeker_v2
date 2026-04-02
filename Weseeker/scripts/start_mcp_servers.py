from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"


def main() -> None:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) if not existing else f"{SRC_DIR};{existing}"

    command = [sys.executable, "-m", "mcp_servers.file_tools.server"]
    print("启动 file_tools MCP Server: http://127.0.0.1:9100/mcp")
    subprocess.run(command, check=True, cwd=ROOT_DIR, env=env)


if __name__ == "__main__":
    main()
