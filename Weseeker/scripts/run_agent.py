from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from listeners.cli import configure_stdio, run


if __name__ == "__main__":
    configure_stdio()
    run()
