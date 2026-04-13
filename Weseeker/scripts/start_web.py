# scripts/start_web.py
"""开发环境启动 WeSeeker Web API。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from listeners.web import main

if __name__ == "__main__":
    main()
