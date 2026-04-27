from __future__ import annotations

import json
from pathlib import Path


def load_manifest(manifest_dir: str, kb_name: str) -> dict[str, object]:
    manifest_path = Path(manifest_dir) / f"{kb_name}.json"
    if not manifest_path.exists():
        return {"files": {}, "stats": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_manifest(manifest_dir: str, kb_name: str, payload: dict[str, object]) -> str:
    manifest_path = Path(manifest_dir) / f"{kb_name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path)
