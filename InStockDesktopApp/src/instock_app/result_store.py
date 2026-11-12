from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


class ResultStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def record_run(self, strategy_key: str, result: dict) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        history_dir = self.root / "history" / strategy_key
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"{timestamp}.json"
        latest_path = self.root / f"{strategy_key}-latest.json"
        content = json.dumps(result, ensure_ascii=False, indent=2)
        history_path.write_text(content, encoding="utf-8")
        temporary = latest_path.with_suffix(".json.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, latest_path)
        return history_path

    def latest(self, strategy_key: str) -> dict | None:
        path = self.root / f"{strategy_key}-latest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
