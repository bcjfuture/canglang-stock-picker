from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def emit(
    kind: str,
    message: str,
    progress: int = 0,
    payload: dict[str, Any] | None = None,
) -> None:
    event = {
        "kind": kind,
        "message": message,
        "progress": max(0, min(100, int(progress))),
        "payload": payload or {},
    }
    print(json.dumps(event, ensure_ascii=False), flush=True)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        for attempt in range(len(REPLACE_RETRY_DELAYS) + 1):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == len(REPLACE_RETRY_DELAYS):
                    raise
                time.sleep(REPLACE_RETRY_DELAYS[attempt])
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
