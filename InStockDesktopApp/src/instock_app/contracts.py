from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskEvent:
    kind: str
    message: str
    progress: int = 0
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, line: str) -> "TaskEvent":
        data = json.loads(line)
        return cls(
            kind=str(data["kind"]),
            message=str(data.get("message", "")),
            progress=max(0, min(100, int(data.get("progress", 0)))),
            payload=dict(data.get("payload", {})),
        )


@dataclass(frozen=True)
class StrategySpec:
    key: str
    title: str
    config_name: str
    result_name: str
    top_sheet: str


@dataclass(frozen=True)
class TaskResult:
    returncode: int
    payload: dict[str, Any] = field(default_factory=dict)
    message: str = ""
