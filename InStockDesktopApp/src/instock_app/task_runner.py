from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from collections.abc import Callable, Sequence

from .contracts import TaskEvent, TaskResult


EventCallback = Callable[[TaskEvent], None]
DoneCallback = Callable[[TaskResult], None]


class TaskRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._task_id: str | None = None
        self._done_event = threading.Event()
        self._result: TaskResult | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(
        self,
        command: Sequence[str],
        on_event: EventCallback,
        on_done: DoneCallback,
    ) -> str:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("已有任务正在运行")
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            process = subprocess.Popen(
                list(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
                env=env,
            )
            task_id = uuid.uuid4().hex
            self._process = process
            self._task_id = task_id
            self._result = None
            self._done_event.clear()
            self._thread = threading.Thread(
                target=self._read_output,
                args=(process, on_event, on_done),
                daemon=True,
                name=f"instock-task-{task_id[:8]}",
            )
            self._thread.start()
            return task_id

    def _read_output(
        self,
        process: subprocess.Popen[str],
        on_event: EventCallback,
        on_done: DoneCallback,
    ) -> None:
        last_payload: dict[str, object] = {}
        last_message = ""
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            try:
                event = TaskEvent.from_json(line)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                event = TaskEvent(kind="log", message=line)
            if event.kind == "result":
                last_payload = dict(event.payload)
                last_message = event.message
            on_event(event)
        process.stdout.close()
        returncode = process.wait()
        result = TaskResult(returncode=returncode, payload=last_payload, message=last_message)
        with self._lock:
            self._result = result
        self._done_event.set()
        on_done(result)

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            process = self._process
            if task_id != self._task_id or process is None or process.poll() is not None:
                return False
            pid = process.pid
        if os.name == "nt":
            subprocess.run(
                ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            process.terminate()
        return True

    def wait(self, timeout: float | None = None) -> TaskResult:
        if not self._done_event.wait(timeout):
            raise TimeoutError("后台任务未在限定时间内结束")
        assert self._result is not None
        return self._result
