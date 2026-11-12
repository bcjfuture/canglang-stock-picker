from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from .contracts import TaskEvent, TaskResult
from .consensus import build_consensus
from .data_status import assess_data_health, latest_complete_data_date, latest_table_dates
from .result_store import ResultStore
from .runtime import InStockRuntime
from .settings import AppPaths, STRATEGIES
from .task_runner import TaskRunner


@dataclass
class ControllerState:
    actions_enabled: bool = True
    current_task_id: str | None = None
    current_kind: str | None = None
    current_strategy: str | None = None
    last_event: TaskEvent | None = None
    last_completed_kind: str | None = None
    current_automatic: bool = False
    last_completed_automatic: bool = False


class AppController:
    def __init__(
        self,
        paths: AppPaths,
        runner: TaskRunner,
        store: ResultStore,
        runtime: InStockRuntime,
    ) -> None:
        self.paths = paths
        self.runner = runner
        self.store = store
        self.runtime = runtime
        self.state = ControllerState()
        self._observer: Callable[[TaskEvent], None] | None = None
        self._done_observer: Callable[[TaskResult], None] | None = None
        self._pending_tasks: list[tuple[str, str | None]] = []
        self._cancel_requested = False

    def set_observers(
        self,
        event_observer: Callable[[TaskEvent], None],
        done_observer: Callable[[TaskResult], None],
    ) -> None:
        self._observer = event_observer
        self._done_observer = done_observer

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _worker_path(self) -> Path:
        installed = self.paths.workers_dir / "worker_cli.py"
        return installed if installed.exists() else self._project_root() / "workers" / "worker_cli.py"

    def _start(
        self,
        kind: str,
        arguments: list[str],
        strategy: str | None = None,
        *,
        automatic: bool = False,
    ) -> str:
        internal_handoff = automatic and self.state.current_kind is None
        if not self.state.actions_enabled and not internal_handoff:
            raise RuntimeError("已有任务正在运行")
        if automatic and self._cancel_requested:
            raise RuntimeError("自动流程已取消")
        if not automatic:
            self._cancel_requested = False
        command = [str(self.paths.python_exe), "-X", "utf8", str(self._worker_path()), *arguments]
        self.state.actions_enabled = False
        self.state.current_kind = kind
        self.state.current_strategy = strategy
        self.state.current_automatic = automatic
        try:
            task_id = self.runner.start(command, self.on_event, self.on_done)
        except Exception:
            self.state.actions_enabled = True
            self.state.current_kind = None
            self.state.current_strategy = None
            self.state.current_automatic = False
            raise
        self.state.current_task_id = task_id
        if self._cancel_requested:
            self.runner.cancel(task_id)
        return task_id

    def start_update(self) -> str:
        if not self.state.actions_enabled:
            raise RuntimeError("已有任务正在运行")
        self.runtime.start_database()
        result = self.paths.data_dir / "update-latest.json"
        return self._start("update", ["update", "--result", str(result)])

    def run_picker(self, strategy_key: str, *, automatic: bool = False) -> str:
        if strategy_key not in STRATEGIES:
            raise ValueError(f"未知策略：{strategy_key}")
        result = self.paths.data_dir / "results" / STRATEGIES[strategy_key].result_name
        return self._start(
            "picker",
            ["pick", "--strategy", strategy_key, "--result", str(result)],
            strategy_key,
            automatic=automatic,
        )

    def load_picker_result(self, strategy_key: str) -> dict | None:
        if strategy_key not in STRATEGIES:
            raise ValueError(f"未知策略：{strategy_key}")
        return self.store.latest(strategy_key)

    def load_five_day_consensus(self) -> dict:
        return self.load_consensus(5)

    def load_consensus(self, day_limit: int) -> dict:
        return build_consensus(
            self.store.root / "history",
            (key for key in STRATEGIES if key != "all_market"),
            day_limit=day_limit,
        )

    def run_backtest(self, refresh: bool = False, *, automatic: bool = False) -> str:
        result = self.paths.data_dir / "recent-backtest-latest.json"
        arguments = ["recent-backtest", "--result", str(result)]
        return self._start("backtest", arguments, automatic=automatic)

    def start_sector_forecast(
        self, refresh_history: bool = False, *, automatic: bool = False
    ) -> str:
        result = self.paths.data_dir / "sector-forecast-latest.json"
        arguments = ["sector-forecast", "--result", str(result)]
        if refresh_history:
            arguments.append("--refresh-history")
        return self._start("sector_forecast", arguments, automatic=automatic)

    def export_latest(self, strategy_key: str, output_dir: Path) -> str:
        source = self.paths.data_dir / "results" / STRATEGIES[strategy_key].result_name
        if not source.exists():
            raise FileNotFoundError("请先运行该策略选股")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = output_dir / f"沧浪之水_{STRATEGIES[strategy_key].title}_{timestamp}.xlsx"
        return self._start("export", ["export", "--input", str(source), "--output", str(output)])

    def on_event(self, event: TaskEvent) -> None:
        if self.state.current_automatic:
            payload = dict(event.payload)
            payload["automatic"] = True
            if self.state.current_kind == "picker":
                key = self.state.current_strategy
                title = STRATEGIES[key].title
                number = list(STRATEGIES).index(key) + 1
                event = TaskEvent(event.kind, f"自动选股 {number}/{len(STRATEGIES)} · {title} · {event.message}", event.progress, payload)
            else:
                event = TaskEvent(event.kind, event.message, event.progress, payload)
        if self.state.current_kind == "sector_forecast":
            payload = dict(event.payload)
            payload["automatic"] = self.state.current_automatic
            payload["scope"] = "sector_forecast"
            event = TaskEvent(event.kind, event.message, event.progress, payload)
        self.state.last_event = event
        if self._observer:
            self._observer(event)

    def on_done(self, result: TaskResult) -> None:
        strategy = self.state.current_strategy
        completed_kind = self.state.current_kind
        completed_automatic = self.state.current_automatic
        self.state.last_completed_kind = completed_kind
        self.state.last_completed_automatic = completed_automatic
        if (
            result.returncode == 0
            and strategy
            and isinstance(result.payload.get("rows"), list)
        ):
            try:
                self.store.record_run(strategy, result.payload)
            except Exception as exc:
                result = TaskResult(1, result.payload, f"选股结果保存失败：{exc}")
                self.on_event(TaskEvent("error", result.message, 100))
        if result.returncode != 0 or self._cancel_requested:
            self._pending_tasks.clear()
        elif completed_kind == "update":
            self._pending_tasks = [("picker", key) for key in STRATEGIES]
            self._pending_tasks.extend([("backtest", None), ("sector_forecast", None)])
        self.state.actions_enabled = not bool(self._pending_tasks)
        self.state.current_task_id = None
        self.state.current_kind = None
        self.state.current_strategy = None
        self.state.current_automatic = False
        if self._done_observer:
            self._done_observer(TaskResult(result.returncode, {
                **result.payload,
                "_completed_kind": completed_kind,
                "_automatic": completed_automatic,
                "_pipeline_pending": bool(self._pending_tasks),
                "_cancelled": self._cancel_requested,
            }, result.message))
        if self._pending_tasks:
            kind, key = self._pending_tasks.pop(0)
            try:
                if kind == "picker":
                    self.run_picker(key, automatic=True)
                elif kind == "backtest":
                    self.run_backtest(automatic=True)
                else:
                    self.start_sector_forecast(automatic=True)
            except Exception as exc:
                self._pending_tasks.clear()
                self.state.actions_enabled = True
                self.on_event(
                    TaskEvent(
                        kind="error",
                        message=f"自动流程已停止，后续任务启动失败：{exc}",
                        progress=100,
                        payload={"automatic": True},
                    )
                )
        elif self.state.current_kind is None:
            self.state.actions_enabled = True

    def cancel(self) -> bool:
        self._cancel_requested = True
        self._pending_tasks.clear()
        task_id = self.state.current_task_id
        return bool(task_id and self.runner.cancel(task_id))

    def start_services(self) -> int:
        return self.runtime.start_database()

    def stop_services(self) -> list[int]:
        return self.runtime.stop_instock()

    def load_backtest(self) -> dict | None:
        path = self.paths.data_dir / "recent-backtest-latest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_sector_forecast(self) -> dict | None:
        path = self.paths.data_dir / "sector-forecast-latest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def data_health(self, db_running: bool) -> dict[str, str]:
        if not db_running:
            return {"label": "需检查", "detail": "数据库未运行", "tone": "danger"}
        try:
            return assess_data_health(latest_table_dates(), date.today())
        except Exception as exc:
            logging.warning("读取数据健康状态失败：%s", exc)
            return {"label": "未知", "detail": "暂时读不到数据日期", "tone": "warning"}

    def latest_data_date(self) -> str | None:
        try:
            return latest_complete_data_date()
        except Exception as exc:
            logging.warning("读取核心数据日期失败：%s", exc)
            return None
