from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from instock_app.settings import AppPaths


EXPECTED_JOBS = [
    "basic_data_daily_job.py",
    "basic_data_other_daily_job.py",
    "selection_data_daily_job.py",
    "indicators_data_daily_job.py",
    "strategy_data_daily_job.py",
    "verify_daily_update.py",
]

JOB_TIMEOUTS = {
    "indicators_data_daily_job.py": 3600,
}

MAINTENANCE_STEP = "data_maintenance"
STEP_LABELS = {
    "basic_data_daily_job.py": "获取每日行情",
    "basic_data_other_daily_job.py": "获取资金流向、龙虎榜等数据",
    "selection_data_daily_job.py": "获取综合选股数据",
    "indicators_data_daily_job.py": "计算技术指标",
    "strategy_data_daily_job.py": "运行内置策略",
    "verify_daily_update.py": "健康检查",
    MAINTENANCE_STEP: "补齐财报与历史行情缺口",
}
# Seconds measured on a full update; replaced by the previous run's timings.
DEFAULT_STEP_SECONDS = {
    "basic_data_daily_job.py": 220.0,
    "basic_data_other_daily_job.py": 530.0,
    "selection_data_daily_job.py": 280.0,
    "indicators_data_daily_job.py": 410.0,
    "strategy_data_daily_job.py": 280.0,
    "verify_daily_update.py": 5.0,
    MAINTENANCE_STEP: 600.0,
}
HEARTBEAT_SECONDS = 5.0
HEARTBEAT_CAP = 0.95  # never claim a step is done before it is


def _duration_text(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}秒"
    return f"{seconds // 60}分{seconds % 60:02d}秒"


class Heartbeat:
    """Reports elapsed time while a long step runs.

    Jobs only report when they finish, so without this the progress bar sat
    still for minutes and looked frozen.  Progress inside the step is
    estimated from how long the step took last time, capped below its end.
    """

    def __init__(
        self,
        emit: "Emitter",
        step: str,
        start: float,
        end: float,
        expected: float,
        interval: float = HEARTBEAT_SECONDS,
    ) -> None:
        self.emit = emit
        self.step = step
        self.start = start
        self.end = end
        self.expected = max(1.0, expected)
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._began = 0.0

    def message(self, elapsed: float) -> tuple[str, int]:
        fraction = min(HEARTBEAT_CAP, elapsed / self.expected)
        progress = int(self.start + (self.end - self.start) * fraction)
        text = f"正在{STEP_LABELS.get(self.step, self.step)}，已运行{_duration_text(elapsed)}"
        if elapsed <= self.expected:
            text += f"（上次约{_duration_text(self.expected)}）"
        else:
            text += "（比上次慢，仍在运行）"
        return text, progress

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            text, progress = self.message(time.monotonic() - self._began)
            self.emit("heartbeat", text, progress, {"job": self.step})

    def __enter__(self) -> "Heartbeat":
        if self.interval > 0:
            self._began = time.monotonic()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def locked_emitter(emit: "Emitter") -> "Emitter":
    lock = threading.Lock()

    def locked(kind: str, message: str, progress: int = 0, payload: dict | None = None) -> None:
        with lock:
            emit(kind, message, progress, payload)

    return locked


def step_plan(steps: list[str], previous_timings: dict) -> dict[str, tuple[float, float]]:
    """Progress range (start, end) per step, sized by how long each took last time."""
    expected = {}
    for step in steps:
        value = previous_timings.get(step) if isinstance(previous_timings, dict) else None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = 0.0
        expected[step] = seconds if seconds > 0 else DEFAULT_STEP_SECONDS.get(step, 60.0)
    total = sum(expected.values()) or 1.0
    plan, done = {}, 0.0
    for step in steps:
        start = done / total * 99
        done += expected[step]
        plan[step] = (start, done / total * 99)
    return plan


@dataclass(frozen=True)
class JobResult:
    returncode: int
    stdout: str
    stderr: str


class UpdateFailed(RuntimeError):
    pass


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ExclusiveFileLock:
    EMPTY_LOCK_GRACE_SECONDS = 30

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "ExclusiveFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError as exc:
                if attempt == 0 and self._stale_lock():
                    self.path.unlink(missing_ok=True)
                    continue
                raise UpdateFailed("另一个沧浪之水数据更新正在运行") from exc
        assert self.fd is not None
        os.write(self.fd, str(os.getpid()).encode("ascii"))
        return self

    def _stale_lock(self) -> bool:
        try:
            owner = self.path.read_text(encoding="ascii").strip()
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return True
        if not owner:
            return age >= self.EMPTY_LOCK_GRACE_SECONDS
        try:
            pid = int(owner)
        except ValueError:
            return age >= self.EMPTY_LOCK_GRACE_SECONDS
        return not _process_exists(pid)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)


JobRunner = Callable[[str, int], JobResult]
Emitter = Callable[[str, str, int, dict | None], None]


def subprocess_job_runner(paths: AppPaths) -> JobRunner:
    job_dir = paths.instock_root / "instock" / "job"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(paths.instock_root),
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "db_host": "127.0.0.1",  # "localhost" tries IPv6 first: ~2 s per connection
            "db_user": "root",
            "db_password": "root",
            "db_database": "instockdb",
            "db_port": "3306",
        }
    )

    def run(name: str, timeout: int) -> JobResult:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        completed = subprocess.run(
            [str(paths.python_exe), str(job_dir / name)],
            cwd=job_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=creationflags,
            check=False,
        )
        return JobResult(completed.returncode, completed.stdout, completed.stderr)

    return run


MAINTENANCE_TIMEOUT = 1800
SCREENER_JOB = "selection_data_daily_job.py"
SCREENER_RETRY_DELAYS = (10, 30, 60)

FreshnessCheck = Callable[[], tuple[str | None, str | None]]


def database_freshness() -> tuple[str | None, str | None]:
    """Latest dates of the screener table and the daily quote table."""
    import pymysql

    connection = pymysql.connect(
        host="127.0.0.1", port=3306, user="root", password="root", database="instockdb",
        connect_timeout=5, read_timeout=30,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT MAX(date) FROM cn_stock_selection")
            screener = cursor.fetchone()[0]
            cursor.execute("SELECT MAX(date) FROM cn_stock_spot")
            quotes = cursor.fetchone()[0]
    finally:
        connection.close()
    return (str(screener) if screener else None, str(quotes) if quotes else None)


def ensure_screener_fresh(
    runner: "JobRunner",
    freshness: FreshnessCheck,
    emit: "Emitter",
    warnings: list[str],
    sleep_fn: Callable[[float], None],
    progress: int = 97,
) -> None:
    """Re-run the screener job when its table lags the daily quotes.

    The screener source fails intermittently but the job still exits 0, so
    a lagging table would otherwise pass the health check unnoticed.
    """
    try:
        screener, quotes = freshness()
    except Exception as exc:
        warnings.append(f"无法检查综合选股数据日期：{exc}")
        return
    for delay in SCREENER_RETRY_DELAYS:
        if not screener or not quotes or screener >= quotes:
            return
        emit("log", f"综合选股数据停在{screener}，落后于行情{quotes}，{delay}秒后重新抓取", progress, {"job": SCREENER_JOB})
        sleep_fn(delay)
        try:
            runner(SCREENER_JOB, 1800)
            screener, quotes = freshness()
        except Exception as exc:
            warnings.append(f"重新抓取综合选股数据失败：{exc}")
            return
    if screener and quotes and screener < quotes:
        warnings.append(
            f"综合选股数据仍停在{screener}（行情已到{quotes}）；选股将用最新行情价格，概念与财务沿用旧数据"
        )


def subprocess_maintenance_runner(paths: AppPaths) -> Callable[[], JobResult]:
    def run() -> JobResult:
        env = os.environ.copy()
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        completed = subprocess.run(
            [str(paths.python_exe), "-X", "utf8", str(paths.engine_dir / "data_maintenance.py")],
            cwd=paths.engine_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=MAINTENANCE_TIMEOUT,
            creationflags=creationflags,
            check=False,
        )
        return JobResult(completed.returncode, completed.stdout, completed.stderr)

    return run


def run_maintenance(
    maintenance: Callable[[], JobResult],
    emit: Emitter,
    warnings: list[str],
    *,
    span: tuple[float, float] = (98.0, 99.0),
    expected: float = DEFAULT_STEP_SECONDS[MAINTENANCE_STEP],
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> dict[str, object]:
    """Refresh statements and repair history gaps; failures only warn."""
    emit("progress", "正在补齐财报与历史行情缺口", int(span[0]), {"job": MAINTENANCE_STEP})
    try:
        with Heartbeat(emit, MAINTENANCE_STEP, span[0], span[1], expected, heartbeat_seconds):
            result = maintenance()
    except subprocess.TimeoutExpired:
        warnings.append("数据维护超时，策略将使用现有财报与行情")
        return {}
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    summary: dict[str, object] = {}
    if lines:
        try:
            summary = json.loads(lines[-1])
        except json.JSONDecodeError:
            summary = {}
    if result.returncode != 0 or not summary:
        warnings.append(f"数据维护未完成：{(result.stderr or result.stdout).strip()[-300:]}")
        return summary
    history = summary.get("history", {})
    if isinstance(history, dict) and history.get("error"):
        warnings.append(f"历史行情修补失败：{history['error']}")
    fundamentals = summary.get("fundamentals", {})
    if isinstance(fundamentals, dict) and (fundamentals.get("error") or fundamentals.get("failures")):
        warnings.append("部分财报季度刷新失败，郑希景气将使用已有财报")
    emit("log", "数据维护完成", int(span[1]), {"job": MAINTENANCE_STEP, "summary": summary})
    return summary


def run_update(
    paths: AppPaths,
    emit: Emitter,
    run_job: JobRunner | None = None,
    *,
    lock_path: Path | None = None,
    result_path: Path | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    maintenance: Callable[[], JobResult] | None = None,
    freshness: FreshnessCheck | None = None,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> dict[str, object]:
    runner = run_job or subprocess_job_runner(paths)
    if run_job is None:
        maintenance = maintenance or subprocess_maintenance_runner(paths)
        freshness = freshness or database_freshness
    emit = locked_emitter(emit)
    lock = lock_path or Path(r"C:\InStockLocal\logs\daily-update.lock")
    retry_delays = (0, 3, 10)
    warnings: list[str] = []
    job_timings: dict[str, float] = {}
    total_started = clock()

    previous: dict[str, object] = {}
    if result_path and result_path.exists():
        try:
            previous = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    previous_steps = previous.get("step_seconds") or previous.get("job_timings_seconds") or {}
    if not isinstance(previous_steps, dict):
        previous_steps = {}

    def trade_date_from_output(output: str) -> str | None:
        match = re.search(r"\bdate=(\d{4}-\d{2}-\d{2})\b", output)
        return match.group(1) if match else None

    def maintain(
        trade_date: str | None, span: tuple[float, float], warning_list: list[str]
    ) -> tuple[dict, str | None]:
        """Maintenance at most once per trade date; it re-downloads statements and bars."""
        if maintenance is None:
            return {}, None
        if trade_date and previous.get("maintenance_trade_date") == trade_date:
            emit("log", f"{trade_date} 的财报与历史行情已维护过，跳过", int(span[1]), {"job": MAINTENANCE_STEP})
            summary = previous.get("maintenance")
            return (summary if isinstance(summary, dict) else {}), trade_date
        started = clock()
        summary = run_maintenance(
            maintenance, emit, warning_list, span=span,
            expected=float(expected_seconds(MAINTENANCE_STEP)), heartbeat_seconds=heartbeat_seconds,
        )
        job_timings[MAINTENANCE_STEP] = round(max(0.0, clock() - started), 3)
        # A failed run leaves the date unset so the next update tries again.
        return summary, (trade_date if summary else None)

    def expected_seconds(step: str) -> float:
        try:
            value = float(previous_steps.get(step, 0))
        except (TypeError, ValueError):
            value = 0.0
        return value if value > 0 else DEFAULT_STEP_SECONDS.get(step, 60.0)

    def finish(result: dict[str, object], message: str) -> dict[str, object]:
        known = {key: value for key, value in previous_steps.items() if key in DEFAULT_STEP_SECONDS}
        result["step_seconds"] = {**known, **job_timings}
        emit("result", message, 100, result)
        return result

    with ExclusiveFileLock(lock):
        previous_date = previous.get("completed_trade_date")
        if previous.get("healthy") is True:
            quick = step_plan(["verify_daily_update.py", MAINTENANCE_STEP], {"verify_daily_update.py": 5.0, **previous_steps})
            emit(
                "progress",
                "正在快速检查数据是否已是最新",
                0,
                {"job": "verify_daily_update.py"},
            )
            try:
                preflight = runner("verify_daily_update.py", 120)
            except subprocess.TimeoutExpired:
                preflight = JobResult(124, "", "快速健康检查超时")
            current_date = trade_date_from_output(preflight.stdout)
            if (
                preflight.returncode == 0
                and "DAILY_UPDATE_HEALTH=PASS" in preflight.stdout
                and current_date is not None
                and (not isinstance(previous_date, str) or current_date == previous_date)
            ):
                skipped_warnings: list[str] = []
                if freshness is not None:
                    ensure_screener_fresh(
                        runner, freshness, emit, skipped_warnings, sleep_fn,
                        progress=int(quick[MAINTENANCE_STEP][0]),
                    )
                maintenance_summary, maintained = maintain(current_date, quick[MAINTENANCE_STEP], skipped_warnings)
                duration = round(max(0.0, clock() - total_started), 3)
                result = {
                    "healthy": True,
                    "skipped": True,
                    "completed_trade_date": current_date,
                    "jobs": [],
                    "job_timings_seconds": {"verify_daily_update.py": duration},
                    "duration_seconds": duration,
                    "warnings": skipped_warnings,
                    "maintenance": maintenance_summary,
                    "maintenance_trade_date": maintained,
                }
                return finish(result, f"数据已是最新（{current_date}），无需重复更新")
            emit("log", "检测到新交易日或数据不完整，开始完整更新", 0, None)

        steps = EXPECTED_JOBS + ([MAINTENANCE_STEP] if maintenance is not None else [])
        plan = step_plan(steps, previous_steps)
        completed_trade_date: str | None = None
        for name in EXPECTED_JOBS:
            is_health = name == "verify_daily_update.py"
            max_attempts = 1 if is_health else 3
            timeout = JOB_TIMEOUTS.get(name, 1800)
            start, end = plan[name]
            progress = int(start)
            emit("progress", f"正在{STEP_LABELS[name]}（{name}）", progress, {"job": name})
            job_started = clock()
            last: JobResult | None = None
            with Heartbeat(emit, name, start, end, expected_seconds(name), heartbeat_seconds):
                for attempt in range(max_attempts):
                    if attempt:
                        sleep_fn(retry_delays[attempt])
                        emit(
                            "log",
                            f"{name} 第 {attempt + 1}/{max_attempts} 次尝试",
                            progress,
                            {"job": name, "attempt": attempt + 1},
                        )
                    try:
                        last = runner(name, timeout)
                    except subprocess.TimeoutExpired:
                        last = JobResult(124, "", f"{name} 运行超过 {timeout} 秒")
                    if last.stdout:
                        for line in last.stdout.splitlines():
                            emit("log", line, progress, {"job": name})
                    if last.returncode == 0:
                        break
                    if last.stderr:
                        emit("log", last.stderr.strip(), progress, {"job": name})
            assert last is not None
            if last.returncode != 0:
                raise UpdateFailed(
                    f"{name} 在 {max_attempts}次尝试后失败，退出码 {last.returncode}"
                )
            warnings.extend(
                line for line in last.stderr.splitlines() if line.startswith("ERROR:")
            )
            job_timings[name] = round(max(0.0, clock() - job_started), 3)
            emit(
                "log",
                f"{name} 完成，用时 {_duration_text(job_timings[name])}",
                int(end),
                {"job": name, "elapsed_seconds": job_timings[name]},
            )
            if is_health and "DAILY_UPDATE_HEALTH=PASS" not in last.stdout:
                raise UpdateFailed("健康检查未通过：没有 DAILY_UPDATE_HEALTH=PASS")
            if is_health:
                completed_trade_date = trade_date_from_output(last.stdout)
        maintenance_span = plan.get(MAINTENANCE_STEP, (99.0, 99.0))
        if freshness is not None:
            ensure_screener_fresh(
                runner, freshness, emit, warnings, sleep_fn, progress=int(maintenance_span[0])
            )
        maintenance_summary, maintained = maintain(completed_trade_date, maintenance_span, warnings)

    duration = round(max(0.0, clock() - total_started), 3)
    result = {
        "healthy": True,
        "skipped": False,
        "completed_trade_date": completed_trade_date,
        "jobs": EXPECTED_JOBS,
        "job_timings_seconds": {key: value for key, value in job_timings.items() if key in EXPECTED_JOBS},
        "duration_seconds": duration,
        "warnings": warnings,
        "maintenance": maintenance_summary,
        "maintenance_trade_date": maintained,
    }
    return finish(result, "数据更新完成并通过健康检查")
