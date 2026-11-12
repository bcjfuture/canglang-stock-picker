from __future__ import annotations

import importlib
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from instock_app.contracts import TaskResult
from instock_app.ui.main_window import MainWindow


class FakeCursor:
    def __init__(self, value) -> None:
        self.value = value
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql: str) -> None:
        self.sql = sql

    def fetchone(self):
        return (self.value,)


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self._cursor


class FakeVariable:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class FakeWidget:
    def configure(self, **_kwargs) -> None:
        return None


class FakeRoot:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []

    def after(self, delay: int, callback) -> None:
        self.after_calls.append((delay, callback))


def load_data_status():
    try:
        return importlib.import_module("instock_app.data_status")
    except ModuleNotFoundError as exc:
        raise AssertionError("instock_app.data_status is missing") from exc


class DataDateQueryTests(unittest.TestCase):
    def test_latest_complete_date_uses_all_core_tables(self) -> None:
        module = load_data_status()
        cursor = FakeCursor(date(2026, 7, 16))

        result = module.latest_complete_data_date(
            connection_factory=lambda: FakeConnection(cursor)
        )

        self.assertEqual("2026-07-16", result)
        self.assertIn("MIN(latest_date)", cursor.sql)
        self.assertIn("cn_stock_indicators", cursor.sql)
        for table in module.CORE_DATA_TABLES:
            self.assertIn(table, cursor.sql)

    def test_latest_complete_date_returns_none_without_usable_date(self) -> None:
        module = load_data_status()
        cursor = FakeCursor(None)

        result = module.latest_complete_data_date(
            connection_factory=lambda: FakeConnection(cursor)
        )

        self.assertIsNone(result)


class MainWindowDataDateTests(unittest.TestCase):
    def test_successful_update_completion_refreshes_data_date(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window.controller = SimpleNamespace(
            state=SimpleNamespace(last_completed_kind="update")
        )
        window.task_text = FakeVariable()
        window.task_pill = FakeWidget()
        window.pages = {}
        refreshes: list[bool] = []
        window._refresh_data_date = lambda: refreshes.append(True)

        MainWindow.handle_done(window, TaskResult(returncode=0))

        self.assertEqual([True], refreshes)

    def test_periodic_refresh_schedules_next_run_after_one_minute(self) -> None:
        schedule = getattr(MainWindow, "_schedule_data_date_refresh", None)
        self.assertIsNotNone(schedule)
        window = MainWindow.__new__(MainWindow)
        window.root = FakeRoot()
        refreshes: list[bool] = []
        window._refresh_data_date = lambda: refreshes.append(True)

        schedule(window)

        self.assertEqual([True], refreshes)
        self.assertEqual(1, len(window.root.after_calls))
        delay, callback = window.root.after_calls[0]
        self.assertEqual(60_000, delay)
        self.assertEqual(window._schedule_data_date_refresh, callback)


if __name__ == "__main__":
    unittest.main()
