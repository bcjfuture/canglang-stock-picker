from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import instock_tech_picker_data as picker
from workers.update_worker import JobResult, ensure_screener_fresh


class FakeCursor:
    """Answers the snapshot queries with fixed latest dates."""

    def __init__(self, dates: dict[str, date]) -> None:
        self.dates = dates
        self.queries: list[tuple[str, tuple]] = []
        self._result: list = []

    def execute(self, sql, params=None):
        self.queries.append((sql, tuple(params or ())))
        text = " ".join(sql.split())
        if "information_schema" in text:
            self._result = [{"n": 1}]
        elif text.upper().startswith("SELECT MAX"):
            table = next((name for name in self.dates if name in text), None)
            self._result = [{"d": self.dates.get(table)}]
        else:
            self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return self._result


class SnapshotQuoteOverlayTests(unittest.TestCase):
    def snapshot(self, screener: date, quotes: date):
        cursor = FakeCursor(
            {"cn_stock_selection": screener, "cn_stock_spot": quotes, "cn_stock_fund_flow": quotes}
        )
        original = (picker.fetch_code_set, picker.fetch_strategy_hits)
        picker.fetch_code_set = lambda cur, name, as_of=None: set()
        picker.fetch_strategy_hits = lambda cur, as_of=None: {}
        try:
            result = picker.load_selection_snapshot(cursor)
        finally:
            picker.fetch_code_set, picker.fetch_strategy_hits = original
        return result, cursor.queries[-1]

    def test_lagging_screener_uses_latest_quotes_and_signal_date(self) -> None:
        """Catches showing last week's prices when the screener download failed."""
        result, (sql, params) = self.snapshot(date(2026, 10, 30), date(2026, 11, 2))

        self.assertEqual(date(2026, 11, 2), result["selection_date"])
        self.assertEqual(date(2026, 10, 30), result["screener_date"])
        self.assertIn("COALESCE(p.new_price, s.new_price)", sql)
        self.assertIn("LEFT JOIN cn_stock_spot p", sql)
        self.assertEqual((date(2026, 11, 2), date(2026, 11, 2), date(2026, 11, 2), date(2026, 10, 30)), params)
        self.assertIn("2026-10-30", picker.screener_lag_warning(result))

    def test_current_screener_is_used_directly(self) -> None:
        result, (sql, _params) = self.snapshot(date(2026, 11, 2), date(2026, 11, 2))

        self.assertNotIn("cn_stock_spot p", sql)
        self.assertEqual("", picker.screener_lag_warning(result))


class ScreenerRetryTests(unittest.TestCase):
    def test_lagging_screener_is_refetched(self) -> None:
        states = iter([("2026-10-30", "2026-11-02"), ("2026-11-02", "2026-11-02")])
        calls, warnings = [], []
        ensure_screener_fresh(
            lambda name, timeout: calls.append(name) or JobResult(0, "", ""),
            lambda: next(states),
            lambda *args: None,
            warnings,
            lambda seconds: None,
        )
        self.assertEqual(["selection_data_daily_job.py"], calls)
        self.assertEqual([], warnings)

    def test_persistent_lag_is_reported(self) -> None:
        calls, warnings = [], []
        ensure_screener_fresh(
            lambda name, timeout: calls.append(name) or JobResult(0, "", ""),
            lambda: ("2026-10-30", "2026-11-02"),
            lambda *args: None,
            warnings,
            lambda seconds: None,
        )
        self.assertEqual(3, len(calls))
        self.assertTrue(any("2026-10-30" in text for text in warnings))


if __name__ == "__main__":
    unittest.main()
