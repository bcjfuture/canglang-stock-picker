from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fundamentals_refresh import quarters_to_refresh
from workers.update_worker import JobResult, run_maintenance, run_update
from instock_app.settings import AppPaths


class QuarterRefreshTests(unittest.TestCase):
    def test_half_year_is_refetched_during_its_disclosure_season(self) -> None:
        """Catches a June report fetched in July never being refreshed."""
        loaded = {"20260331", "20260630", "20251231"}
        requested = quarters_to_refresh(date(2026, 8, 25), loaded)
        self.assertIn("20260630", requested)
        self.assertNotIn("20260331", requested)

    def test_missing_quarter_is_fetched_after_season(self) -> None:
        requested = quarters_to_refresh(date(2026, 10, 8), {"20260331"})
        self.assertIn("20260630", requested)

    def test_closed_season_is_not_refetched(self) -> None:
        loaded = {"20251231", "20260331", "20260630"}
        requested = quarters_to_refresh(date(2026, 10, 8), loaded)
        self.assertNotIn("20260630", requested)
        # Q3 has ended and its season is open, so it is fetched.
        self.assertIn("20260930", requested)


class MaintenanceHookTests(unittest.TestCase):
    def test_maintenance_failure_only_warns(self) -> None:
        warnings: list[str] = []
        events = []
        summary = run_maintenance(
            lambda: JobResult(1, "", "network down"), lambda *args: events.append(args), warnings
        )
        self.assertEqual({}, summary)
        self.assertTrue(any("数据维护未完成" in text for text in warnings))

    def test_update_runs_maintenance_after_jobs(self) -> None:
        calls = []

        def runner(name: str, timeout: int) -> JobResult:
            calls.append(name)
            out = "DAILY_UPDATE_HEALTH=PASS date=2026-09-30" if name == "verify_daily_update.py" else ""
            return JobResult(0, out, "")

        def maintenance() -> JobResult:
            calls.append("maintenance")
            return JobResult(0, json.dumps({"history": {"repaired": 3}}), "")

        with tempfile.TemporaryDirectory() as temporary:
            result = run_update(
                AppPaths.default(),
                lambda *args: None,
                runner,
                lock_path=Path(temporary) / "lock",
                maintenance=maintenance,
            )
        self.assertEqual("maintenance", calls[-1])
        self.assertEqual({"history": {"repaired": 3}}, result["maintenance"])


if __name__ == "__main__":
    unittest.main()
