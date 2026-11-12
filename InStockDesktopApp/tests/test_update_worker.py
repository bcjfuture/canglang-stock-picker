from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from workers.update_worker import EXPECTED_JOBS, JobResult, run_update


class UpdateWorkerTests(unittest.TestCase):
    def test_legacy_healthy_result_can_use_fast_health_check(self):
        calls: list[str] = []

        def runner(name: str, _timeout: int) -> JobResult:
            calls.append(name)
            return JobResult(
                0,
                "cn_stock_spot: date=2026-07-17 rows=1906 unique=1906 null_keys=0\n"
                "DAILY_UPDATE_HEALTH=PASS\n",
                "",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "update-latest.json"
            result_path.write_text(json.dumps({"healthy": True}), encoding="utf-8")
            result = run_update(
                SimpleNamespace(),
                lambda *_event: None,
                runner,
                lock_path=root / "update.lock",
                result_path=result_path,
            )

        self.assertEqual(["verify_daily_update.py"], calls)
        self.assertTrue(result["skipped"])
        self.assertEqual("2026-07-17", result["completed_trade_date"])

    def test_same_completed_trade_date_uses_fast_health_check_and_skips_full_update(self):
        calls: list[str] = []
        events: list[tuple[str, str, int, dict | None]] = []

        def runner(name: str, _timeout: int) -> JobResult:
            calls.append(name)
            self.assertEqual("verify_daily_update.py", name)
            return JobResult(
                0,
                "cn_stock_spot: date=2026-07-17 rows=1906 unique=1906 null_keys=0\n"
                "DAILY_UPDATE_HEALTH=PASS\n",
                "",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "update-latest.json"
            result_path.write_text(
                json.dumps({"healthy": True, "completed_trade_date": "2026-07-17"}),
                encoding="utf-8",
            )
            result = run_update(
                SimpleNamespace(),
                lambda *event: events.append(event),
                runner,
                lock_path=root / "update.lock",
                result_path=result_path,
            )

        self.assertEqual(["verify_daily_update.py"], calls)
        self.assertTrue(result["healthy"])
        self.assertTrue(result["skipped"])
        self.assertEqual("2026-07-17", result["completed_trade_date"])
        self.assertEqual([], result["jobs"])
        self.assertEqual("result", events[-1][0])
        self.assertEqual(100, events[-1][2])

    def test_full_update_records_trade_date_and_job_timings(self):
        calls: list[str] = []
        ticks = iter(range(30))

        def runner(name: str, _timeout: int) -> JobResult:
            calls.append(name)
            stdout = ""
            if name == "verify_daily_update.py":
                stdout = (
                    "cn_stock_spot: date=2026-07-17 rows=1906 unique=1906 null_keys=0\n"
                    "DAILY_UPDATE_HEALTH=PASS\n"
                )
            return JobResult(0, stdout, "")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = run_update(
                SimpleNamespace(),
                lambda *_event: None,
                runner,
                lock_path=root / "update.lock",
                result_path=root / "missing.json",
                clock=lambda: float(next(ticks)),
            )

        self.assertEqual(EXPECTED_JOBS, calls)
        self.assertFalse(result["skipped"])
        self.assertEqual("2026-07-17", result["completed_trade_date"])
        self.assertEqual(set(EXPECTED_JOBS), set(result["job_timings_seconds"]))
        self.assertGreaterEqual(result["duration_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
