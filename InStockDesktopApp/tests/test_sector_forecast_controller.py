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

from instock_app.contracts import TaskResult
from instock_app.controller import AppController
from instock_app.result_store import ResultStore
from instock_app.ui.sector_forecast_page import (
    forecast_status_summary,
    format_sector_row,
    universe_status_summary,
)


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def start(self, command, on_event, on_done) -> str:
        self.commands.append(list(command))
        return f"task-{len(self.commands)}"

    def cancel(self, _task_id: str) -> bool:
        return True


class FakeRuntime:
    def start_database(self) -> int:
        return 1


class SectorForecastControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.paths = SimpleNamespace(
            python_exe=Path(sys.executable),
            workers_dir=ROOT / "workers",
            data_dir=root / "data",
        )
        self.runner = FakeRunner()
        self.controller = AppController(
            paths=self.paths,
            runner=self.runner,
            store=ResultStore(root / "store"),
            runtime=FakeRuntime(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_manual_forecast_builds_expected_worker_command(self) -> None:
        self.controller.start_sector_forecast(refresh_history=True)
        command = self.runner.commands[-1]
        self.assertIn("sector-forecast", command)
        self.assertIn("--refresh-history", command)
        self.assertTrue(any(value.endswith("sector-forecast-latest.json") for value in command))

    def test_successful_update_chains_forecast(self) -> None:
        self.controller.state.current_kind = "update"
        self.controller.state.actions_enabled = False
        self.controller.on_done(TaskResult(returncode=0, payload={"healthy": True}))
        for _ in range(4):
            self.controller.on_done(TaskResult(returncode=0, payload={"rows": []}))
        self.controller.on_done(TaskResult(returncode=0, payload={}))
        self.assertEqual(self.controller.state.current_kind, "sector_forecast")
        self.assertIn("sector-forecast", self.runner.commands[-1])

    def test_failed_update_does_not_chain_forecast(self) -> None:
        self.controller.state.current_kind = "update"
        self.controller.state.actions_enabled = False
        self.controller.on_done(TaskResult(returncode=1, payload={}))
        self.assertIsNone(self.controller.state.current_kind)
        self.assertEqual(self.runner.commands, [])

    def test_load_sector_forecast_reads_latest_json(self) -> None:
        self.paths.data_dir.mkdir(parents=True)
        expected = {"as_of": "2026-07-14", "universes": {}}
        (self.paths.data_dir / "sector-forecast-latest.json").write_text(
            json.dumps(expected), encoding="utf-8"
        )
        self.assertEqual(self.controller.load_sector_forecast(), expected)

    def test_sector_page_row_formatter_handles_missing_optional_values(self) -> None:
        row = {
            "rank": 1,
            "sector_name": "测试",
            "strength": "中",
            "confidence": 61,
            "reasons": [],
            "risks": [],
        }
        values = format_sector_row(row)
        self.assertEqual(values[0:4], (1, "测试", "中", "61"))
        self.assertIn("—", values)

    def test_sector_page_marks_forced_fallback_as_reference(self) -> None:
        result = {
            "universes": {
                "industry": {"status": "forced_fallback", "rows": [{"rank": 1}]},
                "concept": {"status": "forced_fallback", "rows": [{"rank": 1}]},
            }
        }
        self.assertEqual(
            forecast_status_summary(result),
            ("参考榜单｜模型验证未通过", "warning"),
        )

    def test_sector_page_prefers_validated_ready_status(self) -> None:
        result = {"universes": {"industry": {"status": "ready", "rows": [{}]}}}
        self.assertEqual(forecast_status_summary(result), ("预测可用", "success"))

    def test_sector_page_reports_head_validated_status(self) -> None:
        result = {"universes": {"concept": {"status": "head_ready", "rows": [{}]}}}
        self.assertEqual(
            forecast_status_summary(result),
            ("前5已验证｜其余参考", "success"),
        )

    def test_universe_status_summary_marks_ready_as_validated(self) -> None:
        self.assertEqual(
            universe_status_summary({"status": "ready"}),
            ("已验证预测", "success"),
        )

    def test_universe_status_summary_marks_forced_fallback_as_reference(self) -> None:
        self.assertEqual(
            universe_status_summary({"status": "forced_fallback"}),
            ("参考榜单｜模型验证未通过", "warning"),
        )

    def test_universe_status_summary_marks_head_ready_precisely(self) -> None:
        self.assertEqual(
            universe_status_summary({"status": "head_ready"}),
            ("前5已验证｜第6-10名参考", "success"),
        )

    def test_universe_status_summary_marks_unavailable_states_as_unavailable(self) -> None:
        expected = ("本期数据不可用", "danger")
        self.assertEqual(universe_status_summary({"status": "unavailable"}), expected)
        self.assertEqual(
            universe_status_summary({"status": "insufficient_confidence"}), expected
        )


if __name__ == "__main__":
    unittest.main()
