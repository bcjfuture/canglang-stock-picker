from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from instock_app.controller import AppController
from instock_app.result_store import ResultStore
from instock_app.ui.picker_page import PickerPage


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def start(self, command, _on_event, _on_done) -> str:
        self.commands.append(list(command))
        return "task-1"


class RecordingPickerPage(PickerPage):
    def display_result(self, result: dict) -> None:
        self.displayed_results.append(result)


class PickerStrategySwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.runner = RecordingRunner()
        self.store = ResultStore(root / "store")
        self.controller = AppController(
            paths=SimpleNamespace(),
            runner=self.runner,
            store=self.store,
            runtime=SimpleNamespace(),
        )
        self.page = RecordingPickerPage.__new__(RecordingPickerPage)
        self.page.controller = self.controller
        self.page.title_to_key = {
            "全市场精选": "all_market",
            "科技进攻": "tech_attack",
            "反弹趋势": "rebound_trend",
            "郑希景气精选": "zhengxi_prosperity",
        }
        self.page.displayed_results = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_switching_to_completed_strategy_displays_cached_result_without_rerun(self) -> None:
        cached = {
            "strategy": "tech_attack",
            "generated_at": "2026-08-26 20:55:00",
            "rows": [{"code": "000001", "name": "平安银行"}],
        }
        self.store.record_run("tech_attack", cached)

        self.page.on_strategy_change("科技进攻")

        self.assertEqual([cached], self.page.displayed_results)
        self.assertEqual([], self.runner.commands)

    def test_switching_to_never_run_strategy_shows_empty_result_without_rerun(self) -> None:
        self.page.on_strategy_change("反弹趋势")

        self.assertEqual(
            [{"strategy": "rebound_trend", "rows": []}],
            self.page.displayed_results,
        )
        self.assertEqual([], self.runner.commands)


if __name__ == "__main__":
    unittest.main()
