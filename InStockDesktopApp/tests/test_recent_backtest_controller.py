from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from instock_app.contracts import TaskEvent
from instock_app.controller import AppController
from instock_app.settings import AppPaths
from instock_app.ui.main_window import MainWindow
from workers.worker_cli import build_parser


class FakeRunner:
    def __init__(self) -> None:
        self.command = None

    def start(self, command, _on_event, _on_done):
        self.command = list(command)
        return "task-1"


class FakeVariable:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value) -> None:
        self.value = value


class FakeWidget:
    def configure(self, **_kwargs) -> None:
        return None


class FakeBacktestPage:
    def __init__(self) -> None:
        self.results = []

    def display_result(self, result) -> None:
        self.results.append(result)


class RecentBacktestControllerTests(unittest.TestCase):
    def test_cli_has_recent_backtest_command(self) -> None:
        args = build_parser().parse_args(
            ["recent-backtest", "--result", "recent.json"]
        )

        self.assertEqual("recent-backtest", args.command)
        self.assertEqual(Path("recent.json"), args.result)

    def test_controller_runs_recent_backtest_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner()
            paths = replace(
                AppPaths.default(), data_dir=Path(temporary) / "data"
            )
            controller = AppController(
                paths=paths,
                runner=runner,
                store=SimpleNamespace(),
                runtime=SimpleNamespace(),
            )

            controller.run_backtest()

            self.assertIsNotNone(runner.command)
            self.assertIn("recent-backtest", runner.command)
            self.assertIn(
                str(paths.data_dir / "recent-backtest-latest.json"),
                runner.command,
            )

    def test_main_window_routes_recent_backtest_result_event(self) -> None:
        page = FakeBacktestPage()
        recent_result = {"evaluation_date": "2026-07-16", "offsets": {}}
        window = MainWindow.__new__(MainWindow)
        window.controller = SimpleNamespace(load_backtest=lambda: recent_result)
        window.task_text = FakeVariable()
        window.task_pill = FakeWidget()
        window.pages = {
            "数据中心": SimpleNamespace(handle_event=lambda _event: None),
            "板块预测": SimpleNamespace(handle_event=lambda _event: None),
            "选股中心": SimpleNamespace(display_result=lambda _payload: None),
            "回测中心": page,
        }
        shown = []
        window.show_page = shown.append

        MainWindow.handle_event(
            window,
            TaskEvent(
                kind="result",
                message="完成",
                progress=100,
                payload={"recent_backtest": True},
            ),
        )

        self.assertEqual([recent_result], page.results)
        self.assertEqual(["回测中心"], shown)


if __name__ == "__main__":
    unittest.main()
