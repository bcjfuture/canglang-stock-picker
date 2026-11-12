from __future__ import annotations

import importlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from instock_app.contracts import TaskResult
from instock_app.controller import AppController
from instock_app.result_store import ResultStore
from instock_app.ui.main_window import MainWindow


class FakeVariable:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value) -> None:
        self.value = value


class FakeWidget:
    def __init__(self) -> None:
        self.placed = False
        self.options = {}

    def configure(self, **kwargs) -> None:
        self.options.update(kwargs)

    def place(self, **_kwargs) -> None:
        self.placed = True


class FakeTree:
    def __init__(self) -> None:
        self.deleted = []

    def get_children(self):
        return ("old-row",)

    def delete(self, *items) -> None:
        self.deleted.extend(items)


class FakeConsensusPage:
    def __init__(self) -> None:
        self.refresh_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1


class ConsensusPageTests(unittest.TestCase):
    def test_navigation_places_three_and_five_day_consensus_before_sector_forecast(self) -> None:
        picker_index = MainWindow.PAGE_NAMES.index("选股中心")

        self.assertEqual("三日共识", MainWindow.PAGE_NAMES[picker_index + 1])
        self.assertEqual("五日共识", MainWindow.PAGE_NAMES[picker_index + 2])
        self.assertEqual("板块预测", MainWindow.PAGE_NAMES[picker_index + 3])

    def test_heat_strip_exposes_daily_strategy_intensity(self) -> None:
        try:
            module = importlib.import_module("instock_app.ui.consensus_page")
        except ModuleNotFoundError:
            self.fail("五日共识页面尚未实现")

        self.assertEqual("·  ▂  ▄  ▆  █", module.format_daily_heat([0, 1, 2, 3, 4]))

    def test_heat_column_heading_matches_each_consensus_window(self) -> None:
        module = importlib.import_module("instock_app.ui.consensus_page")

        three_day_columns = module.consensus_columns(3)
        five_day_columns = module.consensus_columns(5)

        self.assertEqual("三日热度（旧 → 新）", three_day_columns[-2][1])
        self.assertEqual("五日热度（旧 → 新）", five_day_columns[-2][1])

    def test_coverage_uses_dynamic_strategy_day_denominator(self) -> None:
        module = importlib.import_module("instock_app.ui.consensus_page")

        self.assertTrue(
            hasattr(module, "format_coverage"),
            "页面尚未使用动态样本分母",
        )
        self.assertEqual(
            "覆盖 3/5日 · 9/15样本",
            module.format_coverage(
                {
                    "covered_days": 3,
                    "day_limit": 5,
                    "available_strategy_days": 9,
                    "maximum_appearances": 15,
                }
            ),
        )
        self.assertEqual(
            "覆盖 5/5日 · 19/20样本 · 跳过1个异常快照",
            module.format_coverage(
                {
                    "covered_days": 5,
                    "day_limit": 5,
                    "available_strategy_days": 19,
                    "maximum_appearances": 20,
                    "skipped_snapshot_count": 1,
                }
            ),
        )

    def test_refresh_loads_history_off_the_ui_thread(self) -> None:
        module = importlib.import_module("instock_app.ui.consensus_page")
        delivered = []
        callbacks = []

        class SlowController:
            def load_consensus(self, day_limit):
                time.sleep(0.25)
                return {"day_limit": day_limit, "rows": []}

        page = module.ConsensusPage.__new__(module.ConsensusPage)
        page.controller = SlowController()
        page.day_limit = 3
        page._refresh_serial = 0
        page.updated_text = FakeVariable()
        page.display_result = delivered.append
        page.after = lambda _delay, callback: callbacks.append(callback)

        started = time.perf_counter()
        page.refresh()
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.10, "刷新不应阻塞 Tk 主线程")
        page._refresh_thread.join(timeout=2)
        self.assertEqual(1, len(callbacks))
        callbacks[0]()
        self.assertEqual([{"day_limit": 3, "rows": []}], delivered)

    def test_rapid_refreshes_are_coalesced_without_concurrent_history_scans(self) -> None:
        module = importlib.import_module("instock_app.ui.consensus_page")
        started = threading.Event()
        release = threading.Event()
        callbacks = []
        delivered = []

        class BlockingController:
            def __init__(self) -> None:
                self.calls = 0
                self.active = 0
                self.maximum_active = 0

            def load_consensus(self, day_limit):
                self.calls += 1
                call = self.calls
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                if call == 1:
                    started.set()
                    release.wait(timeout=2)
                self.active -= 1
                return {"generation": call, "day_limit": day_limit, "rows": []}

        controller = BlockingController()
        page = module.ConsensusPage.__new__(module.ConsensusPage)
        page.controller = controller
        page.day_limit = 3
        page._refresh_serial = 0
        page.updated_text = FakeVariable()
        page.display_result = delivered.append
        page.after = lambda _delay, callback: callbacks.append(callback)

        page.refresh()
        self.assertTrue(started.wait(timeout=1))
        page.refresh()
        page.refresh()

        self.assertEqual(1, controller.calls)
        release.set()
        page._refresh_thread.join(timeout=2)
        callbacks.pop(0)()
        page._refresh_thread.join(timeout=2)
        callbacks.pop(0)()

        self.assertEqual(2, controller.calls)
        self.assertEqual(1, controller.maximum_active)
        self.assertEqual(
            [{"generation": 2, "day_limit": 3, "rows": []}],
            delivered,
        )

    def test_failed_refresh_clears_stale_ranking(self) -> None:
        module = importlib.import_module("instock_app.ui.consensus_page")
        self.assertTrue(
            hasattr(module.ConsensusPage, "_display_error"),
            "读取失败时尚未清理旧榜单",
        )
        page = module.ConsensusPage.__new__(module.ConsensusPage)
        page.result = {"rows": [{"code": "000001"}]}
        page.rows_by_item = {"old-row": {"code": "000001"}}
        page.tree = FakeTree()
        page.period_text = FakeVariable()
        page.coverage_text = FakeVariable()
        page.updated_text = FakeVariable()
        page.detail_title = FakeVariable()
        page.detail_text = FakeVariable()
        page.empty_label = FakeWidget()
        page.window_label = "五"

        page._display_error(RuntimeError("磁盘读取失败"))

        self.assertEqual({}, page.result)
        self.assertEqual({}, page.rows_by_item)
        self.assertEqual(["old-row"], page.tree.deleted)
        self.assertEqual("读取失败", page.updated_text.value)
        self.assertTrue(page.empty_label.placed)

        page.display_result(
            {
                "trade_dates": [],
                "covered_days": 0,
                "day_limit": 5,
                "available_strategy_days": 0,
                "maximum_appearances": 20,
                "rows": [],
            }
        )
        self.assertEqual(
            "暂无选股历史，请先运行策略",
            page.empty_label.options.get("text"),
        )

    def test_controller_loads_consensus_from_recorded_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResultStore(Path(temporary) / "store")
            history = store.root / "history" / "tech_attack"
            history.mkdir(parents=True)
            (history / "one.json").write_text(
                json.dumps(
                    {
                        "strategy": "tech_attack",
                        "trade_date": "2026-08-28",
                        "generated_at": "2026-08-28 20:00:00",
                        "rows": [{"code": "000001", "name": "平安银行", "rank": 1}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            controller = AppController(
                paths=SimpleNamespace(),
                runner=SimpleNamespace(),
                store=store,
                runtime=SimpleNamespace(),
            )

            self.assertTrue(
                hasattr(controller, "load_five_day_consensus"),
                "控制器尚未暴露五日共识数据",
            )
            result = controller.load_five_day_consensus()

            self.assertEqual("000001", result["rows"][0]["code"])
            self.assertEqual(3, result["strategy_count"])

            three_day = controller.load_consensus(3)
            self.assertEqual(3, three_day["day_limit"])
            self.assertEqual(9, three_day["maximum_appearances"])

    def test_consensus_excludes_all_market_hits_and_dates_but_keeps_picker_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResultStore(Path(temporary) / "store")
            for day in (20, 21, 22, 23, 24):
                for strategy in ("all_market", "tech_attack", "rebound_trend", "zhengxi_prosperity"):
                    rows = [{"code": "000001", "name": "共同入选", "rank": 4}]
                    if strategy == "all_market":
                        rows = [
                            {"code": "000001", "name": "共同入选", "rank": 1},
                            {"code": "000002", "name": "仅全市场入选", "rank": 2},
                        ]
                    store.record_run(strategy, {
                        "strategy": strategy,
                        "trade_date": f"2026-08-{day}",
                        "generated_at": f"2026-08-{day} 20:00:00",
                        "rows": rows,
                    })
            store.record_run("all_market", {
                "strategy": "all_market", "trade_date": "2026-08-25",
                "generated_at": "2026-08-25 20:00:00",
                "rows": [{"code": "000002", "name": "仅全市场入选", "rank": 1}],
            })
            controller = AppController(SimpleNamespace(), SimpleNamespace(), store, SimpleNamespace())
            for days, maximum, start in ((3, 9, "2026-08-22"), (5, 15, "2026-08-20")):
                with self.subTest(days=days):
                    result = controller.load_consensus(days)
                    self.assertEqual(3, result["strategy_count"])
                    self.assertEqual(maximum, result["maximum_appearances"])
                    self.assertEqual(maximum, result["available_strategy_days"])
                    self.assertEqual(start, result["date_start"])
                    self.assertEqual("2026-08-24", result["as_of"])
                    self.assertEqual(["000001"], [row["code"] for row in result["rows"]])
                    row = result["rows"][0]
                    self.assertEqual(maximum, row["appearance_count"])
                    self.assertEqual(days, row["day_count"])
                    self.assertEqual(3, row["strategy_count"])
                    self.assertEqual(4.0, row["average_rank"])
                    self.assertEqual([3] * days, row["daily_counts"])
                    for detail in row["details"]:
                        self.assertNotIn("all_market", [hit["strategy"] for hit in detail["hits"]])
                    module = importlib.import_module("instock_app.ui.consensus_page")
                    formatted = module.format_consensus_row(row, result)
                    self.assertEqual(f"{maximum}/{maximum}", formatted[3])
                    self.assertEqual("3/3策略", formatted[5])
            self.assertEqual("000002", controller.load_picker_result("all_market")["rows"][0]["code"])

    def test_successful_picker_run_refreshes_consensus_page(self) -> None:
        three_day_consensus = FakeConsensusPage()
        five_day_consensus = FakeConsensusPage()
        window = MainWindow.__new__(MainWindow)
        window.controller = SimpleNamespace(
            state=SimpleNamespace(last_completed_kind="picker"),
        )
        window.task_text = FakeVariable()
        window.task_pill = FakeWidget()
        window.pages = {
            "三日共识": three_day_consensus,
            "五日共识": five_day_consensus,
        }

        MainWindow.handle_done(window, TaskResult(returncode=0))

        self.assertEqual(1, three_day_consensus.refresh_count)
        self.assertEqual(1, five_day_consensus.refresh_count)


if __name__ == "__main__":
    unittest.main()
