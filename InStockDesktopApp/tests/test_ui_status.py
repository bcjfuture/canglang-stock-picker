from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import instock_app.ui.picker_page as picker_page
from instock_app.contracts import TaskEvent
from instock_app.data_status import assess_data_health
from instock_app.ui.dashboard_page import DashboardPage
from instock_app.ui.main_window import MainWindow, fit_window_size
from instock_app.ui.theme import COLORS
from instock_app.ui.picker_page import (
    build_picker_columns,
    change_color,
    format_picker_row,
    format_result_status,
    format_selection_tier,
)


class FakeVariable:
    def set(self, _value) -> None:
        pass


class FakeWidget:
    def configure(self, **_kwargs) -> None:
        pass


class DataHealthTests(unittest.TestCase):
    TODAY = date(2026, 11, 4)  # Wednesday

    def test_fresh_tables_are_healthy(self) -> None:
        dates = {"quotes": "2026-11-03", "screener": "2026-11-03", "fund_flow": "2026-11-03"}
        result = assess_data_health(dates, self.TODAY)
        self.assertEqual("正常", result["label"])

    def test_lagging_screener_is_flagged_even_though_database_runs(self) -> None:
        """Catches the 2 Nov case: quotes current, screener stuck, home page said OK."""
        dates = {"quotes": "2026-11-03", "screener": "2026-10-30", "fund_flow": "2026-11-03"}
        result = assess_data_health(dates, self.TODAY)
        self.assertEqual("需检查", result["label"])
        self.assertIn("综合选股停在2026-10-30", result["detail"])

    def test_old_quotes_ask_for_an_update(self) -> None:
        dates = {"quotes": "2026-10-30", "screener": "2026-10-30", "fund_flow": "2026-10-30"}
        self.assertEqual("需更新", assess_data_health(dates, self.TODAY)["label"])

    def test_monday_accepts_fridays_data(self) -> None:
        dates = {"quotes": "2026-10-30", "screener": "2026-10-30", "fund_flow": "2026-10-30"}
        self.assertEqual("正常", assess_data_health(dates, date(2026, 11, 2))["label"])

    def test_missing_quotes_are_an_error(self) -> None:
        self.assertEqual("danger", assess_data_health({}, self.TODAY)["tone"])


class ResultStatusTests(unittest.TestCase):
    def test_current_result_shows_date_without_warning(self) -> None:
        text, attention = format_result_status(
            {"trade_date": "2026-11-04", "generated_at": "2026-11-04T18:54:38"}, "2026-11-04"
        )
        self.assertIn("选股交易日 2026-11-04", text)
        self.assertIn("18:54", text)
        self.assertFalse(attention)

    def test_old_result_is_called_out(self) -> None:
        text, attention = format_result_status({"trade_date": "2026-10-30"}, "2026-11-04")
        self.assertIn("已过期", text)
        self.assertTrue(attention)

    def test_data_warning_reaches_the_page(self) -> None:
        text, attention = format_result_status(
            {"trade_date": "2026-11-04", "source_summary": {"data_warning": "综合选股数据停在2026-10-30"}},
            "2026-11-04",
        )
        self.assertIn("停在2026-10-30", text)
        self.assertTrue(attention)

    def test_result_without_date_shows_nothing(self) -> None:
        self.assertEqual(("", False), format_result_status({"rows": []}, "2026-11-04"))


class PickerRowLabelTests(unittest.TestCase):
    def row(self, values: dict) -> dict:
        original = picker_page.PickerPage.COLUMNS
        columns = build_picker_columns({"all_market": object()})
        picker_page.PickerPage.COLUMNS = columns
        try:
            return dict(zip(columns, format_picker_row(values)))
        finally:
            picker_page.PickerPage.COLUMNS = original

    def test_reference_advice_is_labelled(self) -> None:
        """Catches generic ATR advice reading like a strategy's own conclusion."""
        reference = self.row({"advice_holding_days": 7, "advice_status": "reference_only"})
        validated = self.row({"advice_holding_days": 7, "advice_status": "ready"})
        self.assertIn("参考", reference["holding_days"])
        self.assertEqual("7日", validated["holding_days"])

    def test_tier_and_holding_count_are_visible(self) -> None:
        self.assertEqual("严格入选", format_selection_tier({"selection_tier": "严格入选", "holding_runs": 1}))
        self.assertEqual("延续持有\n第3次", format_selection_tier({"selection_tier": "延续持有", "holding_runs": 3}))
        self.assertEqual("—", format_selection_tier({}))
        self.assertEqual("严格入选", self.row({"selection_tier": "严格入选"})["selection_tier"])


class ChangeColorTests(unittest.TestCase):
    def test_red_up_green_down(self) -> None:
        """Catches the Western green-up convention slipping into an A-share tool."""
        self.assertEqual(COLORS["danger"], change_color(2.5))
        self.assertEqual(COLORS["success"], change_color(-0.3))

    def test_flat_or_missing_stays_plain(self) -> None:
        self.assertIsNone(change_color(0))
        self.assertIsNone(change_color(None))


class DashboardCopyTests(unittest.TestCase):
    def test_strategy_copy_matches_the_evidence_scoring(self) -> None:
        for key in ("tech_attack", "rebound_trend"):
            _title, tag, description, _color = DashboardPage.STRATEGIES[key]
            for retired in ("动量", "资金", "转强"):
                self.assertNotIn(retired, tag + description)

    def test_view_backtest_only_navigates(self) -> None:
        """Catches the 查看回测 button starting a new backtest."""
        page = DashboardPage.__new__(DashboardPage)
        opened = []
        page.on_navigate = opened.append
        page.controller = SimpleNamespace(run_backtest=lambda: self.fail("must not run a backtest"))
        DashboardPage._open_backtest(page)
        self.assertEqual(["回测中心"], opened)


class WindowFitTests(unittest.TestCase):
    def test_window_stays_inside_the_usable_area(self) -> None:
        """Catches the 1440x900 default running under the taskbar at 150% scaling."""
        width, height = fit_window_size(2240, 1328, 1.5)
        self.assertLessEqual(width * 1.5, 2240)
        self.assertLessEqual(height * 1.5 + 60, 1328)

    def test_large_screen_keeps_the_default_size(self) -> None:
        self.assertEqual((1440, 900), fit_window_size(3840, 2100, 1.0))


class StartupResultTests(unittest.TestCase):
    def test_saved_result_is_shown_on_start_up(self) -> None:
        """Catches the page claiming there are no results when some are saved."""
        page = picker_page.PickerPage.__new__(picker_page.PickerPage)
        shown = []
        page.controller = SimpleNamespace(load_picker_result=lambda key: {"strategy": key, "rows": [1]})
        page.selected_key = lambda: "all_market"
        page.display_result = shown.append
        page.load_latest()
        self.assertEqual("all_market", shown[0]["strategy"])

    def test_missing_or_broken_result_keeps_the_empty_state(self) -> None:
        page = picker_page.PickerPage.__new__(picker_page.PickerPage)
        shown = []
        page.selected_key = lambda: "all_market"
        page.display_result = shown.append
        page.controller = SimpleNamespace(load_picker_result=lambda key: None)
        page.load_latest()

        def broken(_key):
            raise ValueError("bad json")

        page.controller = SimpleNamespace(load_picker_result=broken)
        page.load_latest()
        self.assertEqual([], shown)


class MainWindowBehaviourTests(unittest.TestCase):
    def window(self, selected: str):
        displayed, shown = [], []
        window = MainWindow.__new__(MainWindow)
        window.task_text = FakeVariable()
        window.task_pill = FakeWidget()
        window.pages = {
            "数据中心": SimpleNamespace(handle_event=lambda _event: None),
            "板块预测": SimpleNamespace(handle_event=lambda _event: None),
            "选股中心": SimpleNamespace(display_result=displayed.append, selected_key=lambda: selected),
            "回测中心": SimpleNamespace(display_result=lambda _value: None),
        }
        window.show_page = shown.append
        return window, displayed, shown

    def event(self, strategy: str, automatic: bool) -> TaskEvent:
        payload = {"strategy": strategy, "rows": []}
        if automatic:
            payload["automatic"] = True
        return TaskEvent(kind="result", message="完成", progress=100, payload=payload)

    def test_automatic_run_never_switches_page(self) -> None:
        window, displayed, shown = self.window("all_market")
        MainWindow.handle_event(window, self.event("tech_attack", automatic=True))
        self.assertEqual([], displayed)
        self.assertEqual([], shown)

    def test_automatic_run_refreshes_the_strategy_on_screen(self) -> None:
        window, displayed, shown = self.window("tech_attack")
        MainWindow.handle_event(window, self.event("tech_attack", automatic=True))
        self.assertEqual(1, len(displayed))
        self.assertEqual([], shown)

    def test_manual_run_still_opens_results(self) -> None:
        window, displayed, shown = self.window("all_market")
        MainWindow.handle_event(window, self.event("tech_attack", automatic=False))
        self.assertEqual(1, len(displayed))
        self.assertEqual(["选股中心"], shown)

    def test_busy_refusal_is_shown_to_the_user(self) -> None:
        """Catches clicks during a running task doing nothing visible."""
        window = MainWindow.__new__(MainWindow)
        with patch("instock_app.ui.main_window.messagebox") as messagebox:
            MainWindow._report_callback_exception(window, RuntimeError, RuntimeError("已有任务正在运行"), None)
        messagebox.showinfo.assert_called_once_with("暂时无法执行", "已有任务正在运行")


if __name__ == "__main__":
    unittest.main()
