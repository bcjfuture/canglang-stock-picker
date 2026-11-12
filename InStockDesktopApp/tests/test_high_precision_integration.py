from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from instock_app.contracts import StrategySpec, TaskEvent
from instock_app.settings import AppPaths, build_strategy_registry
from instock_app.ui.main_window import MainWindow
from instock_app.ui.dashboard_page import DashboardPage
from instock_app.ui.picker_page import PickerPage, build_picker_columns
from workers.picker_worker import normalize_picker_payload, run_picker


HIGH_SPEC = StrategySpec(
    "high_precision",
    "高胜率精选",
    "high_precision_config.json",
    "high_precision.json",
    "高胜率精选",
)


class FakeVariable:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value) -> None:
        self.value = value


class FakeWidget:
    def configure(self, **_kwargs) -> None:
        return None


class HighPrecisionIntegrationTests(unittest.TestCase):
    def test_dashboard_describes_the_replaced_all_market_strategy(self) -> None:
        """Catches the home page still advertising the retired trend-first logic."""
        title, tag, description, _color = DashboardPage.STRATEGIES["all_market"]
        self.assertEqual(title, "全市场精选")
        self.assertEqual(tag, "市赚率3.0 × 基本面")
        self.assertIn("PB小于1", description)

    def test_probability_columns_are_hidden_when_strategy_is_not_deployed(self) -> None:
        columns = build_picker_columns({"all_market": object()})
        self.assertEqual(
            (
                "rank", "code", "name", "industry", "price", "change_pct", "holding_days",
                "take_profit", "stop_loss", "score", "selection_tier", "pb", "market_earnings_ratio",
                "news_score", "news_summary",
                "large_order_flow", "retail_flow",
                "fund_flow_1d", "fund_flow_3d", "fund_flow_5d",
                "shareholder_score", "reason",
            ),
            columns,
        )
        self.assertNotIn("win_probability", columns)
        self.assertNotIn("market_regime", columns)
        self.assertNotIn("probability_threshold", columns)

    def test_probability_columns_are_shown_when_strategy_is_deployed(self) -> None:
        columns = build_picker_columns({"high_precision": object()})
        self.assertEqual(
            (
                "large_order_flow", "retail_flow", "fund_flow_1d",
                "fund_flow_3d", "fund_flow_5d", "shareholder_score", "reason",
            ),
            columns[-7:],
        )
        self.assertIn("win_probability", columns)
        self.assertIn("market_regime", columns)
        self.assertIn("probability_threshold", columns)

    def test_registry_never_exposes_high_precision_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "engine"
            data = root / "data"
            engine.mkdir()
            data.mkdir()

            strategies, default = build_strategy_registry(engine, data)
            self.assertEqual(
                [
                    "all_market",
                    "tech_attack",
                    "rebound_trend",
                    "zhengxi_prosperity",
                ],
                list(strategies),
            )
            self.assertEqual("all_market", default)

            (engine / "high_precision_model.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "deployment": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            (data / "high_precision-backtest-latest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "deployment": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            strategies, default = build_strategy_registry(engine, data)
            self.assertNotIn("high_precision", strategies)
            self.assertEqual("all_market", default)

            (data / "high_precision-backtest-latest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-2",
                        "deployment": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            strategies, default = build_strategy_registry(engine, data)
            self.assertNotIn("high_precision", strategies)
            self.assertEqual("all_market", default)

    def test_worker_dispatches_high_precision_script_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = replace(
                AppPaths.default(),
                data_dir=root / "data",
                engine_dir=ROOT / "engine",
            )
            captured = []

            def fake_run(command, **_kwargs):
                captured.extend(str(value) for value in command)
                raw_path = Path(command[4])
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(
                    json.dumps(
                        {
                            "generated_at": "2026-07-13 16:00:00",
                            "latest_dates": {"综合选股": "2026-07-13"},
                            "strategy_gate_passed": True,
                            "trade_policy": {
                                "policy_version": "high-precision-5d-v1",
                                "holding_days": 5,
                                "atr_multiplier": 1.5,
                                "reward_risk": 2.0,
                            },
                            "summary": {"abstain_reason": "弱市空仓"},
                            "sheets": {"高胜率精选": []},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch(
                    "workers.picker_worker.STRATEGIES",
                    {"high_precision": HIGH_SPEC},
                ),
                patch(
                    "workers.picker_worker.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                result = run_picker(
                    "high_precision",
                    paths,
                    lambda *_args: None,
                    as_of_date="2026-07-13",
                )

            self.assertTrue(captured[3].endswith("high_precision_picker_data.py"))
            self.assertTrue(captured[6].endswith("high_precision_model.json"))
            self.assertEqual("2026-07-13", captured[7])
            self.assertEqual([], result["rows"])
            self.assertEqual("弱市空仓", result["abstain_reason"])

    def test_normalization_exposes_probability_fields(self) -> None:
        payload = {
            "generated_at": "2026-07-13 16:00:00",
            "latest_dates": {"综合选股": "2026-07-13"},
            "strategy_gate_passed": True,
            "trade_policy": {
                "policy_version": "high-precision-5d-v1",
                "holding_days": 5,
                "atr_multiplier": 1.5,
                "reward_risk": 2.0,
            },
            "summary": {
                "market_regime": "normal",
                "probability_threshold": 0.8,
                "abstain_reason": "",
            },
            "sheets": {
                "高胜率精选": [
                    {
                        "代码": "000001",
                        "名称": "测试股",
                        "行业": "软件",
                        "最新价": 10.0,
                        "涨跌幅%": 1.0,
                        "综合评分": 87.0,
                        "今日主力净流入_万元": 100.0,
                        "ATR14": 0.5,
                        "5日上涨概率": 0.87,
                        "市场状态": "normal",
                        "概率门槛": 0.8,
                        "主要正向因子": "趋势质量",
                        "主要负向因子": "波动率",
                        "通过高胜率门槛": True,
                    }
                ]
            },
        }

        result = normalize_picker_payload(HIGH_SPEC, payload)
        row = result["rows"][0]
        self.assertEqual(0.87, row["win_probability"])
        self.assertEqual("normal", row["market_regime"])
        self.assertEqual(0.8, row["probability_threshold"])
        self.assertTrue(row["high_precision_gate_passed"])

    def test_main_window_routes_valid_empty_picker_result(self) -> None:
        displayed = []
        window = MainWindow.__new__(MainWindow)
        window.task_text = FakeVariable()
        window.task_pill = FakeWidget()
        window.pages = {
            "数据中心": SimpleNamespace(handle_event=lambda _event: None),
            "板块预测": SimpleNamespace(handle_event=lambda _event: None),
            "选股中心": SimpleNamespace(
                display_result=displayed.append
            ),
            "回测中心": SimpleNamespace(display_result=lambda _value: None),
        }
        shown = []
        window.show_page = shown.append

        with patch(
            "instock_app.ui.main_window.STRATEGIES",
            {"high_precision": HIGH_SPEC},
            create=True,
        ):
            MainWindow.handle_event(
                window,
                TaskEvent(
                    kind="result",
                    message="完成",
                    progress=100,
                    payload={
                        "strategy": "high_precision",
                        "rows": [],
                        "abstain_reason": "弱市空仓",
                    },
                ),
            )

        self.assertEqual(1, len(displayed))
        self.assertEqual(["选股中心"], shown)


if __name__ == "__main__":
    unittest.main()
