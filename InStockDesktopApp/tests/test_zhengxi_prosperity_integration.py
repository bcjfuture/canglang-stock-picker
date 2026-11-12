from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "engine"))

from instock_app.settings import AppPaths, STRATEGIES, build_strategy_registry
from workers.picker_worker import run_picker
from picker_news import NewsItem


class ZhengxiStrategyRegistryTests(unittest.TestCase):
    def test_public_registry_stays_fixed_when_high_precision_artifacts_pass(self) -> None:
        """Catches re-enabling the removed high-precision product entry."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "engine"
            data = root / "data"
            engine.mkdir()
            data.mkdir()
            (engine / "high_precision_model.json").write_text(
                json.dumps({"run_id": "same", "deployment": {"passed": True}}),
                encoding="utf-8",
            )
            (data / "high_precision-backtest-latest.json").write_text(
                json.dumps({"run_id": "same", "deployment": {"passed": True}}),
                encoding="utf-8",
            )

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
        self.assertNotIn("high_precision", strategies)


class ZhengxiPayloadIntegrationTests(unittest.TestCase):
    def test_news_risk_is_applied_before_fixed_count_selection(self) -> None:
        """Catches selecting on the base score before applying a hard news risk."""
        from zhengxi_prosperity_picker_data import apply_news_to_scored_frame

        class Provider:
            def fetch(self, code, start_date, end_date):
                if code == "000001":
                    return [
                        NewsItem(
                            code,
                            datetime(2026, 8, 12, 9),
                            "公司因重大违法被立案调查",
                            "交易所",
                            "announcement",
                            "notice",
                        )
                    ]
                return []

        scored = pd.DataFrame(
            [
                {"code": "000001", "total_score": 80.0, "reason": "行业景气", "risk": ""},
                {"code": "000002", "total_score": 75.0, "reason": "质量改善", "risk": ""},
            ]
        )
        config = {
            "top_n": 1,
            "news_factor": {
                "enabled": True,
                "lookback_days": 7,
                "positive_cap": 8,
                "negative_cap": -15,
                "hard_risk_penalty": -20,
                "min_shortlist": 2,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            updated = apply_news_to_scored_frame(
                scored,
                signal_date=date(2026, 8, 12),
                provider=Provider(),
                cache_dir=Path(temporary),
                config=config,
            )

        self.assertEqual("000002", updated.iloc[0]["code"])
        risky = updated.set_index("code").loc["000001"]
        self.assertLessEqual(risky["news_score"], -20.0)
        self.assertIn("重大新闻风险", risky["risk"])
        self.assertTrue(risky["news_hard_risk"])

    def test_payload_contains_exactly_twenty_compatible_ranked_rows(self) -> None:
        """Catches an engine that scores rows but cannot feed the desktop contract."""
        from zhengxi_prosperity_picker_data import build_payload

        signal_date = pd.Timestamp("2026-07-31")
        rows = []
        statements = []
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            trading_dates = pd.bdate_range(end=signal_date, periods=85)
            for index in range(25):
                code = f"{index + 1:06d}"
                industry = f"行业{index % 6}"
                close = pd.Series(
                    [10.0 + day * (0.015 + (index % 4) * 0.002) for day in range(85)]
                )
                bars = pd.DataFrame(
                    {
                        "date": trading_dates,
                        "open": close - 0.03,
                        "high": close + 0.15,
                        "low": close - 0.15,
                        "close": close,
                        "volume": 20_000_000 + index * 100_000,
                        "amount": close * (20_000_000 + index * 100_000),
                    }
                )
                bars.to_csv(cache_dir / f"{code}.csv", index=False)
                rows.append(
                    {
                        "date": signal_date.date(),
                        "code": code,
                        "name": f"测试公司{index + 1}",
                        "industry": industry,
                        "concept": "测试概念",
                        "style": "成长",
                        "new_price": float(close.iloc[-1]),
                        "change_rate": 1.5,
                        "turnoverrate": 2.0,
                        "volume_ratio": 1.2,
                        "deal_amount": float(bars.iloc[-1]["amount"]),
                        "high_price": float(bars.iloc[-1]["high"]),
                        "low_price": float(bars.iloc[-1]["low"]),
                        "pre_close_price": float(close.iloc[-2]),
                        "volume": float(bars.iloc[-1]["volume"]),
                        "pe9": 20.0 + index,
                        "pbnewmrq": 2.0,
                        "roe_weight": 3.0 + index % 8,
                        "netprofit_yoy_ratio": 15.0,
                        "debt_asset_ratio": 35.0,
                        "fund_amount": 20_000_000 + index * 100_000,
                        "fund_amount_super": 8_000_000,
                        "fund_amount_large": 6_000_000,
                        "fund_amount_medium": -3_000_000,
                        "fund_amount_small": -2_000_000,
                        "fund_amount_3": 40_000_000,
                        "fund_amount_5": 80_000_000,
                    }
                )
                statements.extend(
                    [
                        {
                            "code": code,
                            "report_date": "2026-03-31",
                            "announcement_date": "2026-04-30",
                            "revenue_yoy": 8.0,
                            "net_profit_yoy": 10.0,
                            "net_margin": 0.08,
                            "cfo_to_profit": 0.70,
                            "debt_ratio": 0.38,
                            "receivables_to_assets": 0.14,
                            "inventory_to_assets": 0.18,
                        },
                        {
                            "code": code,
                            "report_date": "2026-06-30",
                            "announcement_date": "2026-07-20",
                            "revenue_yoy": 18.0 + index % 5,
                            "net_profit_yoy": 24.0 + index % 7,
                            "net_margin": 0.11,
                            "cfo_to_profit": 1.00,
                            "debt_ratio": 0.35,
                            "receivables_to_assets": 0.12,
                            "inventory_to_assets": 0.16,
                        },
                    ]
                )

            snapshot = {
                "selection_date": signal_date.date(),
                "spot_date": signal_date.date(),
                "fund_date": signal_date.date(),
                "spot_buy": set(),
                "indicator_buy": set(),
                "indicator_sell": set(),
                "strategy_hits": {},
                "rows": rows,
            }
            config = {
                "title": "郑希景气精选",
                "top_n": 20,
                "industry_cap": 4,
                "minimum_deal_amount": 100_000_000,
                "minimum_average_amount_20": 100_000_000,
                "minimum_turnover": 0.3,
                "minimum_history_days": 40,
                "strict_total_score": 70,
                "strict_industry_score": 28,
                "strict_repair_score": 15,
                "reject_name_keywords": ["ST", "*ST", "退"],
                "top_sheet_name": "郑希景气Top",
                "industry_sheet_name": "行业景气Top",
                "repair_sheet_name": "ROE修复Top",
                "risk_sheet_name": "候补与风险",
                "all_sheet_name": "全部郑希候选",
                "rules_note": ["研究辅助，不构成投资建议。"],
            }

            payload = build_payload(
                snapshot,
                pd.DataFrame(statements),
                config,
                cache_dir,
            )

        selected = payload["sheets"]["郑希景气Top"]
        self.assertEqual(20, len(selected))
        self.assertEqual(20, len({row["代码"] for row in selected}))
        self.assertEqual(20, payload["summary"]["输出Top数"])
        self.assertEqual("", payload["summary"]["abstain_reason"])
        self.assertTrue(
            all(
                row["入选层级"] in {"严格入选", "一级候补", "二级候补"}
                for row in selected
            )
        )
        self.assertTrue(
            all(
                "行业景气分" in row
                and "ROE修复代理分" in row
                and "综合评分" in row
                for row in selected
            )
        )


class ZhengxiWorkerIntegrationTests(unittest.TestCase):
    def test_worker_routes_zhengxi_strategy_to_independent_engine(self) -> None:
        """Catches accidentally executing the unchanged generic scorer."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = replace(
                AppPaths.default(),
                data_dir=root / "data",
                engine_dir=ROOT / "engine",
            )
            spec = STRATEGIES["zhengxi_prosperity"]
            captured: list[str] = []

            def fake_run(command, **_kwargs):
                captured.extend(str(value) for value in command)
                raw_path = Path(command[4])
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(
                    json.dumps(
                        {
                            "generated_at": "2026-07-31 16:00:00",
                            "latest_dates": {"综合选股": "2026-07-31"},
                            "strategy_gate_passed": False,
                            "summary": {"abstain_reason": ""},
                            "sheets": {
                                spec.top_sheet: [
                                    {
                                        "代码": f"{index + 1:06d}",
                                        "综合评分": 80 - index,
                                        "名称": f"测试{index + 1}",
                                        "行业": f"行业{index % 5}",
                                        "最新价": 10,
                                        "涨跌幅%": 1,
                                        "今日主力净流入_万元": 100,
                                        "加分理由": "严格入选；行业景气靠前",
                                        "风险提示": "",
                                        "概念": "",
                                        "ATR14": 0.5,
                                        "入选层级": "严格入选",
                                        "行业景气分": 30,
                                        "ROE修复代理分": 15,
                                        "公司质量分": 12,
                                        "流动性分": 8,
                                        "估值时机分": 7,
                                        "证据状态": "修复证据充分",
                                        "新闻评分": -3.0 if index == 0 else 0.0,
                                        "新闻摘要": "风险：股东减持" if index == 0 else "近7日无合格资讯",
                                        "新闻数量": 1 if index == 0 else 0,
                                        "新闻状态": "已评估" if index == 0 else "无合格资讯",
                                        "新闻重大风险": False,
                                        "新闻截止日": "2026-07-31",
                                    }
                                    for index in range(20)
                                ]
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("workers.picker_worker.subprocess.run", side_effect=fake_run):
                result = run_picker(
                    "zhengxi_prosperity",
                    paths,
                    lambda *_args: None,
                    as_of_date="2026-07-31",
                )

        self.assertTrue(captured[3].endswith("zhengxi_prosperity_picker_data.py"))
        self.assertEqual(20, len(result["rows"]))
        self.assertEqual("严格入选", result["rows"][0]["selection_tier"])
        self.assertEqual(30.0, result["rows"][0]["industry_prosperity_score"])
        self.assertEqual(15.0, result["rows"][0]["roe_repair_score"])
        self.assertEqual(-3.0, result["rows"][0]["news_score"])
        self.assertEqual("风险：股东减持", result["rows"][0]["news_summary"])
        self.assertEqual("2026-07-31", result["rows"][0]["news_as_of"])

    def test_high_precision_is_not_a_publicly_runnable_strategy(self) -> None:
        """Catches leaving a hidden runtime route after removing the product entry."""
        with self.assertRaisesRegex(ValueError, "未知策略"):
            run_picker("high_precision", AppPaths.default(), lambda *_args: None)

    def test_regular_strategy_does_not_run_hidden_probability_engine(self) -> None:
        """Catches high-precision work continuing behind the three retained strategies."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = replace(
                AppPaths.default(),
                data_dir=root / "data",
                engine_dir=ROOT / "engine",
            )
            spec = STRATEGIES["all_market"]

            def fake_run(command, **_kwargs):
                raw_path = Path(command[4])
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(
                    json.dumps(
                        {
                            "generated_at": "2026-07-31 16:00:00",
                            "latest_dates": {"综合选股": "2026-07-31"},
                            "strategy_gate_passed": False,
                            "summary": {"abstain_reason": ""},
                            "sheets": {
                                spec.top_sheet: [
                                    {
                                        "代码": "000001",
                                        "综合评分": 70,
                                        "名称": "测试",
                                        "行业": "银行",
                                        "最新价": 10,
                                        "涨跌幅%": 1,
                                        "加分理由": "测试",
                                        "风险提示": "",
                                        "ATR14": 0.5,
                                    }
                                ]
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("workers.picker_worker.subprocess.run", side_effect=fake_run),
                patch(
                    "workers.picker_worker.load_live_probability_context",
                    side_effect=AssertionError("hidden probability engine was called"),
                ),
            ):
                result = run_picker("all_market", paths, lambda *_args: None)

        self.assertEqual("all_market", result["strategy"])


if __name__ == "__main__":
    unittest.main()
