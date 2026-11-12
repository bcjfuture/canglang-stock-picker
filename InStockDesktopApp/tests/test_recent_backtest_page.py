from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import instock_app.ui.backtest_page as backtest_page
from instock_app.ui.backtest_page import build_view_model


class RecentBacktestPageTests(unittest.TestCase):
    def test_view_model_marks_evaluation_and_snapshot_fallback_dates(self) -> None:
        result = {
            "requested_evaluation_date": "2026-08-03",
            "evaluation_date": "2026-07-31",
            "offsets": {
                "3": {
                    "label": "前4个交易日（目标前3日）",
                    "requested_signal_date": "2026-07-28",
                    "signal_date": "2026-07-27",
                    "holding_trading_days": 4,
                    "snapshot_fallback": True,
                    "profiles": {},
                }
            },
        }

        model = build_view_model(result)

        self.assertIn(
            "evaluation_date_display",
            model,
            "evaluation fallback is not exposed to the page",
        )
        self.assertEqual(
            "2026-07-31（数据最新 2026-08-03）",
            model["evaluation_date_display"],
        )
        self.assertIn(
            "signal_date_display",
            model["horizons"][0],
            "snapshot fallback is not exposed to the page",
        )
        self.assertEqual(
            "2026-07-27（目标 2026-07-28，实际前4日）",
            model["horizons"][0]["signal_date_display"],
        )

    def test_build_view_model_formats_recent_metrics(self) -> None:
        result = {
            "evaluation_date": "2026-07-16",
            "cost": 0.003,
            "offsets": {
                "1": {
                    "label": "前1个交易日",
                    "signal_date": "2026-07-15",
                    "profiles": {
                        "all_market": {
                            "status": "ok",
                            "title": "全市场精选",
                            "summary": {
                                "mean_net_return": 0.023,
                                "median_net_return": 0.018,
                                "win_rate": 0.6,
                                "evaluated_count": 10,
                                "unavailable_count": 2,
                                "benchmark_return": 0.012,
                                "excess_return": 0.011,
                                "buckets": {
                                    "top5": {
                                        "mean_net_return": 0.04,
                                        "win_rate": 0.8,
                                        "excess_return": 0.028,
                                    },
                                    "rank6_20": {
                                        "mean_net_return": 0.01,
                                        "win_rate": 0.5,
                                        "excess_return": -0.002,
                                    },
                                    "baseline_top5": {
                                        "mean_net_return": 0.015,
                                        "win_rate": 0.4,
                                        "excess_return": 0.003,
                                    },
                                },
                                "head_rerank_gate_passed": True,
                            },
                            "details": [{"code": "000001"}],
                        }
                    },
                }
            },
        }

        model = build_view_model(result)

        self.assertEqual("2026-07-16", model["evaluation_date"])
        horizon = model["horizons"][0]
        self.assertEqual("前1个交易日", horizon["label"])
        profile = horizon["profiles"][0]
        self.assertEqual("+2.30%", profile["mean"])
        self.assertEqual("60.0%", profile["win_rate"])
        self.assertEqual("+1.10%", profile["excess"])
        self.assertEqual("10只有效 · 2只缺行情", profile["count"])
        self.assertEqual("80.0%", profile["top5_win_rate"])
        self.assertEqual("+4.00%", profile["top5_mean"])
        self.assertEqual("50.0%", profile["tail_win_rate"])
        self.assertEqual("40.0%", profile["baseline_top5_win_rate"])
        self.assertTrue(profile["head_rerank_gate_passed"])

    def test_build_view_model_formats_failed_profile(self) -> None:
        result = {
            "evaluation_date": "2026-07-16",
            "offsets": {
                "3": {
                    "label": "前3个交易日",
                    "signal_date": "2026-07-13",
                    "profiles": {
                        "tech_attack": {
                            "status": "error",
                            "title": "科技进攻",
                            "error": "历史快照缺失",
                            "summary": {},
                            "details": [],
                        }
                    },
                }
            },
        }

        profile = build_view_model(result)["horizons"][0]["profiles"][0]

        self.assertEqual("失败", profile["mean"])
        self.assertEqual("历史快照缺失", profile["count"])
        self.assertEqual([], profile["details"])


class ProfileCardColorTests(unittest.TestCase):
    def profile(self, **changes):
        base = {
            "title": "科技进攻", "status": "ok", "mean": "+1.20%", "win_rate": "60.0%",
            "excess": "-0.40%", "top5_mean": "+0.00%", "top5_win_rate": "50.0%",
            "top5_excess": "—", "count": "20只有效 · 0只缺行情",
        }
        base.update(changes)
        return base

    def test_gains_and_losses_get_their_own_tags(self) -> None:
        """Catches a whole card line taking one color when the figures disagree."""
        tags = dict(backtest_page.profile_segments(self.profile(), "科技进攻"))
        self.assertEqual("up", tags["+1.20%"])
        self.assertEqual("down", tags["-0.40%"])

    def test_flat_and_missing_values_stay_plain(self) -> None:
        tags = dict(backtest_page.profile_segments(self.profile(), "科技进攻"))
        self.assertIsNone(tags["+0.00%"])
        self.assertIsNone(tags["—"])

    def test_missing_profile_and_failure_read_plainly(self) -> None:
        self.assertEqual([("科技进攻  无结果", None)], backtest_page.profile_segments(None, "科技进攻"))
        failed = backtest_page.profile_segments(self.profile(status="error", count="行情缺失"), "科技进攻")
        self.assertIn("失败", failed[0][0])

if __name__ == "__main__":
    unittest.main()
