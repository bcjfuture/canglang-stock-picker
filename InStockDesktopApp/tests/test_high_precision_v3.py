from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_v3 import (
    build_pullback_scores,
    calculate_bounded_trade_returns,
    evaluate_v3,
    select_pullback_candidates,
    v3_deployment_checks,
)


class HighPrecisionV3Tests(unittest.TestCase):
    def test_same_day_stop_has_priority_over_take_profit(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-05", periods=7),
                "code": ["000001"] * 7,
                "open": [10.0] * 7,
                "high": [10.0, 10.4, 10.0, 10.0, 10.0, 10.0, 10.0],
                "low": [10.0, 9.3, 10.0, 10.0, 10.0, 10.0, 10.0],
                "close": [10.0] * 7,
            }
        )

        result = calculate_bounded_trade_returns(frame, 0.03, 0.06, 0.003)

        self.assertAlmostEqual(-0.063, result.iloc[0]["bounded_return_5"])
        self.assertEqual("stop_loss", result.iloc[0]["exit_reason"])

    def test_take_profit_and_fifth_close_paths(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-05", periods=8),
                "code": ["000001"] * 8,
                "open": [10.0] * 8,
                "high": [10.0, 10.31, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "low": [10.0] * 8,
                "close": [10.0, 10.0, 10.0, 10.0, 10.0, 10.2, 10.0, 10.0],
            }
        )

        taken = calculate_bounded_trade_returns(frame, 0.03, 0.06, 0.003)
        fallback_source = frame.copy()
        fallback_source["high"] = 10.0
        fallback = calculate_bounded_trade_returns(
            fallback_source, 0.03, 0.06, 0.003
        )

        self.assertAlmostEqual(0.027, taken.iloc[0]["bounded_return_5"])
        self.assertEqual("take_profit", taken.iloc[0]["exit_reason"])
        self.assertAlmostEqual(0.017, fallback.iloc[0]["bounded_return_5"])
        self.assertEqual("day_5_close", fallback.iloc[0]["exit_reason"])

    def test_weak_market_abstains_and_ties_use_code(self) -> None:
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-05")] * 2,
                "code": ["000002", "000001"],
                "mom_60": [0.2, 0.2],
                "trend_20_60": [0.1, 0.1],
                "mom_5": [-0.02, -0.02],
                "overheat": [0.0, 0.0],
                "breadth_above_ma20": [0.4, 0.4],
                "market_median_mom20": [0.03, 0.03],
                "market_median_drawdown60": [-0.1, -0.1],
            }
        )
        rule = {
            "breadth_floor": 0.5,
            "momentum_floor": 0.02,
            "drawdown_floor": -0.15,
            "max_picks": 1,
        }

        weak = select_pullback_candidates(build_pullback_scores(frame), rule)
        frame["breadth_above_ma20"] = 0.6
        strong = select_pullback_candidates(build_pullback_scores(frame), rule)

        self.assertTrue(weak.empty)
        self.assertEqual(["000001"], strong["code"].tolist())

    def test_position_weighted_metrics_and_gates(self) -> None:
        selected = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2022-03-01", "2022-03-02", "2023-03-01", "2023-03-02"]
                ),
                "bounded_return_5": [0.027, 0.027, -0.063, 0.027],
            }
        )
        metrics = evaluate_v3(
            selected,
            pd.bdate_range("2022-03-01", "2023-03-02"),
            position_fraction=0.2,
        )
        gates = {
            "minimum_win_rate": 0.65,
            "minimum_trades": 4,
            "minimum_activity_rate": 0.0,
            "minimum_mean_net_return": 0.0,
            "maximum_position_drawdown": 0.12,
            "minimum_good_years": 1,
            "minimum_year_win_rate": 0.6,
            "minimum_worst_year_win_rate": 0.45,
            "minimum_positive_neighbors": 2,
        }

        passed = v3_deployment_checks(
            metrics,
            [
                {"win_rate": 0.65, "mean_net_return": 0.001},
                {"win_rate": 0.7, "mean_net_return": 0.002},
            ],
            gates,
        )

        self.assertAlmostEqual(0.75, metrics["win_rate"])
        self.assertAlmostEqual(-0.0126, metrics["max_position_drawdown"])
        self.assertTrue(passed["passed"])


if __name__ == "__main__":
    unittest.main()
