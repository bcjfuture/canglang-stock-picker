from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_v4 import (
    build_parser,
    build_regime_scores,
    classify_market_regime,
    evaluate_v4,
    select_dynamic_candidates,
    v4_deployment_checks,
)


RULE = {
    "good": {
        "breadth_floor": 0.50,
        "momentum_floor": 0.02,
        "drawdown_floor": -0.15,
        "target_count": 5,
        "score": "pullback_score",
    },
    "normal": {
        "breadth_floor": 0.45,
        "momentum_floor": 0.005,
        "drawdown_floor": -0.22,
        "target_count": 3,
        "score": "low_vol_trend_score",
    },
    "poor": {"target_count": 1, "score": "rebound_score"},
}


def market_rows(date: str, count: int, market: tuple[float, float, float]):
    breadth, momentum, drawdown = market
    return [
        {
            "date": pd.Timestamp(date),
            "code": f"{index:06d}",
            "mom_5": 0.01,
            "mom_20": 0.02 + index * 0.001,
            "mom_60": 0.10 + index * 0.002,
            "trend_20_60": 0.03 + index * 0.001,
            "overheat": 0.0,
            "drawdown_60": -0.10,
            "volatility_20": 0.02 - index * 0.0001,
            "breadth_above_ma20": breadth,
            "market_median_mom20": momentum,
            "market_median_drawdown60": drawdown,
        }
        for index in range(1, count + 1)
    ]


class HighPrecisionV4SelectionTests(unittest.TestCase):
    def test_classification_prefers_good_then_normal_else_poor(self) -> None:
        self.assertEqual(
            "good",
            classify_market_regime(
                pd.Series(
                    {
                        "breadth_above_ma20": 0.50,
                        "market_median_mom20": 0.02,
                        "market_median_drawdown60": -0.15,
                    }
                ),
                RULE,
            ),
        )
        self.assertEqual(
            "normal",
            classify_market_regime(
                pd.Series(
                    {
                        "breadth_above_ma20": 0.46,
                        "market_median_mom20": 0.01,
                        "market_median_drawdown60": -0.20,
                    }
                ),
                RULE,
            ),
        )
        self.assertEqual(
            "poor",
            classify_market_regime(
                pd.Series(
                    {
                        "breadth_above_ma20": 0.20,
                        "market_median_mom20": -0.05,
                        "market_median_drawdown60": -0.30,
                    }
                ),
                RULE,
            ),
        )

    def test_dynamic_selection_returns_five_three_one(self) -> None:
        frame = pd.DataFrame(
            market_rows("2026-07-13", 6, (0.60, 0.04, -0.10))
            + market_rows("2026-07-14", 4, (0.47, 0.01, -0.18))
            + market_rows("2026-07-15", 2, (0.20, -0.03, -0.30))
        )

        scored = build_regime_scores(frame)
        selected = select_dynamic_candidates(scored, RULE)

        counts = selected.groupby("market_regime")["code"].size().to_dict()
        self.assertEqual({"good": 5, "normal": 3, "poor": 1}, counts)
        self.assertEqual(
            {"pullback_score", "low_vol_trend_score", "rebound_score"},
            set(selected["selection_score_name"]),
        )

    def test_equal_poor_scores_use_code_ascending(self) -> None:
        rows = market_rows("2026-07-15", 2, (0.20, -0.03, -0.30))
        for row in rows:
            for column in (
                "mom_5", "mom_20", "mom_60", "trend_20_60",
                "overheat", "drawdown_60", "volatility_20",
            ):
                row[column] = 0.0
        selected = select_dynamic_candidates(
            build_regime_scores(pd.DataFrame(rows)), RULE
        )

        self.assertEqual(["000001"], selected["code"].tolist())


class HighPrecisionV4MetricTests(unittest.TestCase):
    def test_optimizer_parser_accepts_artifact_paths(self) -> None:
        args = build_parser().parse_args(
            [
                "optimize",
                "--config", "config.json",
                "--panel-cache", "panel.pkl",
                "--output", "report.json",
                "--candidate-model", "candidate.json",
                "--verification", "verification.json",
                "--production-model", "production.json",
                "--runtime-report", "runtime.json",
            ]
        )

        self.assertEqual("optimize", args.command)
        self.assertEqual(Path("panel.pkl"), args.panel_cache)
        self.assertEqual(Path("runtime.json"), args.runtime_report)

    def test_drawdown_aggregates_same_date_positions(self) -> None:
        selected = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-13", "2026-07-13", "2026-07-14"]),
                "code": ["000001", "000002", "000003"],
                "market_regime": ["good", "good", "poor"],
                "bounded_return_5": [0.10, -0.05, -0.10],
                "exit_reason": ["take_profit", "day_5_close", "stop_loss"],
            }
        )

        metrics = evaluate_v4(
            selected,
            pd.to_datetime(["2026-07-13", "2026-07-14"]),
            position_fraction=0.04,
        )

        self.assertEqual(3, metrics["trades"])
        self.assertEqual(2, metrics["active_dates"])
        self.assertAlmostEqual(-0.004, metrics["max_signal_basket_drawdown"])

    def test_gate_rejects_non_positive_regime_mean(self) -> None:
        metrics = {
            "trades": 2500,
            "win_rate": 0.72,
            "mean_net_return": 0.001,
            "max_signal_basket_drawdown": -0.10,
            "year_metrics": [
                {"year": year, "win_rate": 0.65} for year in range(2022, 2027)
            ],
            "regime_metrics": [
                {"market_regime": "good", "mean_net_return": 0.002},
                {"market_regime": "normal", "mean_net_return": 0.0},
                {"market_regime": "poor", "mean_net_return": 0.001},
            ],
        }
        gates = {
            "minimum_win_rate": 0.68,
            "minimum_trades": 2000,
            "minimum_mean_net_return": 0.0,
            "maximum_signal_basket_drawdown": 0.15,
            "minimum_good_years": 4,
            "minimum_year_win_rate": 0.60,
            "minimum_worst_year_win_rate": 0.60,
            "minimum_positive_neighbors": 2,
        }
        neighbors = [{"win_rate": 0.70, "mean_net_return": 0.001}] * 3

        result = v4_deployment_checks(metrics, neighbors, gates)

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["all_regimes_positive"])


if __name__ == "__main__":
    unittest.main()
