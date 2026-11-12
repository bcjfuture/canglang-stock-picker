from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_v5 import (
    _recent_validation,
    attach_benchmark_returns,
    build_v5_scores,
    calculate_horizon_returns,
    evaluate_v5,
    evaluate_v5_probability_calibration,
    apply_v5_probability_models,
    fit_v5_probability_models,
    fit_v5_regime_models,
    score_v5_with_model,
    select_v5_candidates,
    v5_deployment_checks,
)


def market_rows(date: str, count: int, market: tuple[float, float, float]):
    breadth, momentum, drawdown = market
    return [
        {
            "date": pd.Timestamp(date),
            "code": f"{index:06d}",
            "mom_5": 0.01,
            "mom_20": 0.02 + index * 0.001,
            "mom_60": 0.08 + index * 0.002,
            "trend_20_60": 0.02 + index * 0.001,
            "overheat": 0.0,
            "drawdown_60": -0.12 + index * 0.001,
            "volatility_20": 0.03 - index * 0.0002,
            "liquidity_20": 1_000_000 + index,
            "breadth_above_ma20": breadth,
            "market_median_mom20": momentum,
            "market_median_drawdown60": drawdown,
        }
        for index in range(1, count + 1)
    ]


RULE = {
    "good": {"breadth_floor": 0.50, "momentum_floor": 0.02, "drawdown_floor": -0.15, "target_count": 5, "score": "growth_score"},
    "normal": {"breadth_floor": 0.45, "momentum_floor": 0.005, "drawdown_floor": -0.22, "target_count": 3, "score": "balanced_score"},
    "poor": {"target_count": 1, "score": "defensive_score"},
}


class V5LabelTests(unittest.TestCase):
    def test_horizons_use_next_open_and_same_horizon_close(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "close": [11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
            }
        )

        result = calculate_horizon_returns(bars, horizons=(1, 3, 5), cost=0.003)

        self.assertAlmostEqual(22.0 / 20.0 - 1.0 - 0.003, result.loc[0, "return_1"])
        self.assertAlmostEqual(44.0 / 20.0 - 1.0 - 0.003, result.loc[0, "return_3"])
        self.assertAlmostEqual(66.0 / 20.0 - 1.0 - 0.003, result.loc[0, "return_5"])

    def test_benchmark_is_aligned_by_signal_date_and_excess_is_added(self) -> None:
        panel = pd.DataFrame(
            {"date": pd.to_datetime(["2026-01-05", "2026-01-06"]), "return_1": [0.02, -0.01]}
        )
        benchmark = pd.DataFrame(
            {"date": pd.to_datetime(["2026-01-05", "2026-01-06"]), "benchmark_return_1": [0.005, -0.02]}
        )

        result = attach_benchmark_returns(panel, benchmark, horizons=(1,))

        self.assertAlmostEqual(0.015, result.loc[0, "excess_1"])
        self.assertAlmostEqual(0.01, result.loc[1, "excess_1"])


class V5SelectionTests(unittest.TestCase):
    def test_weak_market_prefers_capital_preservation_over_deep_rebound(self) -> None:
        rows = market_rows("2026-07-15", 2, (0.25, -0.04, -0.30))
        falling, defensive = rows
        falling.update({"code": "000001", "mom_5": -0.20, "mom_20": -0.25, "mom_60": -0.30, "trend_20_60": -0.15, "drawdown_60": -0.45, "volatility_20": 0.12})
        defensive.update({"code": "000002", "mom_5": 0.00, "mom_20": 0.02, "mom_60": 0.04, "trend_20_60": 0.01, "drawdown_60": -0.08, "volatility_20": 0.02})

        selected = select_v5_candidates(build_v5_scores(pd.DataFrame(rows)), RULE)

        self.assertEqual(["000002"], selected["code"].tolist())


class V5ProbabilityTests(unittest.TestCase):
    @staticmethod
    def probability_rows() -> pd.DataFrame:
        rows = []
        for regime_index, regime in enumerate(("good", "normal", "poor")):
            for section, date_value, count in (
                ("fit", "2023-09-01", 80),
                ("calibration", "2024-06-03", 40),
                ("validation", "2025-06-03", 40),
            ):
                for index, feature in enumerate(np.linspace(0.01, 0.99, count)):
                    rows.append(
                        {
                            "date": pd.Timestamp(date_value) + pd.Timedelta(days=index),
                            "code": f"{regime_index}{index:05d}",
                            "market_regime": regime,
                            "mom_20_rank": float(feature),
                            "return_5": 0.03 if feature > 0.45 + regime_index * 0.05 else -0.02,
                            "section": section,
                        }
                    )
        return pd.DataFrame(rows)

    def test_fitted_probability_varies_by_stock_features(self) -> None:
        rows = self.probability_rows()
        probability_model = fit_v5_probability_models(
            rows,
            fit_end="2023-12-31",
            calibration_start="2024-01-01",
            calibration_end="2024-12-31",
            l2=1.0,
            feature_names=("mom_20_rank",),
            bins=4,
        )
        live = pd.DataFrame(
            {
                "market_regime": ["poor", "poor"],
                "mom_20_rank": [0.05, 0.95],
            }
        )

        result = apply_v5_probability_models(live, probability_model)

        self.assertNotEqual(
            result.loc[0, "calibrated_probability"],
            result.loc[1, "calibrated_probability"],
        )
        self.assertTrue(result["calibrated_probability"].between(0.0, 1.0).all())

    def test_calibration_preserves_continuous_per_stock_differences(self) -> None:
        rows = self.probability_rows()
        probability_model = fit_v5_probability_models(
            rows,
            fit_end="2023-12-31",
            calibration_start="2024-01-01",
            calibration_end="2024-12-31",
            l2=1.0,
            feature_names=("mom_20_rank",),
            bins=4,
        )
        live = pd.DataFrame(
            {
                "market_regime": ["poor"] * 8,
                "mom_20_rank": np.linspace(0.46, 0.53, 8),
            }
        )

        result = apply_v5_probability_models(live, probability_model)

        self.assertEqual(8, result["calibrated_probability"].nunique())

    def test_probability_fit_does_not_read_validation_labels(self) -> None:
        rows = self.probability_rows()
        first = fit_v5_probability_models(
            rows,
            fit_end="2023-12-31",
            calibration_start="2024-01-01",
            calibration_end="2024-12-31",
            l2=1.0,
            feature_names=("mom_20_rank",),
            bins=4,
        )
        changed = rows.copy()
        changed.loc[changed["date"] >= pd.Timestamp("2025-01-01"), "return_5"] *= -1
        second = fit_v5_probability_models(
            changed,
            fit_end="2023-12-31",
            calibration_start="2024-01-01",
            calibration_end="2024-12-31",
            l2=1.0,
            feature_names=("mom_20_rank",),
            bins=4,
        )

        self.assertEqual(first, second)

    def test_dynamic_selection_keeps_five_three_one_contract(self) -> None:
        frame = pd.DataFrame(
            market_rows("2026-07-13", 6, (0.60, 0.04, -0.10))
            + market_rows("2026-07-14", 4, (0.47, 0.01, -0.18))
            + market_rows("2026-07-15", 2, (0.20, -0.03, -0.30))
        )

        selected = select_v5_candidates(build_v5_scores(frame), RULE)

        self.assertEqual({"good": 5, "normal": 3, "poor": 1}, selected.groupby("market_regime")["code"].size().to_dict())

    def test_fitted_model_does_not_read_labels_after_training_cutoff(self) -> None:
        rows = market_rows("2024-12-20", 3, (0.20, -0.03, -0.30))
        future = market_rows("2025-01-02", 3, (0.20, -0.03, -0.30))
        frame = build_v5_scores(pd.DataFrame(rows + future))
        frame["market_regime"] = "poor"
        frame["target_mean_excess"] = [0.01, 0.02, 0.03, 100.0, 100.0, 100.0]
        frame["target_beat_rate"] = [0.0, 1.0, 1.0, -100.0, -100.0, -100.0]

        first = fit_v5_regime_models(frame, "2024-12-20", ridge=10.0)
        frame.loc[frame["date"] > pd.Timestamp("2024-12-20"), ["target_mean_excess", "target_beat_rate"]] = 9999.0
        second = fit_v5_regime_models(frame, "2024-12-20", ridge=10.0)

        self.assertEqual(first, second)

    def test_model_scoring_uses_poor_regime_beat_objective_coefficients(self) -> None:
        rows = market_rows("2026-07-15", 2, (0.20, -0.03, -0.30))
        rows[0]["mom_20"] = -0.20
        rows[1]["mom_20"] = 0.05
        model = {
            "rule": RULE,
            "feature_names": ["mom_20_rank"],
            "regime_models": {
                "good": {"coefficients": [1.0]},
                "normal": {"coefficients": [1.0]},
                "poor": {"coefficients": [1.0]},
            },
        }

        scored = score_v5_with_model(pd.DataFrame(rows), model)
        selected = select_v5_candidates(scored, RULE)

        self.assertEqual(["000002"], selected["code"].tolist())


class V5DeploymentTests(unittest.TestCase):
    def test_recent_validation_uses_latest_fully_observable_signal_date(self) -> None:
        selected = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-15", "2026-07-16"]),
                "code": ["000001", "000002"],
                "return_1": [0.03, np.nan],
                "benchmark_return_1": [0.01, -0.02],
            }
        )
        benchmark_dates = pd.to_datetime(["2026-07-15", "2026-07-16", "2026-07-17"])

        result = _recent_validation(selected, benchmark_dates, horizons=(1,))

        self.assertEqual("2026-07-15", result["1"]["signal_date"])
        self.assertEqual("2026-07-16", result["1"]["evaluation_date"])
        self.assertAlmostEqual(0.03, result["1"]["mean_net_return"])
        self.assertTrue(result["1"]["beat_benchmark"])

    def test_probability_evaluation_reports_brier_and_separation(self) -> None:
        scored = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02"] * 4),
                "return_5": [-0.02, -0.01, 0.01, 0.03],
                "calibrated_probability": [0.2, 0.3, 0.7, 0.8],
            }
        )

        result = evaluate_v5_probability_calibration(scored, "2025-01-01")

        self.assertAlmostEqual(0.065, result["brier_score"])
        self.assertEqual(4, result["rows"])
        self.assertEqual(4, result["unique_probabilities"])
        self.assertGreater(result["high_probability_win_rate"], result["low_probability_win_rate"])

    def test_evaluation_reports_daily_basket_excess_for_every_horizon(self) -> None:
        selected = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03"]),
                "code": ["000001", "000002", "000003"],
                "market_regime": ["good", "good", "poor"],
                "return_1": [0.02, 0.00, -0.01],
                "return_3": [0.04, 0.02, 0.01],
                "return_5": [0.06, 0.04, 0.02],
                "benchmark_return_1": [0.005, 0.005, -0.02],
                "benchmark_return_3": [0.01, 0.01, 0.00],
                "benchmark_return_5": [0.02, 0.02, 0.01],
            }
        )

        metrics = evaluate_v5(selected, position_fraction=0.04)

        self.assertEqual(3, metrics["trades"])
        self.assertAlmostEqual(0.0075, metrics["horizon_metrics"]["1"]["mean_excess_return"])
        self.assertEqual(1.0, metrics["horizon_metrics"]["5"]["benchmark_beat_rate"])

    def test_gate_rejects_any_required_horizon_without_excess(self) -> None:
        metrics = {
            "horizon_metrics": {
                "1": {"mean_excess_return": 0.001, "benchmark_beat_rate": 0.55, "mean_net_return": 0.002},
                "3": {"mean_excess_return": -0.0001, "benchmark_beat_rate": 0.54, "mean_net_return": 0.001},
                "5": {"mean_excess_return": 0.002, "benchmark_beat_rate": 0.56, "mean_net_return": 0.003},
            },
            "trades": 2500,
            "max_signal_basket_drawdown": -0.10,
        }
        gates = {
            "minimum_trades": 2000,
            "minimum_mean_excess_return": 0.0,
            "minimum_benchmark_beat_rate": 0.52,
            "minimum_mean_net_return": 0.0,
            "maximum_signal_basket_drawdown": 0.15,
        }

        result = v5_deployment_checks(metrics, gates)

        self.assertFalse(result["passed"])
        self.assertIn("horizon_3_mean_excess_return", result["failed"])


if __name__ == "__main__":
    unittest.main()
