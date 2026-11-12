from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from workers.sector_forecast_core import (
    FEATURE_COLUMNS,
    build_feature_panel,
    build_live_rows,
    explain_row,
    fit_ridge,
    rank_latest,
    walk_forward_validate,
)
import workers.sector_forecast_core as sector_core


def make_history(days: int = 90, sectors: int = 1) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=days)
    rows: list[dict[str, object]] = []
    for sector in range(sectors):
        sector_daily = 0.001 + sector * 0.00025
        for index, day in enumerate(dates):
            close = 100.0 * (1.0 + sector_daily) ** index
            benchmark = 100.0 * 1.0005**index
            rows.append(
                {
                    "sector_type": "industry",
                    "sector_id": f"BK{sector:04d}",
                    "sector_name": f"板块{sector}",
                    "date": day,
                    "open": close * 0.998,
                    "high": close * 1.012,
                    "low": close * 0.988,
                    "close": close,
                    "volume": 1_000_000 + index * 1_000,
                    "amount": close * (1_000_000 + index * 1_000),
                    "turnover": 1.5 + sector * 0.02 + index * 0.001,
                    "benchmark_close": benchmark,
                }
            )
    return pd.DataFrame(rows)


def make_signal_panel(days: int = 620, sectors: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=days)
    rows: list[dict[str, object]] = []
    midpoint = (sectors - 1) / 2.0
    for day_index, day in enumerate(dates):
        for sector in range(sectors):
            signal = (sector - midpoint) / max(midpoint, 1.0)
            row: dict[str, object] = {
                "sector_type": "concept",
                "sector_id": f"BK{sector:04d}",
                "sector_name": f"板块{sector}",
                "date": day,
                "target_excess_5": 0.02 if sector >= sectors - 5 else -0.02,
                "current_data_completeness": 1.0,
                "history_days": days,
            }
            for column in FEATURE_COLUMNS:
                row[column] = 0.0
            row["ret_60"] = signal
            row["ret_20"] = signal
            row["ret_5"] = -signal + (day_index % 7) * 0.001
            rows.append(row)
    return pd.DataFrame(rows)


class SectorFeatureTests(unittest.TestCase):
    def test_target_uses_fifth_future_close_and_benchmark(self) -> None:
        panel = build_feature_panel(make_history(days=90))
        row = panel.iloc[40]
        expected = (1.001**5 - 1.0) - (1.0005**5 - 1.0)
        self.assertAlmostEqual(float(row["target_excess_5"]), expected, places=10)

    def test_live_row_does_not_require_future_target(self) -> None:
        panel = build_feature_panel(make_history(days=90))
        live = build_live_rows(panel, panel["date"].max())
        self.assertEqual(len(live), 1)
        self.assertTrue(pd.isna(live.iloc[0]["target_excess_5"]))

    def test_feature_windows_do_not_change_when_future_is_mutated(self) -> None:
        history = make_history(days=90)
        cutoff = history["date"].iloc[60]
        before = build_feature_panel(history)
        before = before.loc[before["date"] == cutoff, list(FEATURE_COLUMNS)]
        history.loc[history["date"] > cutoff, "close"] *= 10
        after = build_feature_panel(history)
        after = after.loc[after["date"] == cutoff, list(FEATURE_COLUMNS)]
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


class SectorModelTests(unittest.TestCase):
    def test_foldwise_outperformance_requires_three_baseline_wins(self) -> None:
        gate = getattr(sector_core, "stable_outperformance", None)
        self.assertIsNotNone(gate)

        passed, superior_folds = gate(
            candidate_mean=0.02,
            baseline_mean=0.01,
            candidate_folds=[0.10, 0.0, 0.0, 0.0],
            baseline_folds=[0.01, 0.01, 0.01, 0.01],
        )

        self.assertEqual(superior_folds, 1)
        self.assertFalse(passed)

    def test_ranking_uses_three_five_ten_day_momentum_instead_of_twenty_day_leader(self) -> None:
        panel = make_signal_panel()
        live_day = panel["date"].max()
        live = panel["date"] == live_day
        sector_number = panel.loc[live, "sector_id"].str[-4:].astype(int)
        short_signal = 1.0 - 2.0 * sector_number / 11.0
        panel.loc[live, "ret_3"] = short_signal.to_numpy()
        panel.loc[live, "ret_5"] = short_signal.to_numpy()
        panel.loc[live, "ret_10"] = short_signal.to_numpy()
        panel.loc[live, "ret_20"] = -short_signal.to_numpy()
        validation = {
            "passed": False,
            "head_passed": False,
            "quality": 0.5,
        }

        rows = rank_latest(panel, validation, top_k=10)

        self.assertEqual(rows[0]["sector_id"], "BK0000")

    def test_failed_validation_labels_ranked_rows_as_observations(self) -> None:
        panel = make_signal_panel()
        validation = {
            "passed": False,
            "head_passed": False,
            "quality": 0.5,
        }

        rows = rank_latest(panel, validation, top_k=10)

        self.assertTrue(rows)
        self.assertEqual({"观察"}, {row["strength"] for row in rows})

    def test_head_only_validation_labels_rows_after_fifth_as_observations(self) -> None:
        panel = make_signal_panel()
        validation = {
            "passed": False,
            "head_passed": True,
            "head_selected_mode": "factor",
            "quality": 0.7,
        }

        rows = rank_latest(panel, validation, top_k=10)

        self.assertNotEqual(rows[0]["strength"], "观察")
        self.assertEqual({"观察"}, {row["strength"] for row in rows[5:]})

    def test_ridge_model_produces_finite_predictions(self) -> None:
        panel = build_feature_panel(make_history(days=620, sectors=12))
        model = fit_ridge(panel.dropna(subset=["target_excess_5"]))
        prediction = model.predict(panel.tail(12))
        self.assertEqual(len(prediction), 12)
        self.assertTrue(np.isfinite(prediction).all())

    def test_model_prediction_does_not_use_twenty_or_sixty_day_returns_for_ranking(self) -> None:
        values = np.linspace(-1.0, 1.0, 40)
        train = pd.DataFrame(
            {column: np.zeros(len(values), dtype=float) for column in FEATURE_COLUMNS}
        )
        train["ret_20"] = values
        train["ret_60"] = values
        train["trend_20"] = values
        train["target_excess_5"] = values
        model = fit_ridge(train)
        candidates = pd.DataFrame(
            {column: [0.0, 0.0] for column in FEATURE_COLUMNS}
        )
        candidates["ret_20"] = [-1.0, 1.0]
        candidates["ret_60"] = [-1.0, 1.0]
        candidates["trend_20"] = [-1.0, 1.0]

        prediction = model.predict(candidates)

        self.assertAlmostEqual(float(prediction[0]), float(prediction[1]), places=12)

    def test_walk_forward_uses_five_day_embargo(self) -> None:
        panel = build_feature_panel(make_history(days=620, sectors=12))
        validation = walk_forward_validate(panel)
        self.assertGreaterEqual(len(validation["folds"]), 4)
        dates = sorted(pd.to_datetime(panel["date"].unique()))
        positions = {pd.Timestamp(value): index for index, value in enumerate(dates)}
        for fold in validation["folds"]:
            gap = positions[pd.Timestamp(fold["validation_start"])] - positions[pd.Timestamp(fold["train_end"])]
            self.assertGreaterEqual(gap, 6)

    def test_walk_forward_can_select_pure_model_candidate(self) -> None:
        panel = make_signal_panel()
        panel["ret_20"] = -panel["ret_60"]
        validation = walk_forward_validate(panel)
        self.assertEqual(validation["selected_mode"], "model")

    def test_top_five_can_be_validated_separately_from_full_top_ten(self) -> None:
        panel = make_signal_panel()
        panel["ret_20"] = -panel["ret_20"]
        validation = walk_forward_validate(panel)
        self.assertTrue(validation["head_passed"])
        self.assertFalse(validation["passed"])

    def test_short_model_is_rejected_when_legacy_twenty_day_ranking_is_better(self) -> None:
        panel = make_signal_panel()
        date_number = panel["date"].map(
            {day: index for index, day in enumerate(sorted(panel["date"].unique()))}
        )
        sector_number = panel["sector_id"].str[-4:].astype(int)
        rotating_noise = ((sector_number + date_number) % 12) / 11.0
        for column in FEATURE_COLUMNS:
            panel[column] = rotating_noise
        panel["ret_20"] = sector_number / 11.0

        validation = walk_forward_validate(panel)

        self.assertIn("legacy_outperformance", validation["reasons"])
        self.assertIn("legacy_outperformance", validation["head_reasons"])
        self.assertFalse(validation["passed"])
        self.assertFalse(validation["head_passed"])

    def test_failed_validation_caps_current_context_to_five_percent(self) -> None:
        panel = build_feature_panel(make_history(days=620, sectors=12))
        live_day = panel["date"].max()
        panel.loc[panel["date"] == live_day, "fund_score"] = np.arange(12)[::-1]
        panel.loc[panel["date"] == live_day, "breadth_score"] = np.arange(12)[::-1]
        panel.loc[panel["date"] == live_day, "heat_score"] = np.arange(12)[::-1]
        validation = {
            "passed": False,
            "head_passed": False,
            "selected_mode": "factor",
            "quality": 0.5,
            "model_weight": 0.0,
        }
        rows = rank_latest(panel, validation, top_k=10)
        self.assertTrue(rows)
        self.assertTrue(all(row["context_weight"] == 0.05 for row in rows))

    def test_low_confidence_rows_are_not_used_to_fill_top_ten(self) -> None:
        panel = build_feature_panel(make_history(days=620, sectors=4))
        live_day = panel["date"].max()
        panel.loc[panel["date"] == live_day, "current_data_completeness"] = [0.95, 0.90, 0.20, 0.10]
        validation = {
            "passed": True,
            "quality": 0.85,
            "model_metrics": {"top10_mean_excess": 0.01, "rank_ic": 0.10},
            "baseline_metrics": {"top10_mean_excess": 0.0, "rank_ic": 0.0},
            "folds": [],
        }
        rows = rank_latest(panel, validation, top_k=10)
        self.assertLess(len(rows), 4)
        self.assertTrue(all(int(row["confidence"]) >= 45 for row in rows))

    def test_explanations_use_actual_feature_values(self) -> None:
        reasons, risks = explain_row(
            pd.Series(
                {
                    "ret_3": 0.06,
                    "ret_5": 0.08,
                    "ret_10": 0.10,
                    "ret_20": 0.18,
                    "amount_ratio_5_20": 0.35,
                    "fund_score": 1.4,
                    "volatility_20": 0.08,
                    "drawdown_60": -0.02,
                    "breadth_score": 0.5,
                }
            )
        )
        self.assertIn("3/5/10日短期动量共振", reasons)
        self.assertNotIn("20日相对趋势领先", reasons)
        self.assertIn("短期波动偏高", risks)
        self.assertIn("阶段涨幅较大，存在拥挤回撤风险", risks)


if __name__ == "__main__":
    unittest.main()
