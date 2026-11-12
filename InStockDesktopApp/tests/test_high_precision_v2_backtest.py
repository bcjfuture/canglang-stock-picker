from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_v2 import (
    build_parser,
    build_v2_features,
    run_v2_walk_forward_search,
    v2_parameter_grid,
)
from picker_backtest import build_symbol_panel


def market_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=260)
    frames = []
    for index in range(5):
        base = 10.0 + index
        close = (
            base
            + np.sin(np.arange(len(dates)) / (5.0 + index)) * 0.35
            + np.arange(len(dates)) * (0.004 + index * 0.0005)
        )
        volume = 1_000_000 + np.arange(len(dates)) * (500 + index * 50)
        bars = pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.998,
                "high": close * 1.012,
                "low": close * 0.988,
                "close": close,
                "volume": volume,
                "amount": close * volume,
            }
        )
        frames.append(
            build_symbol_panel(
                f"{index + 1:06d}",
                bars,
                {
                    "name": f"股票{index}",
                    "industry": f"行业{index % 3}",
                    "concept": "",
                },
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    panel["above_ma20"] = (panel["close"] > panel["ma_20"]).astype(float)
    grouped = panel.groupby("date")
    panel["breadth_above_ma20"] = grouped["above_ma20"].transform("mean")
    panel["market_median_mom20"] = grouped["mom_20"].transform("median")
    panel["market_median_drawdown60"] = grouped["drawdown_60"].transform(
        "median"
    )
    return panel


def benchmark() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=260)
    close = 4000.0 + np.arange(len(dates)) * 1.5
    return pd.DataFrame(
        {
            "date": dates,
            "code": "000300",
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1_000_000,
            "amount": close * 1_000_000,
        }
    )


def fundamentals() -> pd.DataFrame:
    rows = []
    for code in [f"{index + 1:06d}" for index in range(5)]:
        rows.append(
            {
                "code": code,
                "report_date": pd.Timestamp("2023-12-31"),
                "announcement_date": pd.Timestamp("2024-03-28"),
                "net_profit_yoy": 10.0,
                "revenue_yoy": 8.0,
                "net_margin": 0.1,
                "cfo_to_profit": 1.1,
                "debt_ratio": 40.0,
                "cash_to_assets": 0.2,
                "receivables_to_assets": 0.1,
                "inventory_to_assets": 0.05,
                "total_assets_yoy": 6.0,
            }
        )
    return pd.DataFrame(rows)


def tiny_config() -> dict:
    config = json.loads(
        (ROOT / "engine" / "high_precision_v2_config.json").read_text(
            encoding="utf-8"
        )
    )
    config.update(
        {
            "calibration_trading_days": 30,
            "calibration_minimum_activity_rate": 0.0,
            "calibration_minimum_trades": 0,
            "minimum_training_months": 0,
            "l2_values": [1.0],
            "ensemble_weights": [
                {
                    "name": "equal",
                    "trend": 1 / 3,
                    "reversal": 1 / 3,
                    "quality": 1 / 3,
                }
            ],
            "strong_thresholds": [0.5],
            "normal_threshold_deltas": [0.0],
            "max_picks_values": [1],
            "market_policies": [
                {
                    "breadth_floor": 0.0,
                    "momentum_floor": -1.0,
                    "drawdown_floor": -1.0,
                }
            ],
        }
    )
    return config


class HighPrecisionV2BacktestTests(unittest.TestCase):
    def test_feature_builder_adds_ranked_groups_and_forward_label(self) -> None:
        result = build_v2_features(
            market_panel(),
            benchmark(),
            fundamentals(),
            pd.DatetimeIndex(sorted(market_panel()["date"].unique())),
        )

        self.assertIn("label_5", result)
        self.assertIn("breadth_above_ma20_raw", result)
        self.assertTrue(result["label_5"].dropna().isin([0, 1]).all())

    def test_feature_builder_recalculates_v2_holding_period(self) -> None:
        source = market_panel()
        source["forward_5"] = 999.0

        result = build_v2_features(
            source,
            benchmark(),
            fundamentals(),
            pd.DatetimeIndex(sorted(source["date"].unique())),
        )
        first = result[
            (result["code"] == "000001") & result["forward_5"].notna()
        ].iloc[0]
        stock = source[source["code"] == "000001"].sort_values("date")
        position = stock.index.get_loc(first.name) if first.name in stock.index else None

        self.assertNotEqual(999.0, first["forward_5"])

    def test_parameter_grid_is_finite_and_exact(self) -> None:
        config = tiny_config()

        grid = v2_parameter_grid(config)

        self.assertEqual(1, len(grid))
        self.assertEqual(1.0, grid[0]["l2"])

    def test_walk_forward_strictly_separates_dates(self) -> None:
        dates = sorted(market_panel()["date"].unique())
        folds = [
            {
                "train_end": pd.Timestamp(dates[189]).date().isoformat(),
                "valid_start": pd.Timestamp(dates[190]).date().isoformat(),
                "valid_end": pd.Timestamp(dates[239]).date().isoformat(),
            }
        ]

        result = run_v2_walk_forward_search(
            market_panel(),
            benchmark(),
            fundamentals(),
            tiny_config(),
            folds,
        )

        fold = result["walk_forward"]["folds"][0]
        self.assertLess(fold["fit_max_date"], fold["calibration_min_date"])
        self.assertLess(
            fold["calibration_max_date"], fold["validation_min_date"]
        )
        self.assertEqual(1, result["attempt_count"])

    def test_optimizer_parser_accepts_artifact_paths(self) -> None:
        args = build_parser().parse_args(
            [
                "optimize",
                "--panel-cache",
                "panel.pkl",
                "--fundamental-cache",
                "fundamentals.pkl",
                "--output",
                "report.json",
                "--candidate-model",
                "candidate.json",
                "--verification",
                "verification.json",
                "--production-model",
                "production.json",
            ]
        )

        self.assertEqual(Path("panel.pkl"), args.panel_cache)
        self.assertEqual(Path("fundamentals.pkl"), args.fundamental_cache)
        self.assertEqual(Path("production.json"), args.production_model)


if __name__ == "__main__":
    unittest.main()
