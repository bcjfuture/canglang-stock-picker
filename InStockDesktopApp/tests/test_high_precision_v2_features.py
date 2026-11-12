from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_v2 import (
    QUALITY_FEATURES,
    REVERSAL_FEATURES,
    TREND_FEATURES,
    attach_benchmark_features,
    attach_industry_features,
    build_price_features,
    build_v2_features,
)
from picker_backtest import build_symbol_panel


def bars(multiplier: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=100)
    trend = np.linspace(10.0, 20.0, len(dates)) * multiplier
    wave = np.sin(np.arange(len(dates)) / 5.0) * 0.25
    close = trend + wave
    volume = np.linspace(1_000_000, 2_000_000, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.997,
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "volume": volume,
            "amount": close * volume,
        }
    )


def panel() -> pd.DataFrame:
    frames = []
    for code, industry, multiplier in (
        ("000001", "软件", 1.0),
        ("000002", "软件", 0.95),
        ("000003", "银行", 0.85),
    ):
        frames.append(
            build_symbol_panel(
                code,
                bars(multiplier),
                {"name": code, "industry": industry, "concept": ""},
            )
        )
    return pd.concat(frames, ignore_index=True)


def benchmark() -> pd.DataFrame:
    frame = bars(1.1)
    frame["code"] = "000300"
    return frame


class HighPrecisionV2FeatureTests(unittest.TestCase):
    def test_future_price_mutation_does_not_change_earlier_features(self) -> None:
        first_panel = panel()
        second_panel = copy.deepcopy(first_panel)
        cutoff = pd.Timestamp("2025-04-30")
        mask = second_panel["date"] > cutoff
        second_panel.loc[mask, ["open", "high", "low", "close"]] *= 4.0

        first = build_price_features(first_panel)
        second = build_price_features(second_panel)
        columns = [
            "mom_3",
            "mom_10",
            "volume_accel_5_20",
            "amount_accel_5_20",
            "up_day_ratio_20",
            "downside_volatility_20",
            "trend_slope_20",
            "trend_r2_20",
            "rsi_14",
            "bollinger_position_20",
            "opening_gap",
            "intraday_close_strength",
            "volume_price_corr_20",
            "max_return_5",
            "min_return_5",
            "atr_contraction",
        ]

        pd.testing.assert_frame_equal(
            first.loc[first["date"] <= cutoff, columns].reset_index(drop=True),
            second.loc[second["date"] <= cutoff, columns].reset_index(drop=True),
        )

    def test_industry_features_use_same_date_members(self) -> None:
        priced = build_price_features(panel())
        result = attach_industry_features(priced)
        day = result[result["date"] == result["date"].max()]
        software = day[day["industry"] == "软件"]

        self.assertEqual(1, software["industry_median_mom20"].nunique())
        self.assertAlmostEqual(
            software["mom_20"].median(),
            software["industry_median_mom20"].iloc[0],
        )
        self.assertTrue(
            np.allclose(
                software["stock_minus_industry_mom20"],
                software["mom_20"] - software["industry_median_mom20"],
            )
        )

    def test_benchmark_excess_momentum_is_stock_minus_index(self) -> None:
        priced = build_price_features(panel())
        result = attach_benchmark_features(priced, benchmark())
        row = result.dropna(subset=["excess_mom20"]).iloc[-1]

        self.assertAlmostEqual(
            row["mom_20"] - row["benchmark_mom20"],
            row["excess_mom20"],
        )

    def test_feature_groups_are_disjoint_and_non_empty(self) -> None:
        self.assertTrue(TREND_FEATURES)
        self.assertTrue(REVERSAL_FEATURES)
        self.assertTrue(QUALITY_FEATURES)
        self.assertFalse(set(TREND_FEATURES) & set(REVERSAL_FEATURES))
        self.assertFalse(set(TREND_FEATURES) & set(QUALITY_FEATURES))
        self.assertFalse(set(REVERSAL_FEATURES) & set(QUALITY_FEATURES))

    def test_raw_daily_panel_builds_market_context_for_live_runtime(self) -> None:
        raw = panel().loc[
            :,
            [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "name",
                "industry",
                "concept",
            ],
        ]
        fundamentals = pd.DataFrame(
            columns=[
                "code",
                "report_date",
                "announcement_date",
            ]
        )

        result = build_v2_features(
            raw,
            benchmark(),
            fundamentals,
            pd.DatetimeIndex(sorted(raw["date"].unique())),
        )

        self.assertIn("breadth_above_ma20_raw", result)
        self.assertIn("market_median_drawdown60_raw", result)
        self.assertTrue(result["breadth_above_ma20_raw"].notna().any())


if __name__ == "__main__":
    unittest.main()
