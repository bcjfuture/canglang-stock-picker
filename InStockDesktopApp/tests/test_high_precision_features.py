from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_picker import (
    build_high_precision_features,
    calculate_high_precision_forward_5,
    make_training_rows,
    split_fit_and_calibration,
)
from picker_backtest import build_symbol_panel


def bars() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=90)
    close = np.linspace(10.0, 19.0, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, len(dates)),
            "amount": close * np.linspace(1_000_000, 2_000_000, len(dates)),
        }
    )


class HighPrecisionFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "engine" / "high_precision_config.json").read_text(
                encoding="utf-8"
            )
        )
        cls.profiles = json.loads(
            (ROOT / "engine" / "picker_factor_profiles.json").read_text(
                encoding="utf-8"
            )
        )

    def test_future_price_mutation_does_not_change_earlier_features(self) -> None:
        first_bars = bars()
        second_bars = copy.deepcopy(first_bars)
        second_bars.loc[
            second_bars.index[-5:], ["open", "high", "low", "close"]
        ] *= 5.0
        first = build_symbol_panel(
            "000001", first_bars, {"industry": "软件", "concept": "AI"}
        )
        second = build_symbol_panel(
            "000001", second_bars, {"industry": "软件", "concept": "AI"}
        )
        cutoff = pd.Timestamp(first_bars.iloc[-7]["date"])
        _, fixed_statistics = build_high_precision_features(
            first[first["date"] <= cutoff], self.profiles, self.config
        )
        first_features, _ = build_high_precision_features(
            first,
            self.profiles,
            self.config,
            train_statistics=fixed_statistics,
        )
        second_features, _ = build_high_precision_features(
            second,
            self.profiles,
            self.config,
            train_statistics=fixed_statistics,
        )
        columns = self.config["features"]
        pd.testing.assert_frame_equal(
            first_features[first_features["date"] <= cutoff][columns],
            second_features[second_features["date"] <= cutoff][columns],
            check_dtype=False,
        )

    def test_label_is_positive_only_when_forward_5_net_return_is_positive(self) -> None:
        panel = build_symbol_panel(
            "000001", bars(), {"industry": "银行", "concept": ""}
        )
        features, _ = build_high_precision_features(
            panel, self.profiles, self.config
        )
        training = make_training_rows(features)
        self.assertTrue(
            (
                training["label_5"]
                == (pd.to_numeric(training["forward_5"]) > 0).astype(int)
            ).all()
        )

    def test_five_day_return_uses_next_open_and_signal_plus_five_close(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-05", periods=7),
                "code": ["000001"] * 7,
                "open": [10.0, 10.0, 10.2, 10.3, 10.4, 10.5, 10.6],
                "close": [10.0, 10.1, 10.2, 10.3, 10.4, 11.0, 12.0],
            }
        )
        result = calculate_high_precision_forward_5(frame, cost=0.003)
        self.assertAlmostEqual(11.0 / 10.0 - 1.0 - 0.003, result.iloc[0])

    def test_stock_features_are_ranked_within_each_date(self) -> None:
        first = build_symbol_panel(
            "000001", bars(), {"industry": "软件", "concept": "AI"}
        )
        second_bars = bars()
        multiplier = np.linspace(1.0, 0.8, len(second_bars))
        for column in ("open", "high", "low", "close"):
            second_bars[column] *= multiplier
        second = build_symbol_panel(
            "000002", second_bars, {"industry": "银行", "concept": ""}
        )
        features, _ = build_high_precision_features(
            pd.concat([first, second], ignore_index=True),
            self.profiles,
            self.config,
        )
        day = features.dropna(subset=["forward_5"]).groupby("date").filter(
            lambda group: len(group) == 2
        )
        ranks = day.groupby("date")["mom_20"].apply(list).iloc[20:]
        self.assertTrue(all(sorted(values) == [0.5, 1.0] for values in ranks))

    def test_market_standardization_statistics_come_from_training_rows(self) -> None:
        panel = build_symbol_panel(
            "000001", bars(), {"industry": "银行", "concept": ""}
        )
        train = panel[panel["date"] <= panel["date"].iloc[75]]
        validation = panel[panel["date"] > panel["date"].iloc[75]]
        _, statistics = build_high_precision_features(
            train, self.profiles, self.config
        )
        validation_features, reused = build_high_precision_features(
            validation,
            self.profiles,
            self.config,
            train_statistics=statistics,
        )
        self.assertEqual(statistics, reused)
        self.assertFalse(validation_features.empty)

    def test_calibration_tail_is_chronologically_after_fit_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=150),
                "label_5": [0, 1] * 75,
            }
        )
        fit, calibration = split_fit_and_calibration(frame, 30)
        self.assertLess(fit["date"].max(), calibration["date"].min())
        self.assertEqual(30, calibration["date"].nunique())


if __name__ == "__main__":
    unittest.main()
