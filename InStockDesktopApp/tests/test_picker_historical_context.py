from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from instock_tech_picker_data import load_historical_context


class PickerHistoricalContextTests(unittest.TestCase):
    def test_empty_factor_weights_use_current_true_range_when_cache_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                context = load_historical_context(
                    [
                        {
                            "code": "000002",
                            "date": "2026-07-14",
                            "new_price": 10.0,
                            "high_price": 10.4,
                            "low_price": 9.8,
                            "pre_close_price": 10.1,
                        }
                    ],
                    {},
                    Path(temporary),
                )

        self.assertIn("000002", context)
        self.assertAlmostEqual(context["000002"]["atr_14"], 0.6)
        self.assertIsNone(context["000002"]["historical_factor_score"])

    def test_empty_factor_weights_still_load_atr_without_creating_a_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            dates = pd.date_range("2026-04-01", periods=80, freq="D")
            close = pd.Series([10.0 + index * 0.05 for index in range(len(dates))])
            bars = pd.DataFrame(
                {
                    "date": dates,
                    "open": close - 0.05,
                    "high": close + 0.20,
                    "low": close - 0.20,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": close * 1_000_000,
                }
            )
            bars.to_csv(cache_dir / "000001.csv", index=False)
            rows = [
                {
                    "code": "000001",
                    "date": dates[-1].strftime("%Y-%m-%d"),
                    "new_price": float(close.iloc[-1]),
                    "volume": 1_000_000,
                }
            ]

            context = load_historical_context(rows, {}, cache_dir)

        self.assertIn("000001", context)
        self.assertGreater(context["000001"]["atr_14"], 0)
        self.assertIsNone(context["000001"]["historical_factor_score"])


if __name__ == "__main__":
    unittest.main()
