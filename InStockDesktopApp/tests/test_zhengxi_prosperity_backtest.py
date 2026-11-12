from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))


class ZhengxiBacktestTests(unittest.TestCase):
    def test_forward_return_uses_signal_close_and_nth_future_session(self) -> None:
        """Catches using pre-signal or same-day movement as forward performance."""
        from zhengxi_prosperity_backtest import forward_returns

        dates = pd.bdate_range("2026-01-02", periods=6)
        bars = pd.DataFrame(
            {"date": dates, "close": [10.0, 10.5, 11.0, 12.0, 12.5, 13.0]}
        )
        selected = pd.DataFrame(
            [
                {
                    "code": "000001",
                    "signal_date": dates[1],
                    "selection_tier": "严格入选",
                    "industry": "测试行业",
                }
            ]
        )

        result = forward_returns(selected, {"000001": bars}, horizons=(2,))

        self.assertEqual(dates[1], result.loc[0, "entry_date"])
        self.assertAlmostEqual(12.0 / 10.5 - 1.0, result.loc[0, "2d_return"])

    def test_summary_keeps_strict_and_fallback_tiers_separate(self) -> None:
        """Catches fallback performance being presented as strict-signal performance."""
        from zhengxi_prosperity_backtest import summarize_backtest

        observations = pd.DataFrame(
            [
                {"code": "000001", "signal_date": "2026-01-05", "selection_tier": "严格入选", "industry": "A", "5d_return": 0.10},
                {"code": "000002", "signal_date": "2026-01-05", "selection_tier": "一级候补", "industry": "B", "5d_return": -0.04},
                {"code": "000003", "signal_date": "2026-01-05", "selection_tier": "二级候补", "industry": "B", "5d_return": 0.01},
            ]
        )
        benchmark = pd.DataFrame(
            [{"signal_date": "2026-01-05", "5d_return": 0.02}]
        )

        report = summarize_backtest(observations, benchmark)

        self.assertEqual(
            {"严格入选", "一级候补", "二级候补"},
            set(report["tier_performance"]),
        )
        self.assertAlmostEqual(0.0033333333, report["horizons"]["5d"]["excess_mean"])
        self.assertEqual(3, report["horizons"]["5d"]["count"])


if __name__ == "__main__":
    unittest.main()
