from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

import daily_history as dh
from selection_refinement import operating_cash_flow_positive, ttm_roe_pct


def calendar(days: int = 80) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-03-02", periods=days)


def history(cal: pd.DatetimeIndex, close: list[float], changes: list[float] | None = None) -> pd.DataFrame:
    close_series = pd.Series(close, dtype=float)
    if changes is None:
        changes = close_series.diff().fillna(0.0).tolist()
    return pd.DataFrame(
        {
            "date": cal[: len(close)].strftime("%Y-%m-%d"),
            "open": close_series,
            "high": close_series,
            "low": close_series,
            "close": close_series,
            "volume": 10_000.0,
            "amount": close_series * 10_000.0 * 100.0,
            "ups_downs": changes,
        }
    )


class ExRightsAdjustmentTests(unittest.TestCase):
    def test_dividend_drop_is_adjusted_out_of_history(self) -> None:
        """Catches a raw ex-dividend drop being read as a 5% price loss."""
        cal = calendar(6)
        # Day 3 goes ex-dividend 0.5 on a 10.0 stock and then closes flat at 9.5.
        frame = dh._prepare(history(cal, [10.0, 10.0, 10.0, 9.5, 9.5, 9.5], [0, 0, 0, 0, 0, 0]))

        adjusted, events = dh.adjust_ex_rights(frame, cal)

        self.assertEqual(1, events)
        self.assertAlmostEqual(9.5, adjusted["close"].iloc[0])
        self.assertAlmostEqual(0.0, adjusted["close"].pct_change().abs().max())

    def test_missing_session_is_not_mistaken_for_ex_rights(self) -> None:
        """Catches hiding a real move that happened on a day absent from the cache."""
        cal = calendar(6)
        frame = history(cal, [10.0, 10.0, 10.0, 9.0, 9.0, 9.0], [0, 0, 0, 0, 0, 0]).drop(index=2)
        prepared = dh._prepare(frame)

        adjusted, events = dh.adjust_ex_rights(prepared, cal)

        self.assertEqual(0, events)
        self.assertAlmostEqual(10.0, adjusted["close"].iloc[0])

    def test_real_price_moves_are_left_alone(self) -> None:
        cal = calendar(4)
        frame = dh._prepare(history(cal, [10.0, 10.5, 9.8, 10.1]))

        adjusted, events = dh.adjust_ex_rights(frame, cal)

        self.assertEqual(0, events)
        self.assertEqual([10.0, 10.5, 9.8, 10.1], adjusted["close"].round(4).tolist())


class SessionCompletenessTests(unittest.TestCase):
    def test_gap_in_recent_sessions_invalidates_history(self) -> None:
        cal = calendar(80)
        bars = dh._prepare(history(cal, [10.0] * 80)).drop(index=70)

        self.assertEqual(1, dh.missing_sessions(bars["date"], cal, cal[-1], 61))
        self.assertFalse(dh.history_window_is_complete(bars, cal, cal[-1]))

    def test_few_older_gaps_are_tolerated_but_many_are_not(self) -> None:
        cal = calendar(80)
        few = dh._prepare(history(cal, [10.0] * 80)).drop(index=[30, 40])
        many = dh._prepare(history(cal, [10.0] * 80)).drop(index=list(range(25, 32)))

        self.assertTrue(dh.history_window_is_complete(few, cal, cal[-1]))
        self.assertFalse(dh.history_window_is_complete(many, cal, cal[-1]))

    def test_alignment_keeps_sixty_session_lookback_exact(self) -> None:
        """Catches iloc-based windows reaching past 60 sessions when days are missing."""
        cal = calendar(80)
        close = [10.0 + i for i in range(80)]
        bars = dh._prepare(history(cal, close)).drop(index=[30, 40])

        aligned = dh.align_to_calendar(bars, cal)

        self.assertEqual(80, len(aligned))
        self.assertEqual(close[19], aligned["close"].iloc[-61])
        self.assertEqual(close[29], aligned.loc[aligned["date"] == cal[30], "close"].item())
        self.assertTrue(pd.isna(aligned.loc[aligned["date"] == cal[30], "volume"].item()))

    def test_complete_window_is_accepted(self) -> None:
        cal = calendar(80)
        bars = dh._prepare(history(cal, [10.0] * 80))

        self.assertTrue(dh.history_window_is_complete(bars, cal, cal[-1]))

    def test_spot_bar_adjusts_history_for_signal_day_ex_rights(self) -> None:
        cal = calendar(70)
        bars = dh._prepare(history(cal[:69], [10.0] * 69))
        row = {"new_price": 9.6, "pre_close_price": 9.5, "volume": 1e6, "deal_amount": 9.6e6}

        result = dh.append_spot_bar(bars, row, cal[69], cal)

        self.assertEqual(70, len(result))
        self.assertAlmostEqual(9.5, result["close"].iloc[-2])
        self.assertTrue(dh.history_window_is_complete(result, cal, cal[69]))

    def test_explicit_cache_directory_is_used_without_instock_history(self) -> None:
        cal = calendar(80)
        with tempfile.TemporaryDirectory() as temporary:
            history(cal, [10.0 + i * 0.01 for i in range(80)]).drop(columns="ups_downs").to_csv(
                Path(temporary) / "000001.csv", index=False
            )
            bars, meta = dh.load_code_history(
                "000001", cal[-1], calendar=cal, hist_dir=None, fallback_dir=Path(temporary)
            )

        self.assertEqual("backtest_cache", meta["source"])
        self.assertEqual(80, len(bars))


class VolumeUnitTests(unittest.TestCase):
    def test_board_lot_rows_are_converted_to_shares(self) -> None:
        """Catches a 100x volume jump when sources mix lots and shares."""
        frame = pd.DataFrame(
            {
                "close": [10.0, 10.0],
                "volume": [1_000_000.0, 10_000.0],
                "amount": [10_000_000.0, 10_000_000.0],
            }
        )

        result = dh.normalize_volume_units(frame)

        self.assertEqual([1_000_000.0, 1_000_000.0], result["volume"].tolist())


class TrailingFundamentalsTests(unittest.TestCase):
    def test_ttm_roe_comes_from_pb_over_pe(self) -> None:
        self.assertAlmostEqual(12.5, ttm_roe_pct({"pe9": 16, "pbnewmrq": 2}))
        self.assertLess(ttm_roe_pct({"pe9": -20, "pbnewmrq": 2}), 0)
        self.assertIsNone(ttm_roe_pct({"pe9": None, "pbnewmrq": 2}))

    def test_cash_flow_sign_prefers_trailing_ratio(self) -> None:
        # Year-to-date cash flow is negative but the trailing ratio is positive.
        self.assertTrue(operating_cash_flow_positive({"pcfjyxjl9": 15, "per_netcash_operate": -0.2}))
        self.assertFalse(operating_cash_flow_positive({"pcfjyxjl9": -8}))
        self.assertIsNone(operating_cash_flow_positive({}))


if __name__ == "__main__":
    unittest.main()
