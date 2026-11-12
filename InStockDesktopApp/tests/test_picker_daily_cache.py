from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from picker_backtest import fetch_daily_bars


def bars(end: str, periods: int) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000.0,
            "amount": 10000.0,
        }
    )


class DailyBarCacheRefreshTests(unittest.TestCase):
    def test_local_snapshot_completes_network_history_that_stops_one_day_early(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            network = bars("2026-08-03", 130)
            local = pd.DataFrame(
                [{
                    "date": "2026-08-04",
                    "open": 10.5,
                    "high": 11.1,
                    "low": 10.4,
                    "close": 10.9,
                    "volume": 2000.0,
                    "amount": 21800.0,
                }]
            )

            result = fetch_daily_bars(
                "000001",
                "2026-01-01",
                "2026-08-04",
                cache,
                fetcher=Mock(return_value=network),
                snapshot_loader=Mock(return_value=local),
            )

            self.assertEqual(result["date"].max(), pd.Timestamp("2026-08-04"))
            latest = result.iloc[-1]
            self.assertEqual(latest["open"], 10.5)
            self.assertEqual(latest["close"], 10.9)
            persisted = pd.read_csv(cache / "000001.csv")
            self.assertEqual(persisted["date"].max(), "2026-08-04")

    def test_cache_touched_today_but_behind_requested_end_is_refetched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            requested_end = pd.Timestamp("2026-08-20")
            cached = bars(
                (requested_end - pd.Timedelta(days=1)).date().isoformat(), 130
            )
            cached.to_csv(cache / "000001.csv", index=False)
            refreshed = bars(requested_end.date().isoformat(), 131)
            fetcher = Mock(return_value=refreshed)
            start = (requested_end - pd.Timedelta(days=250)).date().isoformat()
            end = requested_end.date().isoformat()

            result = fetch_daily_bars(
                "000001", start, end, cache, fetcher=fetcher
            )

            self.assertEqual(requested_end, result["date"].max())
            persisted = pd.read_csv(cache / "000001.csv")
            self.assertEqual(end, persisted["date"].max())

    def test_cache_covering_requested_end_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            requested_end = pd.Timestamp("2026-08-20")
            complete = bars(requested_end.date().isoformat(), 130)
            complete.to_csv(cache / "000001.csv", index=False)
            fetcher = Mock(return_value=bars(requested_end.date().isoformat(), 131))
            start = (requested_end - pd.Timedelta(days=250)).date().isoformat()
            end = requested_end.date().isoformat()

            result = fetch_daily_bars(
                "000001", start, end, cache, fetcher=fetcher
            )

            fetcher.assert_not_called()
            self.assertEqual(requested_end, result["date"].max())


if __name__ == "__main__":
    unittest.main()
