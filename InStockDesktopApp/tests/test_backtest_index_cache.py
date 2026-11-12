from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKERS = ROOT / "workers"
SRC = ROOT / "src"
if str(WORKERS) not in sys.path:
    sys.path.insert(0, str(WORKERS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import backtest_worker


def _bars(end: str, periods: int) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000.0,
            "amount": 100_000.0,
        }
    )


class IndexCacheRefreshTests(unittest.TestCase):
    def test_midweek_cache_behind_requested_end_attempts_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            cached = _bars("2026-07-13", 130)
            cached.assign(date=cached["date"].dt.date.astype(str)).to_csv(
                cache_dir / "000300.csv", index=False
            )
            fresh = _bars("2026-07-14", 131)

            with (
                patch.object(
                    backtest_worker, "_fetch_index_eastmoney", return_value=fresh
                ) as fetch,
                patch.object(
                    backtest_worker,
                    "_fetch_index_tencent",
                    side_effect=ConnectionError("secondary unavailable"),
                ),
                patch.object(
                    backtest_worker,
                    "_fetch_index_fallback",
                    side_effect=ConnectionError("fallback unavailable"),
                ),
            ):
                result = backtest_worker.fetch_index_bars(
                    "000300", "2026-01-01", "2026-07-15", cache_dir
                )

            fetch.assert_called_once()
            self.assertEqual(pd.Timestamp("2026-07-14"), result["date"].max())

    def test_selects_freshest_merged_source_instead_of_first_long_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            cached = _bars("2026-07-31", 130)
            cached.assign(date=cached["date"].dt.date.astype(str)).to_csv(
                cache_dir / "000300.csv", index=False
            )
            current = _bars("2026-08-03", 131)
            stale = _bars("2026-07-31", 131)

            with (
                patch.object(
                    backtest_worker,
                    "_fetch_index_eastmoney",
                    side_effect=ConnectionError("primary unavailable"),
                ) as eastmoney,
                patch.object(
                    backtest_worker,
                    "_fetch_index_tencent",
                    return_value=current,
                    create=True,
                ) as tencent,
                patch.object(
                    backtest_worker, "_fetch_index_fallback", return_value=stale
                ),
            ):
                eastmoney.__name__ = "_fetch_index_eastmoney"
                result = backtest_worker.fetch_index_bars(
                    "000300", "2026-01-01", "2026-08-03", cache_dir
                )

            tencent.assert_called_once()
            self.assertEqual(pd.Timestamp("2026-08-03"), result["date"].max())
            persisted = pd.read_csv(cache_dir / "000300.csv")
            self.assertEqual("2026-08-03", persisted["date"].max())

    def test_tencent_adapter_maps_daily_bar(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "code": 0,
            "msg": "",
            "data": {
                "sh000300": {
                    "day": [
                        [
                            "2026-08-03",
                            "4561.82",
                            "4543.18",
                            "4572.59",
                            "4529.19",
                            "232042584",
                        ]
                    ]
                }
            },
        }
        session = MagicMock()
        session.get.return_value = response

        with patch.object(backtest_worker.requests, "Session", return_value=session):
            result = backtest_worker._fetch_index_tencent(
                "000300", "2026-08-03", "2026-08-03"
            )

        response.raise_for_status.assert_called_once()
        self.assertEqual(pd.Timestamp("2026-08-03"), result.iloc[0]["date"])
        self.assertEqual(4561.82, result.iloc[0]["open"])
        self.assertEqual(4543.18, result.iloc[0]["close"])
        self.assertEqual(4572.59, result.iloc[0]["high"])
        self.assertEqual(4529.19, result.iloc[0]["low"])
        self.assertEqual(232042584.0, result.iloc[0]["volume"])
        self.assertEqual(0.0, result.iloc[0]["amount"])

    def test_tencent_adapter_rejects_missing_daily_rows(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "code": 0,
            "msg": "",
            "data": {"sh000300": {}},
        }
        session = MagicMock()
        session.get.return_value = response

        with (
            patch.object(backtest_worker.requests, "Session", return_value=session),
            self.assertRaisesRegex(RuntimeError, "Tencent index response has no daily bars"),
        ):
            backtest_worker._fetch_index_tencent(
                "000300", "2026-08-03", "2026-08-03"
            )

    def test_fresh_source_replaces_overlapping_cached_daily_bar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            cached = _bars("2026-07-31", 130)
            cached.loc[cached["date"] == pd.Timestamp("2026-07-31"), "close"] = 90.0
            cached.assign(date=cached["date"].dt.date.astype(str)).to_csv(
                cache_dir / "000300.csv", index=False
            )
            fresh = _bars("2026-08-03", 131)
            fresh.loc[fresh["date"] == pd.Timestamp("2026-07-31"), "close"] = 123.0

            with (
                patch.object(
                    backtest_worker, "_fetch_index_eastmoney", return_value=fresh
                ),
                patch.object(
                    backtest_worker,
                    "_fetch_index_tencent",
                    side_effect=ConnectionError("secondary unavailable"),
                ),
                patch.object(
                    backtest_worker,
                    "_fetch_index_fallback",
                    side_effect=ConnectionError("fallback unavailable"),
                ),
            ):
                backtest_worker.fetch_index_bars(
                    "000300", "2026-01-01", "2026-08-03", cache_dir
                )

            persisted = pd.read_csv(cache_dir / "000300.csv")
            overlapping = persisted.loc[persisted["date"] == "2026-07-31"]
            self.assertEqual(123.0, overlapping.iloc[0]["close"])


if __name__ == "__main__":
    unittest.main()
