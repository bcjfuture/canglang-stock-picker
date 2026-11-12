from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from workers import sector_data
from workers.sector_data import (
    AkshareSectorAPI,
    SectorDataUnavailable,
    build_breadth_signals,
    load_sector_bundle,
    normalize_sector_history,
)


class FakeAPI:
    def __init__(
        self,
        days: int = 300,
        fail: bool = False,
        *,
        board_count: int = 1,
        history_fail: bool = False,
    ) -> None:
        self.days = days
        self.fail = fail
        self.board_count = board_count
        self.history_fail = history_fail
        self.fetch_calls = 0

    def list_boards(self, sector_type: str) -> pd.DataFrame:
        if self.fail:
            raise RuntimeError("offline")
        return pd.DataFrame(
            {
                "sector_id": [f"BK{index:04d}" for index in range(1, self.board_count + 1)],
                "sector_name": ["测试行业"] * self.board_count,
            }
        )

    def fetch_history(self, sector_type: str, name: str, start: str, end: str) -> pd.DataFrame:
        self.fetch_calls += 1
        if self.fail or self.history_fail:
            raise RuntimeError("offline")
        dates = pd.bdate_range("2025-01-01", periods=self.days)
        close = pd.Series(range(self.days), dtype=float) * 0.1 + 100
        return pd.DataFrame(
            {
                "日期": dates,
                "开盘": close - 0.2,
                "收盘": close,
                "最高": close + 0.5,
                "最低": close - 0.5,
                "成交量": 1000,
                "成交额": 100_000,
                "换手率": 2.0,
            }
        )


def benchmark_loader(start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=400)
    return pd.DataFrame({"date": dates, "close": 3000 + pd.Series(range(len(dates)))})


def signal_loader(_sector_type: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    return (
        pd.DataFrame(
            {
                "sector_name": ["测试行业"],
                "fund_score": [0.8],
                "breadth_score": [0.6],
                "heat_score": [0.5],
                "up_ratio": [0.7],
                "current_data_completeness": [1.0],
            }
        ),
        pd.Timestamp("2026-02-24"),
    )


def next_day_signal_loader(
    _sector_type: str, *, as_of: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.Timestamp]:
    signals, _ = signal_loader(_sector_type)
    return signals, pd.Timestamp("2026-02-25")


class SectorDataTests(unittest.TestCase):
    def test_daily_change_prefers_fund_flow_and_weights_constituents(self) -> None:
        selection = pd.DataFrame(
            {
                "industry": ["行业A", "行业A", "行业B", "行业B"],
                "concept": ["", "", "", ""],
                "change_rate": [1.0, 3.0, 4.0, -2.0],
                "free_cap": [100.0, 300.0, 100.0, 100.0],
            }
        )
        fund = pd.DataFrame(
            {"name": ["行业A"], "change_rate": [2.5]}
        )
        builder = getattr(sector_data, "build_daily_change_signals", None)
        self.assertIsNotNone(builder)
        if builder is None:
            return

        result = builder(selection, fund, "industry").set_index("sector_name")

        self.assertAlmostEqual(2.5, float(result.loc["行业A", "daily_change_rate"]))
        self.assertEqual("fund_flow", result.loc["行业A", "daily_change_source"])
        self.assertAlmostEqual(1.0, float(result.loc["行业B", "daily_change_rate"]))
        self.assertEqual("constituents", result.loc["行业B", "daily_change_source"])

    def test_daily_change_expands_concepts(self) -> None:
        selection = pd.DataFrame(
            {
                "industry": ["", ""],
                "concept": ["芯片,AI", "AI"],
                "change_rate": [1.0, 3.0],
                "free_cap": [100.0, 300.0],
            }
        )
        builder = getattr(sector_data, "build_daily_change_signals", None)
        self.assertIsNotNone(builder)
        if builder is None:
            return

        result = builder(selection, pd.DataFrame(), "concept").set_index("sector_name")

        self.assertAlmostEqual(1.0, float(result.loc["芯片", "daily_change_rate"]))
        self.assertAlmostEqual(2.5, float(result.loc["AI", "daily_change_rate"]))

    def test_append_derived_latest_row_uses_daily_change(self) -> None:
        history = normalize_sector_history(
            pd.DataFrame(
                {
                    "日期": ["2026-07-14"],
                    "开盘": [99.0],
                    "收盘": [100.0],
                    "最高": [101.0],
                    "最低": [98.0],
                    "成交量": [1000.0],
                    "成交额": [100000.0],
                    "换手率": [2.0],
                }
            ),
            "industry",
            "BK0001",
            "测试行业",
        )
        signals = pd.DataFrame(
            {
                "sector_name": ["测试行业"],
                "daily_change_rate": [2.0],
                "daily_change_source": ["fund_flow"],
            }
        )
        appender = getattr(sector_data, "append_derived_latest_rows", None)
        self.assertIsNotNone(appender)
        if appender is None:
            return

        result, stats = appender(history, signals, pd.Timestamp("2026-07-15"))
        latest = result.sort_values("date").iloc[-1]

        self.assertEqual(pd.Timestamp("2026-07-15"), latest["date"])
        self.assertAlmostEqual(102.0, float(latest["close"]))
        self.assertEqual(1, stats["derived_latest_count"])
        self.assertEqual("fund_flow", latest["derived_source"])
        self.assertEqual(1, len(history))

    def test_append_derived_latest_row_keeps_real_same_day_row(self) -> None:
        history = pd.DataFrame(
            {
                "sector_type": ["industry"],
                "sector_id": ["BK0001"],
                "sector_name": ["测试行业"],
                "date": [pd.Timestamp("2026-07-15")],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [104.0],
                "volume": [1000.0],
                "amount": [100000.0],
                "turnover": [2.0],
            }
        )
        signals = pd.DataFrame(
            {
                "sector_name": ["测试行业"],
                "daily_change_rate": [20.0],
                "daily_change_source": ["fund_flow"],
            }
        )
        appender = getattr(sector_data, "append_derived_latest_rows", None)
        self.assertIsNotNone(appender)
        if appender is None:
            return

        result, stats = appender(history, signals, pd.Timestamp("2026-07-15"))

        self.assertEqual(1, len(result))
        self.assertAlmostEqual(104.0, float(result.iloc[0]["close"]))
        self.assertEqual(0, stats["derived_latest_count"])

    def test_history_host_failover_is_bounded(self) -> None:
        class FailingSession:
            trust_env = True

            def __init__(self) -> None:
                self.timeouts: list[int] = []

            def get(self, *_args, **kwargs):
                self.timeouts.append(kwargs["timeout"])
                raise RuntimeError("offline")

        session = FailingSession()
        with patch("requests.Session", return_value=session):
            with self.assertRaises(SectorDataUnavailable):
                AkshareSectorAPI().fetch_history(
                    "industry", "BK0001", "20260715", "20260715"
                )

        self.assertEqual([5, 5, 5], session.timeouts)

    def test_history_outage_uses_one_probe_instead_of_fanning_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = SimpleNamespace(data_dir=Path(temporary))
            load_sector_bundle(
                paths,
                "industry",
                fetch_api=FakeAPI(days=300, board_count=3),
                benchmark_loader=benchmark_loader,
                signal_loader=signal_loader,
            )
            offline_api = FakeAPI(
                days=301, board_count=3, history_fail=True
            )

            result = load_sector_bundle(
                paths,
                "industry",
                fetch_api=offline_api,
                benchmark_loader=benchmark_loader,
                signal_loader=next_day_signal_loader,
            )

            self.assertEqual(1, offline_api.fetch_calls)
            self.assertTrue(result.quality["using_stale_cache"])

    def test_one_day_old_midweek_cache_attempts_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = SimpleNamespace(data_dir=Path(temporary))
            load_sector_bundle(
                paths,
                "industry",
                fetch_api=FakeAPI(days=300),
                benchmark_loader=benchmark_loader,
                signal_loader=signal_loader,
            )
            next_day_api = FakeAPI(days=301)

            load_sector_bundle(
                paths,
                "industry",
                fetch_api=next_day_api,
                benchmark_loader=benchmark_loader,
                signal_loader=next_day_signal_loader,
            )

            self.assertEqual(1, next_day_api.fetch_calls)

    def test_normalize_akshare_history_uses_expected_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "日期": ["2026-07-14"],
                "开盘": [100],
                "收盘": [102],
                "最高": [103],
                "最低": [99],
                "成交量": [10],
                "成交额": [1000],
                "换手率": [2.5],
            }
        )
        result = normalize_sector_history(raw, "industry", "BK0001", "测试行业")
        self.assertEqual(
            result.columns.tolist(),
            [
                "sector_type",
                "sector_id",
                "sector_name",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "turnover",
            ],
        )

    def test_industry_breadth_groups_latest_selection(self) -> None:
        selection = pd.DataFrame(
            {
                "industry": ["半导体", "半导体", "银行"],
                "concept": ["芯片,AI", "芯片", "高股息"],
                "change_rate": [2.0, -1.0, 0.5],
                "now_newhigh": [1, 0, 0],
                "long_avg_array": [1, 1, 0],
                "popularity_rank": [10, 30, 80],
            }
        )
        result = build_breadth_signals(selection, "industry").set_index("sector_name")
        self.assertAlmostEqual(float(result.loc["半导体", "up_ratio"]), 0.5)
        self.assertAlmostEqual(float(result.loc["半导体", "long_ma_ratio"]), 1.0)

    def test_cached_history_is_used_when_remote_refresh_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = SimpleNamespace(data_dir=Path(temporary))
            first = load_sector_bundle(
                paths,
                "industry",
                fetch_api=FakeAPI(days=300),
                benchmark_loader=benchmark_loader,
                signal_loader=signal_loader,
            )
            self.assertGreaterEqual(first.quality["valid_days"], 250)
            second = load_sector_bundle(
                paths,
                "industry",
                fetch_api=FakeAPI(fail=True),
                benchmark_loader=benchmark_loader,
                signal_loader=signal_loader,
            )
            self.assertTrue(second.quality["using_stale_cache"])
            self.assertGreaterEqual(second.quality["valid_days"], 250)

    def test_insufficient_cache_is_not_promoted_after_fetch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = SimpleNamespace(data_dir=Path(temporary))
            with self.assertRaises(SectorDataUnavailable):
                load_sector_bundle(
                    paths,
                    "concept",
                    fetch_api=FakeAPI(fail=True),
                    benchmark_loader=benchmark_loader,
                    signal_loader=signal_loader,
                )


if __name__ == "__main__":
    unittest.main()
