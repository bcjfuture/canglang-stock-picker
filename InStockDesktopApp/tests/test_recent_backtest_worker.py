from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from workers.recent_backtest_worker import (
    _load_bars_for_codes,
    evaluate_profiles,
    evaluate_snapshot,
    find_stored_snapshot,
    run_recent_backtest,
    select_signal_dates,
)
import workers.recent_backtest_worker as recent_worker


def bars(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "open", "close"])


class RecentBacktestWorkerTests(unittest.TestCase):
    def test_latest_trading_date_falls_back_before_requested_database_date(self) -> None:
        self.assertTrue(
            hasattr(recent_worker, "latest_trading_date_on_or_before"),
            "latest trading-date fallback helper is missing",
        )
        trading_dates = pd.to_datetime(["2026-07-30", "2026-07-31"])

        self.assertEqual(
            "2026-07-31",
            recent_worker.latest_trading_date_on_or_before(
                trading_dates, "2026-08-03"
            ),
        )

    def test_signal_dates_use_nearest_prior_snapshot_and_actual_holding_days(self) -> None:
        self.assertTrue(
            hasattr(recent_worker, "align_signal_dates"),
            "snapshot-date alignment helper is missing",
        )
        trading_dates = pd.bdate_range("2026-07-20", "2026-07-31")
        requested = {
            1: "2026-07-30",
            3: "2026-07-28",
            5: "2026-07-24",
        }

        aligned = recent_worker.align_signal_dates(
            trading_dates,
            "2026-07-31",
            requested,
            ["2026-07-20", "2026-07-27", "2026-07-30"],
        )

        self.assertEqual("2026-07-30", aligned[1]["signal_date"])
        self.assertEqual(1, aligned[1]["holding_trading_days"])
        self.assertFalse(aligned[1]["snapshot_fallback"])
        self.assertEqual("2026-07-27", aligned[3]["signal_date"])
        self.assertEqual(4, aligned[3]["holding_trading_days"])
        self.assertTrue(aligned[3]["snapshot_fallback"])
        self.assertEqual("2026-07-20", aligned[5]["signal_date"])
        self.assertEqual(9, aligned[5]["holding_trading_days"])
        self.assertTrue(aligned[5]["snapshot_fallback"])

    def test_signal_date_alignment_never_uses_future_snapshot(self) -> None:
        self.assertTrue(
            hasattr(recent_worker, "align_signal_dates"),
            "snapshot-date alignment helper is missing",
        )

        aligned = recent_worker.align_signal_dates(
            pd.bdate_range("2026-07-27", "2026-07-31"),
            "2026-07-31",
            {3: "2026-07-28"},
            ["2026-07-29", "2026-07-30"],
        )

        self.assertIsNone(aligned[3]["signal_date"])

    def test_run_recent_backtest_uses_common_index_date_and_exact_signal_dates(self) -> None:
        paths = SimpleNamespace(
            data_dir=Path("unused"),
            engine_dir=Path("unused"),
        )
        index = pd.DataFrame(
            {
                "date": pd.bdate_range("2026-07-20", "2026-07-31"),
                "open": [100.0] * 10,
                "close": [101.0] * 10,
            }
        )

        def snapshot(_paths, strategy, signal_date):
            return {
                "trade_date": signal_date,
                "rows": [
                    {"rank": 1, "code": "000001", "name": strategy}
                ],
                "_source": "test",
            }

        with patch(
            "workers.recent_backtest_worker.latest_complete_data_date",
            return_value="2026-08-03",
        ), patch(
            "workers.recent_backtest_worker.fetch_index_bars",
            return_value=index,
        ), patch(
            "workers.recent_backtest_worker._resolve_snapshot",
            side_effect=snapshot,
        ), patch(
            "workers.recent_backtest_worker._load_bars_for_codes",
            return_value=({"000001": index}, {}),
        ):
            try:
                result = run_recent_backtest(
                    paths,
                    lambda *_args: None,
                    selection_date_loader=lambda _date: [
                        "2026-07-20",
                        "2026-07-27",
                        "2026-07-30",
                    ],
                )
            except TypeError as exc:
                self.fail(f"recent backtest has no date-loader boundary: {exc}")

        self.assertEqual("2026-08-03", result["requested_evaluation_date"])
        self.assertEqual("2026-07-31", result["evaluation_date"])
        self.assertEqual(
            "2026-07-28",
            result["offsets"]["3"]["requested_signal_date"],
        )
        self.assertEqual(
            "2026-07-28", result["offsets"]["3"]["signal_date"]
        )
        self.assertEqual(
            3, result["offsets"]["3"]["holding_trading_days"]
        )
        self.assertFalse(result["offsets"]["3"]["snapshot_fallback"])
        self.assertTrue(
            any("2026-08-03" in item for item in result["warnings"])
        )

    def test_run_recent_backtest_keeps_requested_trading_date_when_selection_table_skips_it(self) -> None:
        paths = SimpleNamespace(
            data_dir=Path("unused"),
            engine_dir=Path("unused"),
        )
        index = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-08-03", "2026-08-04", "2026-08-05"]
                ),
                "open": [100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 103.0],
            }
        )
        resolved_dates: list[str] = []

        def snapshot(_paths, strategy, signal_date):
            resolved_dates.append(signal_date)
            return {
                "trade_date": signal_date,
                "rows": [
                    {"rank": 1, "code": "000001", "name": strategy}
                ],
                "_source": "test",
            }

        with patch(
            "workers.recent_backtest_worker.latest_complete_data_date",
            return_value="2026-08-05",
        ), patch(
            "workers.recent_backtest_worker.fetch_index_bars",
            return_value=index,
        ), patch(
            "workers.recent_backtest_worker._resolve_snapshot",
            side_effect=snapshot,
        ), patch(
            "workers.recent_backtest_worker._load_bars_for_codes",
            return_value=({"000001": index}, {}),
        ):
            result = run_recent_backtest(
                paths,
                lambda *_args: None,
                offsets=(1,),
                selection_date_loader=lambda _date: [
                    "2026-08-03",
                    "2026-08-05",
                ],
            )

        horizon = result["offsets"]["1"]
        self.assertEqual("2026-08-04", horizon["requested_signal_date"])
        self.assertEqual("2026-08-04", horizon["signal_date"])
        self.assertEqual(1, horizon["holding_trading_days"])
        self.assertFalse(horizon["snapshot_fallback"])
        self.assertTrue(resolved_dates)
        self.assertEqual({"2026-08-04"}, set(resolved_dates))

    def test_select_signal_dates_uses_trading_days(self) -> None:
        trading_dates = pd.to_datetime(
            [
                "2026-07-09",
                "2026-07-10",
                "2026-07-13",
                "2026-07-14",
                "2026-07-15",
                "2026-07-16",
            ]
        )

        result = select_signal_dates(trading_dates, "2026-07-16")

        self.assertEqual(
            {1: "2026-07-15", 3: "2026-07-13", 5: "2026-07-09"},
            result,
        )

    def test_evaluate_snapshot_uses_next_open_latest_close_and_cost(self) -> None:
        snapshot_rows = [
            {"rank": 1, "code": "000001", "name": "测试股"},
        ]
        bars_by_code = {
            "000001": bars(
                [
                    ("2026-07-13", 9.8, 9.9),
                    ("2026-07-14", 10.0, 10.4),
                    ("2026-07-15", 10.5, 10.8),
                    ("2026-07-16", 10.9, 11.0),
                ]
            )
        }

        result = evaluate_snapshot(
            snapshot_rows,
            bars_by_code,
            signal_date="2026-07-13",
            evaluation_date="2026-07-16",
            cost=0.003,
        )

        detail = result["details"][0]
        self.assertEqual("2026-07-14", detail["entry_date"])
        self.assertEqual("2026-07-16", detail["exit_date"])
        self.assertAlmostEqual(0.10, detail["gross_return"], places=12)
        self.assertAlmostEqual(0.097, detail["net_return"], places=12)
        self.assertAlmostEqual(0.097, result["summary"]["mean_net_return"], places=12)
        self.assertEqual(1, result["summary"]["evaluated_count"])

    def test_missing_entry_price_is_unavailable_not_zero_return(self) -> None:
        result = evaluate_snapshot(
            [{"rank": 1, "code": "000001", "name": "停牌股"}],
            {"000001": bars([("2026-07-13", 10.0, 10.0)])},
            signal_date="2026-07-13",
            evaluation_date="2026-07-16",
        )

        detail = result["details"][0]
        self.assertEqual("unavailable", detail["status"])
        self.assertIsNone(detail["net_return"])
        self.assertEqual(0, result["summary"]["evaluated_count"])
        self.assertEqual(1, result["summary"]["unavailable_count"])

    def test_evaluate_snapshot_reports_top5_tail15_and_all20_buckets(self) -> None:
        rows = [
            {
                "rank": rank,
                "original_rank": rank + 5 if rank <= 5 else rank - 5 if rank <= 10 else rank,
                "code": f"{rank:06d}",
                "name": f"股票{rank}",
            }
            for rank in range(1, 21)
        ]
        bars_by_code = {
            f"{rank:06d}": bars(
                [
                    ("2026-07-13", 10.0, 10.0),
                    ("2026-07-14", 10.0, 10.0),
                    ("2026-07-16", 10.0, 11.0 if rank <= 5 else 9.0),
                ]
            )
            for rank in range(1, 21)
        }
        result = evaluate_snapshot(
            rows,
            bars_by_code,
            signal_date="2026-07-13",
            evaluation_date="2026-07-16",
            cost=0.0,
        )
        buckets = result["summary"]["buckets"]
        self.assertEqual(1.0, buckets["top5"]["win_rate"])
        self.assertEqual(0.0, buckets["rank6_20"]["win_rate"])
        self.assertEqual(0.25, buckets["all20"]["win_rate"])
        self.assertEqual(0.0, buckets["baseline_top5"]["win_rate"])

    def test_external_core_slot_is_not_counted_in_original_top_five(self) -> None:
        rows = [
            {"rank": 1, "original_rank": 0, "code": "600360", "name": "核心席位"},
            *[
                {
                    "rank": rank + 1,
                    "original_rank": rank,
                    "code": f"{rank:06d}",
                    "name": f"原股票{rank}",
                }
                for rank in range(1, 6)
            ],
        ]
        bars_by_code = {
            row["code"]: bars(
                [
                    ("2026-07-13", 10.0, 10.0),
                    ("2026-07-14", 10.0, 10.0),
                    ("2026-07-16", 10.0, 11.0),
                ]
            )
            for row in rows
        }
        result = evaluate_snapshot(
            rows,
            bars_by_code,
            "2026-07-13",
            "2026-07-16",
            cost=0.0,
        )
        self.assertEqual(5, result["summary"]["buckets"]["baseline_top5"]["selected_count"])

    def test_find_stored_snapshot_uses_newest_exact_trade_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            strategy_dir = root / "all_market"
            strategy_dir.mkdir(parents=True)
            older = strategy_dir / "older.json"
            newer = strategy_dir / "newer.json"
            wrong = strategy_dir / "wrong.json"
            older.write_text(
                json.dumps(
                    {
                        "trade_date": "2026-07-15",
                        "generated_at": "2026-07-15 19:00:00",
                        "rows": [{"code": "000001"}],
                    }
                ),
                encoding="utf-8",
            )
            newer.write_text(
                json.dumps(
                    {
                        "trade_date": "2026-07-15",
                        "generated_at": "2026-07-15 21:00:00",
                        "rows": [{"code": "000002"}],
                    }
                ),
                encoding="utf-8",
            )
            wrong.write_text(
                json.dumps(
                    {
                        "trade_date": "2026-07-14",
                        "generated_at": "2026-07-16 22:00:00",
                        "rows": [{"code": "000003"}],
                    }
                ),
                encoding="utf-8",
            )

            result = find_stored_snapshot(root, "all_market", "2026-07-15")

            self.assertIsNotNone(result)
            self.assertEqual("000002", result["rows"][0]["code"])

    def test_find_stored_snapshot_rejects_stale_high_precision_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            strategy_dir = root / "high_precision"
            strategy_dir.mkdir(parents=True)
            (strategy_dir / "old.json").write_text(
                json.dumps(
                    {
                        "trade_date": "2026-07-15",
                        "generated_at": "2026-07-17T12:00:00",
                        "source_summary": {"run_id": "hpv4-old"},
                        "rows": [{"code": "000001"}],
                    }
                ),
                encoding="utf-8",
            )

            result = find_stored_snapshot(
                root,
                "high_precision",
                "2026-07-15",
                required_run_id="hpv5-current",
            )

            self.assertIsNone(result)

    def test_find_stored_snapshot_rejects_old_head_rerank_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            strategy_dir = root / "all_market"
            strategy_dir.mkdir(parents=True)
            (strategy_dir / "old.json").write_text(
                json.dumps(
                    {
                        "trade_date": "2026-07-15",
                        "source_summary": {"head_rerank_version": "old"},
                        "rows": [{"code": "000001"}],
                    }
                ),
                encoding="utf-8",
            )
            result = find_stored_snapshot(
                root,
                "all_market",
                "2026-07-15",
                required_head_rerank_version="top5-precision-sector-v1",
            )
            self.assertIsNone(result)

    def test_profile_failure_does_not_remove_other_profiles(self) -> None:
        snapshots = {
            "all_market": {
                "rows": [{"rank": 1, "code": "000001", "name": "正常股"}]
            },
            "tech_attack": {"rows": None},
        }
        bars_by_code = {
            "000001": bars(
                [
                    ("2026-07-14", 10.0, 10.2),
                    ("2026-07-16", 10.4, 10.5),
                ]
            )
        }

        result = evaluate_profiles(
            snapshots,
            bars_by_code,
            signal_date="2026-07-13",
            evaluation_date="2026-07-16",
            benchmark_return=0.01,
        )

        self.assertEqual("ok", result["all_market"]["status"])
        self.assertEqual("error", result["tech_attack"]["status"])

    def test_profile_gate_passes_only_when_reranked_top5_improves(self) -> None:
        snapshot_rows = [
            {
                "rank": rank,
                "original_rank": rank + 5 if rank <= 5 else rank - 5,
                "code": f"{rank:06d}",
                "name": f"股票{rank}",
            }
            for rank in range(1, 11)
        ]
        bars_by_code = {
            f"{rank:06d}": bars(
                [
                    ("2026-07-13", 10.0, 10.0),
                    ("2026-07-14", 10.0, 10.0),
                    ("2026-07-16", 10.0, 11.0 if rank <= 5 else 9.0),
                ]
            )
            for rank in range(1, 11)
        }
        result = evaluate_profiles(
            {"all_market": {"rows": snapshot_rows}},
            bars_by_code,
            signal_date="2026-07-13",
            evaluation_date="2026-07-16",
            benchmark_return=0.0,
            cost=0.0,
        )
        self.assertTrue(result["all_market"]["summary"]["head_rerank_gate_passed"])

    def test_market_bar_loader_is_serial_by_default(self) -> None:
        state = {"active": 0, "maximum": 0}
        lock = threading.Lock()

        def fake_fetch(code, _start, _end, _cache):
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.02)
            with lock:
                state["active"] -= 1
            return bars([("2026-07-16", 10.0, 10.0)])

        with patch(
            "workers.recent_backtest_worker.fetch_daily_bars",
            side_effect=fake_fetch,
        ):
            loaded, errors = _load_bars_for_codes(
                {"000001", "000002", "000003"}, "2026-07-16"
            )

        self.assertEqual({}, errors)
        self.assertEqual(3, len(loaded))
        self.assertEqual(1, state["maximum"])


if __name__ == "__main__":
    unittest.main()
