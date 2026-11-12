from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_snapshot(
    history_root: Path,
    strategy: str,
    filename: str,
    trade_date: str,
    generated_at: str,
    rows: list[tuple[str, str, int]],
) -> None:
    directory = history_root / strategy
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": strategy,
        "trade_date": trade_date,
        "generated_at": generated_at,
        "rows": [
            {"code": code, "name": name, "rank": rank}
            for code, name, rank in rows
        ],
    }
    (directory / filename).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


class ConsensusRankingTests(unittest.TestCase):
    def _builder(self):
        try:
            module = importlib.import_module("instock_app.consensus")
        except ModuleNotFoundError:
            self.fail("五日共识统计模块尚未实现")
        return module.build_five_day_consensus

    def test_uses_latest_snapshot_per_strategy_day_and_only_five_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history_root = Path(temporary) / "history"
            strategies = ("s1", "s2", "s3", "s4")

            write_snapshot(
                history_root,
                "s1",
                "20260828-old.json",
                "2026-08-28",
                "2026-08-28 09:00:00",
                [("000001", "甲公司", 1)],
            )
            write_snapshot(
                history_root,
                "s1",
                "20260828-new.json",
                "2026-08-28",
                "2026-08-28 10:00:00",
                [("000002", "乙公司", 1), ("000003", "丙公司", 2)],
            )
            write_snapshot(history_root, "s1", "27.json", "2026-08-27", "2026-08-27 10:00:00", [("000002", "乙公司", 3)])
            write_snapshot(history_root, "s1", "26.json", "2026-08-26", "2026-08-26 10:00:00", [("000001", "甲公司", 1)])
            write_snapshot(history_root, "s1", "25.json", "2026-08-25", "2026-08-25 10:00:00", [("000001", "甲公司", 2)])
            write_snapshot(history_root, "s1", "24.json", "2026-08-24", "2026-08-24 10:00:00", [("000003", "丙公司", 1)])
            write_snapshot(history_root, "s1", "23.json", "2026-08-23", "2026-08-23 10:00:00", [("000009", "过期公司", 1)])

            write_snapshot(history_root, "s2", "28.json", "2026-08-28", "2026-08-28 11:00:00", [("000002", "乙公司", 2)])
            write_snapshot(history_root, "s2", "27.json", "2026-08-27", "2026-08-27 11:00:00", [("000001", "甲公司", 1)])
            write_snapshot(history_root, "s2", "26.json", "2026-08-26", "2026-08-26 11:00:00", [("000002", "乙公司", 2)])
            write_snapshot(history_root, "s2", "25.json", "2026-08-25", "2026-08-25 11:00:00", [("000001", "甲公司", 1)])
            write_snapshot(history_root, "s2", "24.json", "2026-08-24", "2026-08-24 11:00:00", [("000003", "丙公司", 2)])
            write_snapshot(history_root, "s3", "28.json", "2026-08-28", "2026-08-28 12:00:00", [("000003", "丙公司", 1)])

            result = self._builder()(history_root, strategies)

            self.assertEqual(
                ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28"],
                result["trade_dates"],
            )
            self.assertEqual(5, result["covered_days"])
            self.assertEqual(11, result["available_strategy_days"])
            self.assertEqual(20, result["maximum_appearances"])
            self.assertEqual(["000001", "000002", "000003"], [row["code"] for row in result["rows"]])

            first, second, third = result["rows"]
            self.assertEqual(4, first["appearance_count"])
            self.assertEqual(3, first["day_count"])
            self.assertEqual(2, first["strategy_count"])
            self.assertEqual(1.25, first["average_rank"])
            self.assertEqual([0, 2, 1, 1, 0], first["daily_counts"])

            self.assertEqual([0, 0, 1, 1, 2], second["daily_counts"])
            self.assertEqual(2, third["day_count"])
            self.assertEqual(3, third["strategy_count"])
            self.assertNotIn("000009", [row["code"] for row in result["rows"]])

    def test_three_day_window_excludes_older_hits_and_caps_at_twelve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history_root = Path(temporary) / "history"
            strategies = ("s1", "s2", "s3", "s4")
            for strategy in strategies:
                write_snapshot(
                    history_root,
                    strategy,
                    f"{strategy}-28.json",
                    "2026-08-28",
                    "2026-08-28 20:00:00",
                    [("000001", "三日热门", 1)],
                )
                write_snapshot(
                    history_root,
                    strategy,
                    f"{strategy}-27.json",
                    "2026-08-27",
                    "2026-08-27 20:00:00",
                    [("000001", "三日热门", 2)],
                )
                write_snapshot(
                    history_root,
                    strategy,
                    f"{strategy}-26.json",
                    "2026-08-26",
                    "2026-08-26 20:00:00",
                    [("000001", "三日热门", 3)],
                )
                write_snapshot(
                    history_root,
                    strategy,
                    f"{strategy}-25.json",
                    "2026-08-25",
                    "2026-08-25 20:00:00",
                    [("000002", "窗口外股票", 1)],
                )

            module = importlib.import_module("instock_app.consensus")
            result = module.build_consensus(
                history_root,
                strategies,
                day_limit=3,
            )

            self.assertEqual(
                ["2026-08-26", "2026-08-27", "2026-08-28"],
                result["trade_dates"],
            )
            self.assertEqual(3, result["covered_days"])
            self.assertEqual(12, result["available_strategy_days"])
            self.assertEqual(12, result["maximum_appearances"])
            self.assertEqual(["000001"], [row["code"] for row in result["rows"]])
            self.assertEqual(12, result["rows"][0]["appearance_count"])
            self.assertEqual([4, 4, 4], result["rows"][0]["daily_counts"])

    def test_empty_or_malformed_history_returns_an_honest_empty_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history_root = Path(temporary) / "history"
            invalid_dir = history_root / "s1"
            invalid_dir.mkdir(parents=True)
            (invalid_dir / "broken.json").write_text("not-json", encoding="utf-8")
            (invalid_dir / "array.json").write_text("[]", encoding="utf-8")
            (invalid_dir / "null.json").write_text("null", encoding="utf-8")
            (invalid_dir / "scalar.json").write_text("42", encoding="utf-8")
            (invalid_dir / "binary.json").write_bytes(b"\xff\xfe\xfd")
            (invalid_dir / "bad-date.json").write_text(
                json.dumps({"trade_date": "2026-99-99", "rows": []}),
                encoding="utf-8",
            )
            (invalid_dir / "future.json").write_text(
                json.dumps({"trade_date": "2099-01-01", "rows": []}),
                encoding="utf-8",
            )

            try:
                result = self._builder()(history_root, ("s1", "s2", "s3", "s4"))
            except Exception as exc:
                self.fail(f"单个异常历史文件不应中断整个共识统计：{exc}")

            self.assertEqual([], result["trade_dates"])
            self.assertEqual([], result["rows"])
            self.assertEqual(0, result["covered_days"])
            self.assertEqual(20, result["maximum_appearances"])
            self.assertEqual(7, result.get("skipped_snapshot_count"))


if __name__ == "__main__":
    unittest.main()
