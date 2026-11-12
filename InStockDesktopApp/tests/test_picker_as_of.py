from __future__ import annotations

import sys
import tempfile
import unittest
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "engine"))

import instock_tech_picker_data as picker_engine
from instock_app.settings import AppPaths, STRATEGIES
from workers.picker_worker import run_picker


class RecordingCursor:
    def __init__(self, value: date) -> None:
        self.value = value
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql: str, params=None) -> None:
        self.calls.append((sql, params))

    def fetchone(self):
        return {"n": 1, "d": self.value}


class PickerAsOfTests(unittest.TestCase):
    def test_latest_date_is_bounded_by_as_of_date(self) -> None:
        cursor = RecordingCursor(date(2026, 7, 13))

        result = picker_engine.latest_date(
            cursor, "cn_stock_selection", date(2026, 7, 13)
        )

        self.assertEqual(date(2026, 7, 13), result)
        sql, params = cursor.calls[-1]
        self.assertIn("MAX(`date`)", sql)
        self.assertIn("`date` <= %s", sql)
        self.assertEqual((date(2026, 7, 13),), params)

    def test_run_picker_passes_as_of_date_to_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = replace(
                AppPaths.default(),
                data_dir=root / "data",
                engine_dir=ROOT / "engine",
            )
            spec = STRATEGIES["all_market"]
            captured: list[list[str]] = []

            def fake_run(command, **_kwargs):
                captured.append([str(value) for value in command])
                raw_path = Path(command[4])
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(
                    json.dumps(
                        {
                            "generated_at": "2026-07-09 15:30:00",
                            "latest_dates": {"综合选股": "2026-07-09"},
                            "strategy_gate_passed": False,
                            "summary": {},
                            "sheets": {
                                spec.top_sheet: [
                                    {
                                        "代码": "000001",
                                        "综合评分": 88,
                                        "名称": "测试股",
                                        "行业": "银行",
                                        "最新价": 10,
                                        "涨跌幅%": 1,
                                        "今日主力净流入_万元": 100,
                                        "加分理由": "测试",
                                        "风险提示": "",
                                        "概念": "",
                                        "ATR14": 0.5,
                                    }
                                ]
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("workers.picker_worker.subprocess.run", side_effect=fake_run):
                result = run_picker(
                    "all_market",
                    paths,
                    lambda *_args: None,
                    as_of_date="2026-07-09",
                )

            engine_commands = [
                command
                for command in captured
                if command[3].endswith("instock_tech_picker_data.py")
            ]
            self.assertEqual("2026-07-09", engine_commands[0][-1])
            self.assertEqual("2026-07-09", result["trade_date"])

    def test_load_selection_snapshot_reuses_bounded_source_helpers(self) -> None:
        class Cursor:
            def __init__(self) -> None:
                self.calls = []

            def execute(self, sql, params=None) -> None:
                self.calls.append((sql, params))

            def fetchall(self):
                return [{"date": date(2026, 7, 13), "code": "000001"}]

        cursor = Cursor()
        with (
            patch(
                "instock_tech_picker_data.latest_date",
                side_effect=[
                    date(2026, 7, 13),
                    date(2026, 7, 13),
                    date(2026, 7, 13),
                ],
            ) as latest,
            patch(
                "instock_tech_picker_data.fetch_code_set",
                side_effect=[{"000001"}, {"000002"}, {"000003"}],
            ),
            patch(
                "instock_tech_picker_data.fetch_strategy_hits",
                return_value={"000001": ["均线多头"]},
            ),
        ):
            result = picker_engine.load_selection_snapshot(
                cursor, date(2026, 7, 13)
            )

        self.assertEqual(date(2026, 7, 13), result["selection_date"])
        self.assertEqual({"000001"}, result["spot_buy"])
        self.assertEqual({"000002"}, result["indicator_buy"])
        self.assertEqual({"000003"}, result["indicator_sell"])
        self.assertEqual({"000001": ["均线多头"]}, result["strategy_hits"])
        self.assertEqual("000001", result["rows"][0]["code"])
        self.assertTrue(
            all(call.args[2] == date(2026, 7, 13) for call in latest.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
