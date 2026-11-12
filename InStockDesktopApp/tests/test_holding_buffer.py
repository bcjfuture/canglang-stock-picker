from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from instock_app.settings import STRATEGIES, AppPaths
from workers.picker_worker import apply_holding_buffer, load_previous_result


SPEC = STRATEGIES["tech_attack"]


def ranked(count: int) -> list[dict]:
    return [{"代码": f"{index:06d}", "综合评分": 100 - index, "风险提示": ""} for index in range(1, count + 1)]


def payload(top_codes: list[int], all_rows: list[dict], trade_date: str = "2026-09-30") -> dict:
    by_code = {row["代码"]: row for row in all_rows}
    return {
        "latest_dates": {"综合选股": trade_date},
        "summary": {},
        "sheets": {
            SPEC.top_sheet: [dict(by_code[f"{code:06d}"]) for code in top_codes],
            "全部": all_rows,
        },
    }


def previous(codes: list[int], runs: int = 1, trade_date: str = "2026-09-29") -> dict:
    return {"trade_date": trade_date, "rows": [{"code": f"{code:06d}", "holding_runs": runs} for code in codes]}


class HoldingBufferTests(unittest.TestCase):
    def test_previous_pick_inside_buffer_is_kept(self) -> None:
        """Catches the list being rebuilt from scratch each day."""
        data = payload([1, 2, 3], ranked(60))
        apply_holding_buffer(SPEC, data, previous([30]), {"buffer_rank": 40}, rank_sheet_name="全部", top_n=3)

        codes = [row["代码"] for row in data["sheets"][SPEC.top_sheet]]
        self.assertEqual(["000001", "000002", "000030"], codes)
        kept = data["sheets"][SPEC.top_sheet][-1]
        self.assertEqual("延续持有", kept["入选层级"])
        self.assertEqual(2, kept["持有次数"])

    def test_pick_outside_buffer_or_held_too_long_is_dropped(self) -> None:
        data = payload([1, 2], ranked(60))
        apply_holding_buffer(SPEC, data, previous([50]), {"buffer_rank": 40}, rank_sheet_name="全部", top_n=3)
        self.assertNotIn("000050", [row["代码"] for row in data["sheets"][SPEC.top_sheet]])

        data = payload([1, 2], ranked(60))
        apply_holding_buffer(
            SPEC, data, previous([10], runs=15), {"buffer_rank": 40, "max_hold_runs": 15}, rank_sheet_name="全部", top_n=3
        )
        self.assertNotIn("000010", [row["代码"] for row in data["sheets"][SPEC.top_sheet]])

    def test_sell_signal_forces_exit(self) -> None:
        rows = ranked(60)
        rows[9]["风险提示"] = "指标卖出信号"
        data = payload([1], rows)
        apply_holding_buffer(SPEC, data, previous([10]), {"buffer_rank": 40}, rank_sheet_name="全部", top_n=3)
        self.assertEqual(["000001"], [row["代码"] for row in data["sheets"][SPEC.top_sheet]])

    def test_retained_names_fill_an_empty_strict_list(self) -> None:
        data = payload([], ranked(60))
        data["summary"]["abstain_reason"] = "今日没有通过严格条件的候选，保持观望"
        apply_holding_buffer(SPEC, data, previous([5]), {"buffer_rank": 40}, rank_sheet_name="全部", top_n=3)
        self.assertEqual(["000005"], [row["代码"] for row in data["sheets"][SPEC.top_sheet]])
        self.assertEqual("", data["summary"]["abstain_reason"])

    def test_same_day_rerun_is_not_its_own_predecessor(self) -> None:
        """Catches a same-day re-run resetting the holding count."""
        with tempfile.TemporaryDirectory() as temporary:
            paths = replace(AppPaths.default(), data_dir=Path(temporary))
            history = paths.data_dir / "store" / "history" / SPEC.key
            history.mkdir(parents=True)
            (history / "a.json").write_text(json.dumps(previous([7], trade_date="2026-09-29")), encoding="utf-8")
            (history / "b.json").write_text(json.dumps(previous([8], trade_date="2026-09-30")), encoding="utf-8")

            found = load_previous_result(paths, SPEC, "2026-09-30")

        self.assertEqual("2026-09-29", found["trade_date"])


if __name__ == "__main__":
    unittest.main()
