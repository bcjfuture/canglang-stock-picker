from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_v2 import (
    align_fundamentals_point_in_time,
    normalize_quarterly_statements,
    save_frame_atomic,
)


def statement_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    income = pd.DataFrame(
        {
            "股票代码": ["000001"],
            "净利润": [10.0],
            "净利润同比": [30.0],
            "营业总收入": [100.0],
            "营业总收入同比": [20.0],
            "公告日期": ["2025-04-28"],
        }
    )
    balance = pd.DataFrame(
        {
            "股票代码": ["000001"],
            "货币资金": [30.0],
            "应收账款": [20.0],
            "存货": [10.0],
            "总资产": [200.0],
            "总资产同比": [8.0],
            "总负债": [80.0],
            "资产负债率": [40.0],
            "公告日期": ["2025-04-28"],
        }
    )
    cashflow = pd.DataFrame(
        {
            "股票代码": ["000001"],
            "经营性现金流净额": [12.0],
            "公告日期": ["2025-04-28"],
        }
    )
    return income, balance, cashflow


class HighPrecisionV2FundamentalTests(unittest.TestCase):
    def test_statement_normalization_keeps_dates_and_ratios(self) -> None:
        income, balance, cashflow = statement_frames()

        normalized = normalize_quarterly_statements(
            "20250331", income, balance, cashflow
        )
        row = normalized.set_index("code").loc["000001"]

        self.assertEqual(pd.Timestamp("2025-03-31"), row["report_date"])
        self.assertEqual(pd.Timestamp("2025-04-28"), row["announcement_date"])
        self.assertAlmostEqual(0.10, row["net_margin"])
        self.assertAlmostEqual(1.20, row["cfo_to_profit"])
        self.assertAlmostEqual(0.15, row["cash_to_assets"])
        self.assertAlmostEqual(0.10, row["receivables_to_assets"])
        self.assertAlmostEqual(0.05, row["inventory_to_assets"])

    def test_statement_normalization_accepts_akshare_prefixed_columns(self) -> None:
        income, balance, cashflow = statement_frames()
        balance = balance.rename(
            columns={
                "货币资金": "资产-货币资金",
                "应收账款": "资产-应收账款",
                "存货": "资产-存货",
                "总资产": "资产-总资产",
                "总资产同比": "资产-总资产同比",
                "总负债": "负债-总负债",
            }
        )
        cashflow = cashflow.rename(
            columns={
                "经营性现金流净额": "经营性现金流-现金流量净额"
            }
        )

        normalized = normalize_quarterly_statements(
            "20250331", income, balance, cashflow
        )
        row = normalized.iloc[0]

        self.assertAlmostEqual(0.15, row["cash_to_assets"])
        self.assertAlmostEqual(1.20, row["cfo_to_profit"])

    def test_atomic_frame_writer_round_trips_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fundamentals.pkl"
            expected = pd.DataFrame({"code": ["000001"], "value": [1.0]})

            save_frame_atomic(path, expected)

            pd.testing.assert_frame_equal(pd.read_pickle(path), expected)
            self.assertFalse(path.with_suffix(".pkl.tmp").exists())

    def test_statement_becomes_available_next_trading_day(self) -> None:
        income, balance, cashflow = statement_frames()
        fundamentals = normalize_quarterly_statements(
            "20250331", income, balance, cashflow
        )
        signals = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-04-28", "2025-04-29"]),
                "code": ["000001", "000001"],
            }
        )

        result = align_fundamentals_point_in_time(
            signals,
            fundamentals,
            pd.bdate_range("2025-04-25", "2025-04-30"),
        )

        self.assertEqual(1.0, result.iloc[0]["fundamental_missing"])
        self.assertEqual(0.0, result.iloc[1]["fundamental_missing"])
        self.assertAlmostEqual(30.0, result.iloc[1]["net_profit_yoy_raw"])

    def test_future_statement_mutation_does_not_change_past_signal(self) -> None:
        income, balance, cashflow = statement_frames()
        first = normalize_quarterly_statements(
            "20250331", income, balance, cashflow
        )
        later_income = income.copy()
        later_income["公告日期"] = "2025-08-30"
        later_income["净利润同比"] = 9999.0
        later = normalize_quarterly_statements(
            "20250630", later_income, balance.assign(公告日期="2025-08-30"),
            cashflow.assign(公告日期="2025-08-30")
        )
        fundamentals = pd.concat([first, later], ignore_index=True)
        signals = pd.DataFrame(
            {"date": [pd.Timestamp("2025-05-15")], "code": ["000001"]}
        )
        trading_dates = pd.bdate_range("2025-04-25", "2025-09-02")

        before = align_fundamentals_point_in_time(
            signals, fundamentals, trading_dates
        )
        changed = fundamentals.copy()
        changed.loc[
            changed["announcement_date"] > pd.Timestamp("2025-05-15"),
            "net_profit_yoy",
        ] = -9999.0
        after = align_fundamentals_point_in_time(
            signals, changed, trading_dates
        )

        pd.testing.assert_frame_equal(before, after)


if __name__ == "__main__":
    unittest.main()
