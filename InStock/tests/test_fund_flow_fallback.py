import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from instock.core.fund_flow_fallback import build_individual_rank, build_sector_rank
import instock.core.stockfetch as stockfetch
import instock.core.tablestructure as tablestructure
import instock.job.basic_data_other_daily_job as fund_flow_job


class FundFlowFallbackTests(unittest.TestCase):
    def test_builds_current_individual_flow_with_units_and_leading_zero_code(self):
        raw = pd.DataFrame([{
            "股票代码": 1,
            "股票简称": "平安银行",
            "最新价": 11.25,
            "涨跌幅": "2.50%",
            "流入资金": "2.00亿",
            "流出资金": "1.25亿",
            "净额": "7500.00万",
            "成交额": "5.00亿",
        }])

        result = build_individual_rank(raw, 0)

        row = result.iloc[0]
        self.assertEqual(row["code"], "000001")
        self.assertEqual(row["fund_amount"], 75_000_000)
        self.assertAlmostEqual(row["fund_rate"], 15.0)
        self.assertEqual(row["change_rate"], 2.5)

    def test_builds_multi_day_individual_flow(self):
        raw = pd.DataFrame([{
            "股票代码": 600000,
            "股票简称": "浦发银行",
            "最新价": 8.61,
            "阶段涨跌幅": "-1.37%",
            "资金流入净额": "-1.50亿",
        }])

        result = build_individual_rank(raw, 2)

        row = result.iloc[0]
        self.assertEqual(row["code"], "600000")
        self.assertEqual(row["fund_amount_5"], -150_000_000)
        self.assertEqual(row["change_rate_5"], -1.37)

    def test_builds_current_sector_flow(self):
        raw = pd.DataFrame([{
            "行业": "半导体",
            "行业-涨跌幅": 6.32,
            "流入资金": 2821.23,
            "流出资金": 2515.92,
            "净额": 305.31,
            "领涨股": "格科微",
        }])

        result = build_sector_rank(raw, 0)

        row = result.iloc[0]
        self.assertEqual(row["name"], "半导体")
        self.assertEqual(row["fund_amount"], 30_531_000_000)
        self.assertAlmostEqual(row["fund_rate"], 305.31 / (2821.23 + 2515.92) * 100)
        self.assertEqual(row["stock_name"], "格科微")

    def test_stockfetch_uses_independent_fund_flow_source_when_primary_fails(self):
        fallback = build_individual_rank(pd.DataFrame([{
            "股票代码": 600000,
            "股票简称": "浦发银行",
            "最新价": 8.61,
            "涨跌幅": "-1.37%",
            "流入资金": "1.00亿",
            "流出资金": "1.50亿",
            "净额": "-5000万",
            "成交额": "5.00亿",
        }]), 0)

        with patch.object(
            stockfetch.sff,
            "stock_individual_fund_flow_rank",
            side_effect=ConnectionError("primary unavailable"),
        ), patch(
            "instock.core.fund_flow_fallback.fetch_individual_rank",
            return_value=fallback,
        ):
            result = stockfetch.fetch_stocks_fund_flow(0)

        self.assertEqual(len(result.index), 1)
        self.assertEqual(result.iloc[0]["code"], "600000")

    def test_stockfetch_deduplicates_fallback_rows_by_code(self):
        columns = list(tablestructure.CN_STOCK_FUND_FLOW[0]["columns"])
        first = {column: None for column in columns}
        first.update({"code": "600141", "name": "兴发集团", "new_price": 21.10})
        second = first.copy()
        second["new_price"] = 21.25
        fallback = pd.DataFrame([first, second], columns=columns)

        with patch.object(
            stockfetch.sff,
            "stock_individual_fund_flow_rank",
            side_effect=ConnectionError("primary unavailable"),
        ), patch(
            "instock.core.fund_flow_fallback.fetch_individual_rank",
            return_value=fallback,
        ):
            result = stockfetch.fetch_stocks_fund_flow(0)

        self.assertEqual(len(result.index), 1)
        self.assertEqual(result.iloc[0]["code"], "600141")
        self.assertEqual(result.iloc[0]["new_price"], 21.25)

    def test_stockfetch_deduplicates_sector_fallback_rows_by_name(self):
        columns = list(tablestructure.CN_STOCK_SECTOR_FUND_FLOW[1][2]["columns"])
        first = {column: None for column in columns}
        first.update({"name": "风电设备", "change_rate_10": -2.29})
        second = first.copy()
        second["change_rate_10"] = -2.26
        fallback = pd.DataFrame([first, second], columns=columns)

        with patch.object(
            stockfetch.sff,
            "stock_sector_fund_flow_rank",
            side_effect=ConnectionError("primary unavailable"),
        ), patch(
            "instock.core.fund_flow_fallback.fetch_sector_rank",
            return_value=fallback,
        ):
            result = stockfetch.fetch_stocks_sector_fund_flow(0, 2)

        self.assertEqual(len(result.index), 1)
        self.assertEqual(result.iloc[0]["name"], "风电设备")
        self.assertEqual(result.iloc[0]["change_rate_10"], -2.26)

    def test_job_does_not_delete_or_insert_when_merged_codes_are_not_unique(self):
        duplicate = pd.DataFrame([
            {"code": "600141", "name": "兴发集团", "new_price": 21.10},
            {"code": "600141", "name": "兴发集团", "new_price": 21.25},
        ])

        with patch.object(
            fund_flow_job,
            "run_check_stock_fund_flow",
            return_value={0: duplicate},
        ), patch.object(
            fund_flow_job.mdb,
            "checkTableIsExist",
        ) as check_table, patch.object(
            fund_flow_job.mdb,
            "executeSql",
        ) as execute_sql, patch.object(
            fund_flow_job.mdb,
            "insert_db_from_df",
        ) as insert_rows:
            fund_flow_job.save_nph_stock_fund_flow_data(
                datetime(2026, 7, 2),
                before=False,
            )

        check_table.assert_not_called()
        execute_sql.assert_not_called()
        insert_rows.assert_not_called()

    def test_sector_job_does_not_delete_or_insert_when_merged_names_are_not_unique(self):
        duplicate = pd.DataFrame([
            {"name": "通信服务", "change_rate": 1.10},
            {"name": "通信服务", "change_rate": 1.20},
        ])

        with patch.object(
            fund_flow_job,
            "run_check_stock_sector_fund_flow",
            return_value={0: duplicate},
        ), patch.object(
            fund_flow_job.mdb,
            "checkTableIsExist",
        ) as check_table, patch.object(
            fund_flow_job.mdb,
            "executeSql",
        ) as execute_sql, patch.object(
            fund_flow_job.mdb,
            "insert_db_from_df",
        ) as insert_rows:
            fund_flow_job.stock_sector_fund_flow_data(
                datetime(2026, 7, 3),
                0,
            )

        check_table.assert_not_called()
        execute_sql.assert_not_called()
        insert_rows.assert_not_called()


if __name__ == "__main__":
    unittest.main()
