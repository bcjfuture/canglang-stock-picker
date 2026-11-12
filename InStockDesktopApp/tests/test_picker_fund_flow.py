from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "src"))

import instock_tech_picker_data
from workers.export_worker import COLUMNS


class PickerFundFlowTests(unittest.TestCase):
    def resolve_grouped_flows(self, **values):
        self.assertTrue(
            hasattr(instock_tech_picker_data, "resolve_grouped_flows_wan"),
            "engine must expose the grouped component resolver",
        )
        return instock_tech_picker_data.resolve_grouped_flows_wan(**values)

    def test_grouped_flows_sum_super_large_and_medium_small(self) -> None:
        self.assertEqual(
            self.resolve_grouped_flows(
                fund_amount=150_000_000,
                fund_amount_super=120_000_000,
                fund_amount_large=30_000_000,
                fund_amount_medium=-10_000_000,
                fund_amount_small=-20_000_000,
            ),
            (15000.0, -3000.0, "数据库分项"),
        )

    def test_missing_direct_components_use_same_day_main_flow_fallback(self) -> None:
        self.assertEqual(
            self.resolve_grouped_flows(
                fund_amount=150_000_000,
                fund_amount_super=None,
                fund_amount_large=None,
                fund_amount_medium=None,
                fund_amount_small=None,
            ),
            (15000.0, -15000.0, "数据库汇总推导"),
        )

    def test_incomplete_group_is_not_partially_reported(self) -> None:
        self.assertEqual(
            self.resolve_grouped_flows(
                fund_amount=150_000_000,
                fund_amount_super=99_000_000,
                fund_amount_large=12_340_000,
                fund_amount_medium=-87_000_000,
                fund_amount_small=None,
            ),
            (11134.0, None, "数据库分项"),
        )

    def test_missing_components_and_main_flow_remain_unavailable(self) -> None:
        self.assertEqual(
            self.resolve_grouped_flows(
                fund_amount=None,
                fund_amount_super=None,
                fund_amount_large=None,
                fund_amount_medium=None,
                fund_amount_small=None,
            ),
            (None, None, "分项暂缺"),
        )

    def test_excel_columns_expose_both_flow_groups(self) -> None:
        titles = {key: title for title, key, _width in COLUMNS}
        keys = list(titles)
        self.assertIn("large_order_flow", keys)
        self.assertIn("retail_flow", keys)
        self.assertEqual(titles["large_order_flow"], "大单净额(超大单+大单,万元)")
        self.assertEqual(titles["retail_flow"], "散户净额(中单+小单,万元)")
        self.assertEqual(titles["fund_flow_1d"], "1日资金净流向(万元)")
        self.assertEqual(titles["fund_flow_3d"], "3日资金净流向(万元)")
        self.assertEqual(titles["fund_flow_5d"], "5日资金净流向(万元)")
        self.assertEqual(titles["shareholder_score"], "股东结构评分")
        self.assertEqual(titles["institutional_holding_ratio"], "机构持股比例%")
        self.assertEqual(titles["shareholder_count_change"], "股东户数变化%")


if __name__ == "__main__":
    unittest.main()
