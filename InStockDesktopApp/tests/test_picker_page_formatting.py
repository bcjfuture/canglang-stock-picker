from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import instock_app.ui.picker_page as picker_page
from instock_app.ui.picker_page import (
    build_picker_columns,
    format_order_flow_detail,
    format_picker_row,
    format_shareholder_review,
    format_signed_flow,
    summarize_reason,
)


class PickerPageFormattingTests(unittest.TestCase):
    def test_signed_flow_uses_wan_and_yi_units(self) -> None:
        self.assertEqual(format_signed_flow(1234), "+1,234万")
        self.assertEqual(format_signed_flow(-856), "-856万")
        self.assertEqual(format_signed_flow(12345), "+1.23亿")
        self.assertEqual(format_signed_flow(None), "—")

    def test_reason_summary_keeps_three_unique_items(self) -> None:
        summary = summarize_reason("当日上涨；量比放大；成交额充足；当日上涨；资金净流入")
        self.assertEqual(summary, "✓ 当日上涨；量比放大；成交额充足；等1项")

    def test_picker_columns_use_compact_summary_fields(self) -> None:
        columns = build_picker_columns({"all_market": object()})
        self.assertIn("large_order_flow", columns)
        self.assertIn("retail_flow", columns)
        self.assertIn("fund_flow_1d", columns)
        self.assertIn("fund_flow_3d", columns)
        self.assertIn("fund_flow_5d", columns)
        self.assertIn("shareholder_score", columns)
        self.assertIn("news_score", columns)
        self.assertIn("news_summary", columns)
        self.assertIn("reason", columns)
        self.assertNotIn("fund_flow", columns)
        self.assertNotIn("performance_review", columns)
        self.assertNotIn("risk", columns)

    def test_market_earnings_columns_are_visible_and_formatted(self) -> None:
        """Catches PB and PR3 being available only in exported/raw data."""
        columns = build_picker_columns({"all_market": object()})
        self.assertIn("pb", columns)
        self.assertIn("market_earnings_ratio", columns)
        headings = picker_page.build_picker_headings()
        self.assertEqual(headings["pb"], "PB")
        self.assertEqual(headings["market_earnings_ratio"], "市赚率3.0")

        original = picker_page.PickerPage.COLUMNS
        picker_page.PickerPage.COLUMNS = columns
        try:
            values = format_picker_row({"pb": 0.81234, "market_earnings_ratio": 0.56789})
            row = dict(zip(columns, values))
        finally:
            picker_page.PickerPage.COLUMNS = original
        self.assertEqual(row["pb"], "0.81")
        self.assertEqual(row["market_earnings_ratio"], "0.568")

    def test_picker_headings_match_dynamic_flow_units_and_core_reason(self) -> None:
        self.assertTrue(
            hasattr(picker_page, "build_picker_headings"),
            "picker page must expose the headings used by the table",
        )
        headings = picker_page.build_picker_headings()
        self.assertEqual(headings["large_order_flow"], "大单净额")
        self.assertEqual(headings["retail_flow"], "散户净额")
        self.assertEqual(headings["fund_flow_1d"], "1日净流向")
        self.assertEqual(headings["fund_flow_3d"], "3日净流向")
        self.assertEqual(headings["fund_flow_5d"], "5日净流向")
        self.assertEqual(headings["shareholder_score"], "股东分")
        self.assertEqual(headings["news_score"], "新闻分")
        self.assertEqual(headings["news_summary"], "新闻摘要")
        self.assertEqual(headings["reason"], "核心理由")

    def test_order_flow_detail_explains_components_source_and_date(self) -> None:
        detail = format_order_flow_detail(
            {
                "large_order_flow": 50638.1024,
                "retail_flow": -28276.5216,
                "order_flow_as_of": "2026-07-30",
                "order_flow_source": "实时分项",
            }
        )
        self.assertIn("大单净额（净超大单+净大单） +5.06亿", detail)
        self.assertIn("散户净额（净中单+净小单） -2.83亿", detail)
        self.assertIn("2026-07-30", detail)
        self.assertIn("实时分项", detail)

    def test_missing_order_flow_detail_is_honest(self) -> None:
        detail = format_order_flow_detail(
            {
                "large_order_flow": None,
                "retail_flow": None,
                "order_flow_as_of": None,
                "order_flow_source": "分项暂缺",
            }
        )
        self.assertIn("大单净额（净超大单+净大单） —", detail)
        self.assertIn("散户净额（净中单+净小单） —", detail)
        self.assertIn("分项暂缺", detail)

    def test_picker_row_formats_all_multiday_flow_columns(self) -> None:
        original = picker_page.PickerPage.COLUMNS
        picker_page.PickerPage.COLUMNS = build_picker_columns({"all_market": object()})
        try:
            values = format_picker_row(
                {
                    "fund_flow_1d": 1200,
                    "fund_flow_3d": -25000,
                    "fund_flow_5d": 0,
                }
            )
            row = dict(zip(picker_page.PickerPage.COLUMNS, values))
        finally:
            picker_page.PickerPage.COLUMNS = original
        self.assertEqual(row["fund_flow_1d"], "+1,200万")
        self.assertEqual(row["fund_flow_3d"], "-2.50亿")
        self.assertEqual(row["fund_flow_5d"], "0万")

    def test_shareholder_review_discloses_score_and_periodic_nature(self) -> None:
        detail = format_shareholder_review(
            {
                "shareholder_score": 11.5,
                "shareholder_review": "机构持股28.00%（35家）；股东户数变化-6.00%",
            }
        )
        self.assertIn("股东结构 +11.5分", detail)
        self.assertIn("机构持股28.00%", detail)
        self.assertIn("定期披露口径", detail)

    def test_picker_row_formats_news_score_and_summary(self) -> None:
        """Catches normalized news evidence being invisible in the result table."""
        original = picker_page.PickerPage.COLUMNS
        picker_page.PickerPage.COLUMNS = build_picker_columns({"all_market": object()})
        try:
            values = format_picker_row(
                {
                    "news_score": -20,
                    "news_summary": "重大风险：公司因重大违法被立案调查",
                    "news_status": "已评估",
                }
            )
            row = dict(zip(picker_page.PickerPage.COLUMNS, values))
        finally:
            picker_page.PickerPage.COLUMNS = original
        self.assertEqual(row["news_score"], "-20.0")
        self.assertIn("立案调查", row["news_summary"])

    def test_unavailable_news_state_is_rendered_honestly(self) -> None:
        """Catches a provider outage being shown as an evaluated neutral result."""
        original = picker_page.PickerPage.COLUMNS
        picker_page.PickerPage.COLUMNS = build_picker_columns({"all_market": object()})
        try:
            values = format_picker_row(
                {"news_score": 0, "news_summary": "", "news_status": "资讯不可用"}
            )
            row = dict(zip(picker_page.PickerPage.COLUMNS, values))
        finally:
            picker_page.PickerPage.COLUMNS = original
        self.assertEqual(row["news_summary"], "资讯不可用（按0分处理）")


if __name__ == "__main__":
    unittest.main()
