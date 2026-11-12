from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

import instock_tech_picker_data
from picker_news import NewsFactorResult


class PickerFundScoringTests(unittest.TestCase):
    def fund_score(self, one_day: float, three_day: float, five_day: float):
        self.assertTrue(hasattr(instock_tech_picker_data, "score_fund_flows"))
        return instock_tech_picker_data.score_fund_flows(one_day, three_day, five_day)

    def test_continuous_inflow_is_an_important_positive_factor(self) -> None:
        score, reasons, risks = self.fund_score(200_000_000, 500_000_000, 800_000_000)
        self.assertGreaterEqual(score, 24)
        self.assertIn("资金连续净流入", reasons)
        self.assertEqual(risks, [])

    def test_continuous_outflow_has_symmetric_material_penalty(self) -> None:
        score, reasons, risks = self.fund_score(-200_000_000, -500_000_000, -800_000_000)
        self.assertLessEqual(score, -20)
        self.assertEqual(reasons, [])
        self.assertIn("资金连续净流出", risks)

    def test_mixed_periods_explain_both_positive_and_negative_signals(self) -> None:
        score, reasons, risks = self.fund_score(100_000_000, -100_000_000, 200_000_000)
        self.assertGreater(score, 0)
        self.assertIn("1日资金净流入", reasons)
        self.assertIn("5日资金净流入", reasons)
        self.assertIn("3日资金净流出", risks)


class PickerShareholderScoringTests(unittest.TestCase):
    def shareholder_score(self, **values):
        self.assertTrue(hasattr(instock_tech_picker_data, "score_shareholder_structure"))
        return instock_tech_picker_data.score_shareholder_structure(**values)

    def test_moderate_institutional_holding_and_fewer_holders_score_positive(self) -> None:
        score, reasons, risks, summary = self.shareholder_score(
            institutional_ratio=28.0,
            institutional_count=35,
            holder_count_change=-6.0,
            top10_ratio=55.0,
            survey_count=12,
        )
        self.assertGreaterEqual(score, 10)
        self.assertIn("机构持股适中", reasons)
        self.assertIn("股东户数下降", reasons)
        self.assertEqual(risks, [])
        self.assertIn("机构持股28.00%", summary)

    def test_excessive_concentration_and_holder_growth_are_penalized(self) -> None:
        score, reasons, risks, _summary = self.shareholder_score(
            institutional_ratio=92.0,
            institutional_count=2,
            holder_count_change=18.0,
            top10_ratio=93.0,
            survey_count=0,
        )
        self.assertLessEqual(score, -8)
        self.assertIn("股东户数明显增加", risks)
        self.assertIn("股权集中度过高", risks)

    def test_missing_shareholder_data_is_neutral_and_honest(self) -> None:
        score, reasons, risks, summary = self.shareholder_score(
            institutional_ratio=None,
            institutional_count=None,
            holder_count_change=None,
            top10_ratio=None,
            survey_count=None,
        )
        self.assertEqual(score, 0)
        self.assertEqual(reasons, [])
        self.assertEqual(risks, [])
        self.assertEqual(summary, "股东结构数据不足")


class PickerNewsIntegrationTests(unittest.TestCase):
    def test_news_factor_changes_score_and_explanation(self) -> None:
        """Catches a computed news score being dropped before final ranking."""
        row = {"综合评分": 70.0, "加分理由": "资金流入", "风险提示": ""}
        result = NewsFactorResult(6.0, "已评估", 2, 2, 0, False, "利好：回购；利好：中标")

        updated = instock_tech_picker_data.apply_news_factor(
            row, result, signal_date=date(2026, 8, 12)
        )

        self.assertEqual(updated["综合评分"], 76.0)
        self.assertIn("新闻利好+6.0分", updated["加分理由"])
        self.assertEqual(updated["新闻截止日"], "2026-08-12")

    def test_hard_news_risk_is_visible_and_material(self) -> None:
        """Catches a hard-risk penalty without a corresponding user-facing warning."""
        row = {"综合评分": 70.0, "加分理由": "", "风险提示": ""}
        result = NewsFactorResult(-20.0, "已评估", 1, 0, 1, True, "重大风险：立案调查")

        updated = instock_tech_picker_data.apply_news_factor(
            row, result, signal_date=date(2026, 8, 12)
        )

        self.assertEqual(updated["综合评分"], 50.0)
        self.assertIn("重大新闻风险", updated["风险提示"])
        self.assertTrue(updated["新闻重大风险"])


if __name__ == "__main__":
    unittest.main()
