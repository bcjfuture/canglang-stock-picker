from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

import zhengxi_prosperity_core as core


point_in_time_fundamentals = core.point_in_time_fundamentals


TEST_CONFIG = {
    "reject_name_keywords": ["ST", "*ST", "退"],
    "minimum_deal_amount": 100_000_000,
    "minimum_average_amount_20": 100_000_000,
    "minimum_turnover": 0.3,
    "minimum_history_days": 40,
}


def sample_scoring_rows() -> pd.DataFrame:
    common = {
        "new_price": 20.0,
        "change_rate": 2.0,
        "turnoverrate": 2.0,
        "deal_amount": 300_000_000,
        "average_amount_20": 280_000_000,
        "history_days": 80,
        "pe9": 25.0,
        "pbnewmrq": 2.0,
        "fund_amount_5": 100_000_000,
        "above_ma20": 1.0,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "code": "000001",
                "name": "修复公司",
                "industry": "强景气行业",
                "roe_weight": 4.0,
                "revenue_yoy": 22.0,
                "net_profit_yoy": 30.0,
                "net_margin": 0.12,
                "net_margin_prev": 0.08,
                "cfo_to_profit": 1.10,
                "cfo_to_profit_prev": 0.70,
                "debt_ratio": 0.35,
                "debt_ratio_prev": 0.38,
                "receivables_to_assets": 0.10,
                "receivables_to_assets_prev": 0.12,
                "inventory_to_assets": 0.15,
                "inventory_to_assets_prev": 0.16,
                "mom_20": 0.16,
                "mom_60": 0.31,
            },
            {
                **common,
                "code": "000002",
                "name": "恶化公司",
                "industry": "强景气行业",
                "roe_weight": 3.0,
                "revenue_yoy": -12.0,
                "net_profit_yoy": -25.0,
                "net_margin": 0.03,
                "net_margin_prev": 0.06,
                "cfo_to_profit": 0.10,
                "cfo_to_profit_prev": 0.50,
                "debt_ratio": 0.75,
                "debt_ratio_prev": 0.65,
                "receivables_to_assets": 0.25,
                "receivables_to_assets_prev": 0.18,
                "inventory_to_assets": 0.30,
                "inventory_to_assets_prev": 0.20,
                "mom_20": 0.10,
                "mom_60": 0.20,
            },
            {
                **common,
                "code": "000003",
                "name": "成熟公司",
                "industry": "强景气行业",
                "roe_weight": 25.0,
                "revenue_yoy": 2.0,
                "net_profit_yoy": 1.0,
                "net_margin": 0.15,
                "net_margin_prev": 0.15,
                "cfo_to_profit": 0.90,
                "cfo_to_profit_prev": 0.90,
                "debt_ratio": 0.30,
                "debt_ratio_prev": 0.30,
                "receivables_to_assets": 0.10,
                "receivables_to_assets_prev": 0.10,
                "inventory_to_assets": 0.10,
                "inventory_to_assets_prev": 0.10,
                "mom_20": 0.12,
                "mom_60": 0.22,
            },
            {
                **common,
                "code": "000010",
                "name": "弱行业强图形",
                "industry": "弱景气行业",
                "roe_weight": 5.0,
                "revenue_yoy": -15.0,
                "net_profit_yoy": -18.0,
                "net_margin": 0.04,
                "net_margin_prev": 0.05,
                "cfo_to_profit": 0.40,
                "cfo_to_profit_prev": 0.50,
                "debt_ratio": 0.60,
                "debt_ratio_prev": 0.55,
                "receivables_to_assets": 0.22,
                "receivables_to_assets_prev": 0.20,
                "inventory_to_assets": 0.24,
                "inventory_to_assets_prev": 0.20,
                "mom_20": 0.03,
                "mom_60": -0.12,
                "change_rate": 6.0,
            },
            {
                **common,
                "code": "000011",
                "name": "弱行业同伴",
                "industry": "弱景气行业",
                "roe_weight": 6.0,
                "revenue_yoy": -20.0,
                "net_profit_yoy": -30.0,
                "net_margin": 0.02,
                "net_margin_prev": 0.04,
                "cfo_to_profit": 0.20,
                "cfo_to_profit_prev": 0.40,
                "debt_ratio": 0.68,
                "debt_ratio_prev": 0.60,
                "receivables_to_assets": 0.24,
                "receivables_to_assets_prev": 0.20,
                "inventory_to_assets": 0.28,
                "inventory_to_assets_prev": 0.22,
                "mom_20": -0.08,
                "mom_60": -0.20,
                "above_ma20": 0.0,
            },
        ]
    )


class PointInTimeFundamentalsTests(unittest.TestCase):
    def test_future_announcement_is_excluded_and_same_quarter_last_year_is_compared(self) -> None:
        """Catches selecting a report that was not public on the signal date."""
        statements = pd.DataFrame(
            [
                {
                    "code": "000001",
                    "report_date": "2025-03-31",
                    "announcement_date": "2025-04-28",
                    "net_margin": 0.08,
                },
                {
                    "code": "000001",
                    "report_date": "2025-12-31",
                    "announcement_date": "2026-03-20",
                    "net_margin": 0.30,
                },
                {
                    "code": "000001",
                    "report_date": "2026-03-31",
                    "announcement_date": "2026-04-30",
                    "net_margin": 0.12,
                },
                {
                    "code": "000001",
                    "report_date": "2026-06-30",
                    "announcement_date": "2026-08-30",
                    "net_margin": 0.20,
                },
            ]
        )

        result = point_in_time_fundamentals(statements, "2026-08-03").set_index("code")

        self.assertEqual(pd.Timestamp("2026-03-31"), result.loc["000001", "report_date"])
        self.assertEqual(0.12, result.loc["000001", "net_margin"])
        self.assertEqual(0.08, result.loc["000001", "net_margin_prev"])

    def test_revised_older_report_does_not_replace_latest_period(self) -> None:
        """A late revision of last year's report must not become the current statement."""
        statements = pd.DataFrame(
            [
                {"code": "000001", "report_date": "2025-06-30", "announcement_date": "2025-08-20", "net_margin": 0.05},
                {"code": "000001", "report_date": "2026-06-30", "announcement_date": "2026-08-20", "net_margin": 0.10},
                {"code": "000001", "report_date": "2025-12-31", "announcement_date": "2026-09-01", "net_margin": 0.07},
            ]
        )

        result = point_in_time_fundamentals(statements, "2026-09-10").set_index("code")

        self.assertEqual(pd.Timestamp("2026-06-30"), result.loc["000001", "report_date"])
        self.assertEqual(0.05, result.loc["000001", "net_margin_prev"])

    def test_missing_prior_year_period_leaves_comparison_empty(self) -> None:
        statements = pd.DataFrame(
            [
                {"code": "000001", "report_date": "2025-12-31", "announcement_date": "2026-03-20", "net_margin": 0.08},
                {"code": "000001", "report_date": "2026-03-31", "announcement_date": "2026-04-30", "net_margin": 0.12},
            ]
        )

        result = point_in_time_fundamentals(statements, "2026-05-10").set_index("code")

        self.assertTrue(pd.isna(result.loc["000001", "net_margin_prev"]))


class ZhengxiScoringTests(unittest.TestCase):
    def score(self) -> pd.DataFrame:
        self.assertTrue(
            hasattr(core, "score_candidates"),
            "the Zhengxi scorer must be implemented",
        )
        return core.score_candidates(sample_scoring_rows(), TEST_CONFIG).set_index("code")

    def test_good_low_roe_repair_beats_structurally_bad_low_roe(self) -> None:
        """Catches treating every low-ROE company as a repair opportunity."""
        scored = self.score()

        self.assertGreater(
            scored.loc["000001", "roe_repair_score"],
            scored.loc["000002", "roe_repair_score"],
        )
        self.assertIn("结构性低ROE风险", scored.loc["000002", "risk"])

    def test_high_roe_alone_does_not_win_repair_score(self) -> None:
        """Catches reverting to the old rule that mechanically rewards high ROE."""
        scored = self.score()

        self.assertGreater(
            scored.loc["000001", "roe_repair_score"],
            scored.loc["000003", "roe_repair_score"],
        )

    def test_industry_prosperity_beats_single_stock_chart_strength(self) -> None:
        """Catches allowing one technical spike to override a weak industry cycle."""
        scored = self.score()

        self.assertGreater(
            scored.loc["000001", "total_score"],
            scored.loc["000010", "total_score"],
        )
        self.assertGreater(
            scored.loc["000001", "industry_prosperity_score"],
            scored.loc["000010", "industry_prosperity_score"],
        )

    def test_low_liquidity_is_a_hard_exclusion(self) -> None:
        """Catches fallback selection relaxing the liquidity safety boundary."""
        rows = sample_scoring_rows()
        illiquid = rows.iloc[0].copy()
        illiquid["code"] = "000099"
        illiquid["deal_amount"] = 20_000_000
        illiquid["average_amount_20"] = 30_000_000
        scored = core.score_candidates(
            pd.concat([rows, illiquid.to_frame().T], ignore_index=True),
            TEST_CONFIG,
        )

        self.assertNotIn("000099", set(scored["code"]))

    def test_moderate_trend_beats_overheated_momentum(self) -> None:
        """Catches paying the highest score after a trend becomes crowded."""
        base = sample_scoring_rows().iloc[0].copy()
        moderate = base.copy()
        moderate["code"] = "000101"
        moderate["mom_20"] = 0.08
        moderate["mom_60"] = 0.20
        overheated = base.copy()
        overheated["code"] = "000102"
        overheated["mom_20"] = 0.55
        overheated["mom_60"] = 1.00

        scored = core.score_candidates(
            pd.DataFrame([moderate, overheated]), TEST_CONFIG
        ).set_index("code")

        self.assertGreater(
            scored.loc["000101", "total_score"],
            scored.loc["000102", "total_score"],
        )
        self.assertIn("动量过热", scored.loc["000102", "risk"])


def make_scored_universe(count: int, strict_count: int = 7) -> pd.DataFrame:
    rows = []
    for index in range(count):
        if index < strict_count:
            total, industry, repair = 74.0 - index * 0.1, 30.0, 16.0
        elif index < strict_count + 8:
            total, industry, repair = 56.0 - index * 0.1, 27.0, 8.0
        else:
            total, industry, repair = 45.0 - index * 0.1, 18.0, 5.0
        rows.append(
            {
                "code": f"{index + 1:06d}",
                "industry": f"行业{index % 8}",
                "total_score": total,
                "industry_prosperity_score": industry,
                "roe_repair_score": repair,
                "liquidity_score": 8.0,
                "risk": "",
            }
        )
    return pd.DataFrame(rows)


class FixedCountSelectionTests(unittest.TestCase):
    def select(self, scored: pd.DataFrame) -> pd.DataFrame:
        self.assertTrue(
            hasattr(core, "select_fixed_count"),
            "the fixed-count selector must be implemented",
        )
        return core.select_fixed_count(scored)

    def test_strict_shortage_is_filled_to_twenty_with_visible_tiers(self) -> None:
        """Catches returning fewer than 20 when strict qualifiers are scarce."""
        selected = self.select(make_scored_universe(40, strict_count=7))

        self.assertEqual(20, len(selected))
        self.assertEqual(7, int((selected["selection_tier"] == "严格入选").sum()))
        self.assertLessEqual(
            set(selected["selection_tier"]),
            {"严格入选", "一级候补", "二级候补"},
        )
        self.assertEqual(list(range(1, 21)), selected["rank"].tolist())

    def test_industry_cap_relaxation_is_disclosed(self) -> None:
        """Catches silently concentrating the fallback portfolio in one industry."""
        scored = make_scored_universe(40, strict_count=30)
        scored["industry"] = ["集中行业" if index < 30 else "其他行业" for index in range(40)]

        selected = self.select(scored)

        self.assertEqual(20, len(selected))
        relaxed = selected[selected["concentration_relaxed"]]
        self.assertFalse(relaxed.empty)
        self.assertTrue(relaxed["risk"].str.contains("行业集中度补位").all())

    def test_hard_eligible_shortage_is_reported_as_data_error(self) -> None:
        """Catches fabricating positions when fewer than 20 tradable rows exist."""
        with self.assertRaisesRegex(ValueError, "硬条件候选不足20只"):
            self.select(make_scored_universe(19, strict_count=19))

    def test_strict_label_requires_high_confidence_thresholds(self) -> None:
        """Catches presenting ordinary fallback-quality rows as strict evidence."""
        scored = make_scored_universe(20, strict_count=20)
        scored.loc[0, "total_score"] = 69.0
        scored.loc[0, "industry_prosperity_score"] = 30.0
        scored.loc[0, "roe_repair_score"] = 16.0

        selected = self.select(scored).set_index("code")

        self.assertEqual("一级候补", selected.loc["000001", "selection_tier"])

    def test_overheated_row_cannot_be_labeled_strict(self) -> None:
        """Catches a severe crowding warning contradicting the strict label."""
        scored = make_scored_universe(20, strict_count=20)
        scored.loc[0, "total_score"] = 80.0
        scored.loc[0, "industry_prosperity_score"] = 32.0
        scored.loc[0, "roe_repair_score"] = 20.0
        scored.loc[0, "risk"] = "动量过热"

        selected = self.select(scored).set_index("code")

        self.assertEqual("一级候补", selected.loc["000001", "selection_tier"])


if __name__ == "__main__":
    unittest.main()
