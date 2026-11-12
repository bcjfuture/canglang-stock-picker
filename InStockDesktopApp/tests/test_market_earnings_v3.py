from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

import instock_tech_picker_data


class MarketEarningsV3FormulaTests(unittest.TestCase):
    def test_formula_uses_historical_roe_instead_of_pb_pe_implied_roe(self) -> None:
        """Catches replacing the independent historical ROE input with PB/PE."""
        metrics = instock_tech_picker_data.calculate_market_earnings_v3(
            pe=8.0,
            pb=0.8,
            historical_roe_pct=20.0,
            dividend_yield=0.05,
        )

        self.assertAlmostEqual(metrics["historical_roe_pct"], 20.0)
        self.assertAlmostEqual(metrics["payout_ratio"], 0.4)
        self.assertAlmostEqual(metrics["dividend_correction"], 1.25)
        self.assertAlmostEqual(metrics["market_earnings_ratio"], 0.25)

    def test_forty_percent_payout_uses_one_point_two_five_correction(self) -> None:
        """Catches omitting or inverting the dividend-payout correction."""
        self.assertTrue(
            hasattr(instock_tech_picker_data, "calculate_market_earnings_v3"),
            "the all-market engine must expose the market-earnings 3.0 formula",
        )

        metrics = instock_tech_picker_data.calculate_market_earnings_v3(
            pe=8.0,
            pb=0.8,
            historical_roe_pct=10.0,
            dividend_yield=0.05,
        )

        self.assertAlmostEqual(metrics["historical_roe_pct"], 10.0)
        self.assertAlmostEqual(metrics["implied_roe_pct"], 10.0)
        self.assertAlmostEqual(metrics["payout_ratio"], 0.4)
        self.assertAlmostEqual(metrics["dividend_correction"], 1.25)
        self.assertAlmostEqual(metrics["market_earnings_ratio"], 1.0)


class MarketEarningsV3SelectionTests(unittest.TestCase):
    def test_formula_candidate_does_not_require_technical_history_cache(self) -> None:
        """Catches an unrelated daily-bar cache excluding stocks before PR is calculated."""
        config = instock_tech_picker_data.load_config(
            ROOT / "engine" / "all_market_config.json"
        )
        source = {
            "code": "000001", "name": "公式候选", "industry": "银行",
            "pe9": 8.0, "pbnewmrq": 0.8, "roe_weight": 10.0,
            "zxgxl": 5.0,
        }

        scored = instock_tech_picker_data.score_row(
            source, config, set(), set(), set(), {}
        )

        self.assertIsNotNone(scored)

    def test_market_earnings_ratio_must_be_strictly_below_one(self) -> None:
        """Catches PR=1 or PR>1 being used to fill the twenty slots."""
        rows = [
            {
                "代码": "000001", "名称": "低于一", "行业": "银行",
                "PE": 8.0, "PB": 0.799, "历史最低ROE%": 10.0,
                "股息率%": 5.0, "综合评分": 50,
            },
            {
                "代码": "000002", "名称": "刚好一", "行业": "银行",
                "PE": 8.0, "PB": 0.8, "历史最低ROE%": 10.0,
                "股息率%": 5.0, "综合评分": 50,
            },
            {
                "代码": "000003", "名称": "大于一", "行业": "银行",
                "PE": 8.0, "PB": 0.9, "历史最低ROE%": 10.0,
                "股息率%": 5.0, "综合评分": 50,
            },
        ]

        selected = instock_tech_picker_data.select_market_earnings_v3(
            rows,
            {
                "top_n": 20,
                "market_earnings_v3": {"max_market_earnings_ratio": 1.0},
            },
        )

        self.assertEqual([row["代码"] for row in selected], ["000001"])

    def test_negative_growth_and_cash_flow_are_quality_warnings_not_hidden_gates(self) -> None:
        """Catches extra quality filters shrinking a valid PR<1 pool below twenty."""
        rows = [
            {
                "代码": "000001", "名称": "公式合格", "行业": "机械",
                "PE": 8.0, "PB": 0.5, "历史最低ROE%": 12.0,
                "股息率%": 6.25, "净利同比%": -3.0,
                "三年净利增速%": -1.0, "资产负债率%": 75.0,
                "每股经营现金流": -0.2, "综合评分": 50,
                "风险提示": "",
            }
        ]

        selected = instock_tech_picker_data.select_market_earnings_v3(
            rows,
            {
                "top_n": 20,
                "market_earnings_v3": {"max_market_earnings_ratio": 1.0},
            },
        )

        self.assertEqual([row["代码"] for row in selected], ["000001"])
        self.assertIn("净利润同比下降", selected[0]["风险提示"])
        self.assertIn("经营现金流为负", selected[0]["风险提示"])

    def test_quality_ranking_removes_the_weakest_stock_when_formula_pool_exceeds_twenty(self) -> None:
        """Catches a very low PR overriding obvious multi-factor fundamental weakness."""
        healthy = [
            {
                "代码": f"1{index:05d}", "名称": f"稳健{index}", "行业": "银行",
                "PE": 8.0, "PB": 0.8, "历史最低ROE%": 15.0,
                "股息率%": 7.0, "净利同比%": 5.0,
                "三年净利增速%": 3.0, "综合评分": 50,
            }
            for index in range(1, 21)
        ]
        weak = {
            "代码": "000001", "名称": "六项风险", "行业": "建筑施工",
            "PE": 4.0, "PB": 0.2, "历史最低ROE%": 20.0,
            "股息率%": 0.0, "净利同比%": -10.0,
            "三年净利增速%": -5.0, "资产负债率%": 80.0,
            "每股经营现金流": -1.0, "综合评分": 99,
        }

        selected = instock_tech_picker_data.select_market_earnings_v3(
            [weak, *healthy],
            {
                "top_n": 20,
                "market_earnings_v3": {
                    "max_market_earnings_ratio": 1.0,
                    "financial_industries": ["银行", "非银行金融"],
                },
            },
        )

        self.assertEqual(len(selected), 20)
        self.assertNotIn("000001", [row["代码"] for row in selected])


class HistoricalRoeTests(unittest.TestCase):
    def test_recent_three_disclosed_annual_reports_supply_the_minimum_roe(self) -> None:
        """Catches interim reports, future disclosures, or older years entering the minimum."""
        records = [
            {"REPORT_DATE": "2026-06-30", "NOTICE_DATE": "2026-08-20", "ROEJQ": 4.0},
            {"REPORT_DATE": "2025-12-31", "NOTICE_DATE": "2026-03-20", "ROEJQ": 14.0},
            {"REPORT_DATE": "2024-12-31", "NOTICE_DATE": "2025-03-20", "ROEJQ": 12.0},
            {"REPORT_DATE": "2023-12-31", "NOTICE_DATE": "2024-03-20", "ROEJQ": 13.0},
            {"REPORT_DATE": "2022-12-31", "NOTICE_DATE": "2023-03-20", "ROEJQ": 3.0},
            {"REPORT_DATE": "2026-12-31", "NOTICE_DATE": "2027-03-20", "ROEJQ": 2.0},
        ]

        result = instock_tech_picker_data.minimum_disclosed_annual_roe(
            records,
            as_of_date=date(2026, 9, 7),
            years=3,
        )

        self.assertEqual(result, {"minimum_roe_pct": 12.0, "annual_count": 3})

    def test_history_loader_enriches_only_pb_discount_candidates_and_reuses_cache(self) -> None:
        """Catches full-market over-fetching or ignoring the persisted annual ROE history."""
        rows = [
            {"代码": "000001", "PE": 8.0, "PB": 0.8},
            {"代码": "000002", "PE": 9.0, "PB": 0.9},
            {"代码": "000003", "PE": 8.0, "PB": 1.1},
        ]
        requested = []

        def fetcher(code):
            requested.append(code)
            return [
                {"REPORT_DATE": "2025-12-31", "NOTICE_DATE": "2026-03-20", "ROEJQ": 14.0},
                {"REPORT_DATE": "2024-12-31", "NOTICE_DATE": "2025-03-20", "ROEJQ": 12.0},
                {"REPORT_DATE": "2023-12-31", "NOTICE_DATE": "2024-03-20", "ROEJQ": 13.0},
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "roe-history.json"
            first = instock_tech_picker_data.load_market_earnings_roe_minimums(
                rows,
                as_of_date=date(2026, 9, 7),
                years=3,
                fetcher=fetcher,
                cache_path=cache_path,
                max_workers=1,
            )
            second = instock_tech_picker_data.load_market_earnings_roe_minimums(
                rows,
                as_of_date=date(2026, 9, 7),
                years=3,
                fetcher=lambda code: self.fail(f"unexpected refetch for {code}"),
                cache_path=cache_path,
                max_workers=1,
            )

        self.assertEqual(requested, ["000001", "000002"])
        self.assertEqual(first["000001"]["minimum_roe_pct"], 12.0)
        self.assertEqual(first, second)
        self.assertNotIn("000003", first)

    def test_history_minimums_are_attached_to_scored_candidates(self) -> None:
        """Catches the fetched ROE history being lost before formula selection."""
        rows = [{"代码": "000001", "PE": 8.0, "PB": 0.8}]

        enriched = instock_tech_picker_data.attach_market_earnings_roe_minimums(
            rows,
            {
                "000001": {"minimum_roe_pct": 12.0, "annual_count": 3},
            },
        )

        self.assertEqual(enriched[0]["历史最低ROE%"], 12.0)
        self.assertEqual(enriched[0]["历史ROE年度数"], 3)
    def test_candidates_require_positive_pe_and_pb_strictly_below_one(self) -> None:
        """Catches the all-market pool silently admitting PB>=1 or loss-makers."""
        self.assertTrue(
            hasattr(instock_tech_picker_data, "select_market_earnings_v3"),
            "the all-market engine must expose the 3.0 selector",
        )
        rows = [
            {"代码": "000001", "名称": "合格", "PE": 8.0, "PB": 0.8, "历史最低ROE%": 15.0},
            {"代码": "000002", "名称": "净资产不打折", "PE": 8.0, "PB": 1.0, "历史最低ROE%": 15.0},
            {"代码": "000003", "名称": "亏损", "PE": -8.0, "PB": 0.7, "历史最低ROE%": 15.0},
        ]

        selected = instock_tech_picker_data.select_market_earnings_v3(
            rows,
            {"top_n": 20, "market_earnings_v3": {}},
        )

        self.assertEqual([row["代码"] for row in selected], ["000001"])

    def test_hard_news_risk_cannot_enter_the_market_earnings_pool(self) -> None:
        """Catches a superficially cheap stock with a major disclosure risk ranking first."""
        rows = [
            {
                "代码": "000001", "名称": "正常", "行业": "银行",
                "PE": 8.0, "PB": 0.8, "综合评分": 50,
                "历史最低ROE%": 15.0,
                "新闻重大风险": False,
            },
            {
                "代码": "000002", "名称": "重大风险", "行业": "银行",
                "PE": 4.0, "PB": 0.5, "综合评分": 99,
                "历史最低ROE%": 15.0,
                "新闻重大风险": True,
            },
        ]

        selected = instock_tech_picker_data.select_market_earnings_v3(
            rows, {"top_n": 20, "market_earnings_v3": {}}
        )

        self.assertEqual([row["代码"] for row in selected], ["000001"])

    def test_candidates_require_three_year_historical_roe_evidence(self) -> None:
        """Catches substituting current growth or implied ROE for annual history."""
        rows = [
            {
                "代码": "000001", "名称": "合格", "PE": 8.0, "PB": 0.8,
                "历史最低ROE%": 15.0,
                "净利同比%": 5.0, "三年净利增速%": 3.0,
            },
            {
                "代码": "000002", "名称": "缺少历史ROE", "PE": 8.0, "PB": 0.8,
                "净利同比%": 5.0, "三年净利增速%": 3.0,
            },
            {
                "代码": "000003", "名称": "当期下滑", "PE": 8.0, "PB": 0.8,
                "历史最低ROE%": 15.0,
                "净利同比%": -0.1, "三年净利增速%": 3.0,
            },
            {
                "代码": "000004", "名称": "长期下滑", "PE": 8.0, "PB": 0.8,
                "历史最低ROE%": 15.0,
                "净利同比%": 5.0, "三年净利增速%": -0.1,
            },
        ]
        config = {
            "top_n": 20,
            "market_earnings_v3": {"max_market_earnings_ratio": 1.0},
        }

        selected = instock_tech_picker_data.select_market_earnings_v3(rows, config)

        self.assertEqual(
            [row["代码"] for row in selected],
            ["000001", "000003", "000004"],
        )

    def test_nonfinancial_quality_uses_cash_flow_and_debt_with_financial_exception(self) -> None:
        """Catches applying industrial-company leverage rules to banks or skipping them."""
        common = {
            "PE": 8.0, "PB": 0.8, "历史最低ROE%": 15.0,
            "净利同比%": 5.0, "三年净利增速%": 3.0,
        }
        rows = [
            {
                **common, "代码": "000001", "名称": "工业合格", "行业": "机械",
                "资产负债率%": 60.0, "每股经营现金流": 0.5,
            },
            {
                **common, "代码": "000002", "名称": "工业高负债", "行业": "机械",
                "资产负债率%": 75.0, "每股经营现金流": 0.5,
            },
            {
                **common, "代码": "000003", "名称": "工业现金差", "行业": "机械",
                "资产负债率%": 60.0, "每股经营现金流": -0.1,
            },
            {
                **common, "代码": "000004", "名称": "银行合格", "行业": "银行",
                "资产负债率%": 93.0, "每股经营现金流": -2.0,
            },
        ]
        config = {
            "top_n": 20,
            "market_earnings_v3": {
                "max_nonfinancial_debt_asset_ratio_pct": 70.0,
                "require_positive_nonfinancial_operating_cash_flow": True,
                "financial_industries": ["银行", "非银行金融"],
            },
        }

        selected = instock_tech_picker_data.select_market_earnings_v3(rows, config)

        self.assertEqual([row["代码"] for row in selected], ["000001", "000004", "000002", "000003"])
        self.assertEqual([row["基本面风险分"] for row in selected], [0, 0, 1, 1])
        self.assertIn("资产负债率偏高", selected[2]["风险提示"])
        self.assertIn("经营现金流为负", selected[3]["风险提示"])

    def test_all_market_configuration_returns_only_the_best_twenty(self) -> None:
        """Catches the default all-market strategy reverting to its old broad ranking."""
        config = instock_tech_picker_data.load_config(
            ROOT / "engine" / "all_market_config.json"
        )
        self.assertEqual(config.get("selection_mode"), "market_earnings_v3")
        rows = [
            {
                "代码": f"{index:06d}", "名称": f"候选{index}", "行业": f"行业{index}",
                "PE": 5.0 + index / 10.0, "PB": 0.8, "历史最低ROE%": 20.0,
                "股息率%": 10.0,
                "净利同比%": 5.0, "三年净利增速%": 3.0,
                "综合评分": 50.0,
            }
            for index in range(1, 26)
        ]
        rows.append(
            {
                "代码": "999999", "名称": "PB越界", "行业": "银行",
                "PE": 4.0, "PB": 1.01, "历史最低ROE%": 20.0,
                "股息率%": 20.0, "净利同比%": 10.0,
                "三年净利增速%": 10.0, "综合评分": 99.0,
            }
        )

        selected = instock_tech_picker_data.select_market_earnings_v3(rows, config)

        self.assertEqual(len(selected), 20)
        self.assertEqual(selected[0]["代码"], "000001")
        self.assertEqual(selected[-1]["代码"], "000020")
        self.assertTrue(all(0 < row["PB"] < 1 for row in selected))

    def test_pr_at_or_above_one_is_never_used_to_fill_twenty_slots(self) -> None:
        """Catches threshold-breaking fallback when fewer than twenty pass PR<1."""
        config = instock_tech_picker_data.load_config(
            ROOT / "engine" / "all_market_config.json"
        )
        eligible_rows = [
            {
                "代码": f"1{index:05d}", "名称": f"合格{index}", "行业": f"行业{index}",
                "PE": 8.0, "PB": 0.7, "历史最低ROE%": 15.0,
                "股息率%": 7.0,
                "净利同比%": 5.0, "三年净利增速%": 3.0,
                "综合评分": 50.0,
            }
            for index in range(1, 16)
        ]
        ineligible_rows = [
            {
                "代码": f"2{index:05d}", "名称": f"超限{index}", "行业": "银行",
                "PE": 8.0, "PB": 0.8, "历史最低ROE%": 10.0,
                "股息率%": 5.0,
                "净利同比%": -2.0, "三年净利增速%": -1.0,
                "综合评分": 40.0,
            }
            for index in range(1, 11)
        ]
        rows = eligible_rows + ineligible_rows + [
            {
                "代码": "999999", "名称": "禁止补足", "行业": "银行",
                "PE": 4.0, "PB": 1.01, "历史最低ROE%": 20.0,
                "股息率%": 20.0, "净利同比%": 20.0,
                "三年净利增速%": 20.0, "综合评分": 100.0,
            }
        ]

        selected = instock_tech_picker_data.select_market_earnings_v3(rows, config)

        self.assertEqual(len(selected), 15)
        self.assertTrue(all(row["PB"] < 1 for row in selected))

    def test_single_industry_is_capped(self) -> None:
        """Catches one cheap sector (e.g. banks) filling the whole list."""
        config = instock_tech_picker_data.load_config(ROOT / "engine" / "all_market_config.json")
        rows = [
            {
                "代码": f"{index:06d}", "名称": f"银行{index}", "行业": "银行" if index <= 10 else f"行业{index}",
                "PE": 5.0 + index / 10.0, "PB": 0.8, "历史最低ROE%": 20.0, "股息率%": 10.0,
                "净利同比%": 5.0, "三年净利增速%": 3.0, "综合评分": 50.0,
            }
            for index in range(1, 31)
        ]

        selected = instock_tech_picker_data.select_market_earnings_v3(rows, config)

        self.assertEqual(20, len(selected))
        self.assertEqual(4, sum(row["行业"] == "银行" for row in selected))

    def test_low_risk_tier_then_ratio_decides_order(self) -> None:
        """Catches a 0-6 risk count overriding the valuation metric inside a tier."""
        config = instock_tech_picker_data.load_config(ROOT / "engine" / "all_market_config.json")
        base = {"PB": 0.8, "历史最低ROE%": 20.0, "股息率%": 10.0, "综合评分": 50.0, "资产负债率%": 40.0, "经营现金流为正": True}
        cheap_one_flag = dict(base, **{"代码": "000001", "行业": "甲", "PE": 5.0, "PB": 0.7, "净利同比%": None, "三年净利增速%": 3.0})
        dearer_clean = dict(base, **{"代码": "000002", "行业": "乙", "PE": 8.0, "净利同比%": 5.0, "三年净利增速%": 3.0})
        risky_cheapest = dict(base, **{"代码": "000003", "行业": "丙", "PE": 4.0, "PB": 0.5, "净利同比%": -5.0, "三年净利增速%": -5.0})

        selected = instock_tech_picker_data.select_market_earnings_v3(
            [risky_cheapest, dearer_clean, cheap_one_flag], config
        )

        self.assertEqual(["000001", "000002", "000003"], [row["代码"] for row in selected])
        self.assertTrue(all(row["入选层级"] == "市赚率<1" for row in selected))
        self.assertTrue(all(row["市赚率3.0"] < 1 for row in selected))

    def test_scored_database_row_retains_every_quality_and_dividend_input(self) -> None:
        """Catches source fundamentals being dropped before the 3.0 selector runs."""
        config = instock_tech_picker_data.load_config(
            ROOT / "engine" / "all_market_config.json"
        )
        source = {
            "code": "000001", "name": "测试银行", "industry": "银行",
            "pe9": 8.0, "pbnewmrq": 0.8, "roe_weight": 10.0,
            "netprofit_yoy_ratio": 5.0, "netprofit_growthrate_3y": 3.0,
            "debt_asset_ratio": 93.0, "per_netcash_operate": -2.0,
            "zxgxl": 5.0,
            "historical_factor_score": 0.0,
        }

        scored = instock_tech_picker_data.score_row(
            source, config, set(), set(), set(), {}
        )
        scored["历史最低ROE%"] = 20.0
        selected = instock_tech_picker_data.select_market_earnings_v3(
            [scored], config
        )

        self.assertEqual([row["代码"] for row in selected], ["000001"])
        self.assertEqual(selected[0]["三年净利增速%"], 3.0)
        self.assertEqual(selected[0]["资产负债率%"], 93.0)
        self.assertEqual(selected[0]["每股经营现金流"], -2.0)
        self.assertEqual(selected[0]["股息率%"], 5.0)
        self.assertEqual(selected[0]["分红修正系数"], 1.25)

    def test_zero_profit_growth_is_preserved_as_valid_boundary_value(self) -> None:
        """Catches zero growth being converted to missing and wrongly rejected."""
        config = instock_tech_picker_data.load_config(
            ROOT / "engine" / "all_market_config.json"
        )
        source = {
            "code": "000001", "name": "零增长银行", "industry": "银行",
            "pe9": 8.0, "pbnewmrq": 0.8, "roe_weight": 10.0,
            "netprofit_yoy_ratio": 0.0, "netprofit_growthrate_3y": 0.0,
            "debt_asset_ratio": 93.0, "per_netcash_operate": -2.0,
            "zxgxl": 5.0,
            "historical_factor_score": 0.0,
        }

        scored = instock_tech_picker_data.score_row(
            source, config, set(), set(), set(), {}
        )
        scored["历史最低ROE%"] = 20.0
        selected = instock_tech_picker_data.select_market_earnings_v3(
            [scored], config
        )

        self.assertEqual(scored["净利同比%"], 0.0)
        self.assertEqual([row["代码"] for row in selected], ["000001"])

    def test_selection_snapshot_uses_the_full_market_latest_dividend_yield(self) -> None:
        """Catches reverting to sparse single-report dividend-plan data."""
        class RecordingCursor:
            def __init__(self) -> None:
                self.calls = []

            def execute(self, sql, params=None) -> None:
                self.calls.append((sql, params))

            def fetchall(self):
                return []

        cursor = RecordingCursor()
        trade_date = date(2026, 9, 7)
        with (
            patch(
                "instock_tech_picker_data.latest_date",
                side_effect=[trade_date, trade_date, trade_date],
            ),
            patch("instock_tech_picker_data.fetch_code_set", return_value=set()),
            patch("instock_tech_picker_data.fetch_strategy_hits", return_value={}),
        ):
            instock_tech_picker_data.load_selection_snapshot(cursor, trade_date)

        sql = cursor.calls[-1][0]
        self.assertIn("s.netprofit_growthrate_3y", sql)
        self.assertIn("s.per_netcash_operate", sql)
        self.assertIn("s.zxgxl", sql)
        self.assertNotIn("LEFT JOIN cn_stock_bonus b", sql)

    def test_all_market_output_path_invokes_market_earnings_selector(self) -> None:
        """Catches main output still slicing the legacy comprehensive-score list."""
        self.assertTrue(
            hasattr(instock_tech_picker_data, "select_output_rows"),
            "the main output path must have a strategy-aware selection boundary",
        )
        config = instock_tech_picker_data.load_config(
            ROOT / "engine" / "all_market_config.json"
        )
        rows = [
            {
                "代码": "000001", "名称": "合格", "行业": "银行",
                "PE": 8.0, "PB": 0.8, "历史最低ROE%": 20.0,
                "股息率%": 5.0, "净利同比%": 5.0,
                "三年净利增速%": 3.0, "综合评分": 10.0,
            },
            {
                "代码": "000002", "名称": "旧榜高分但PB越界", "行业": "银行",
                "PE": 8.0, "PB": 1.2, "历史最低ROE%": 20.0,
                "股息率%": 5.0, "净利同比%": 5.0,
                "三年净利增速%": 3.0, "综合评分": 99.0,
            },
        ]

        selected = instock_tech_picker_data.select_output_rows(
            rows, config, active_count=20, regime="legacy"
        )

        self.assertEqual([row["代码"] for row in selected], ["000001"])
        self.assertIn("市赚率3.0", selected[0])

    def test_missing_dividend_data_uses_conservative_cap_and_discloses_it(self) -> None:
        """Catches missing dividends being treated as a generous payout or hidden."""
        selected = instock_tech_picker_data.select_market_earnings_v3(
            [
                {
                    "代码": "000001", "名称": "分红待核", "行业": "银行",
                    "PE": 8.0, "PB": 0.8, "历史最低ROE%": 15.0,
                    "综合评分": 50,
                    "加分理由": "基本面合格", "风险提示": "",
                }
            ],
            {"top_n": 20, "market_earnings_v3": {}},
        )

        self.assertEqual(selected[0]["分红修正系数"], 2.0)
        self.assertEqual(selected[0]["市赚率3.0"], 0.7111)
        self.assertIn("市赚率3.0", selected[0]["加分理由"])
        self.assertIn("分红数据缺失", selected[0]["风险提示"])

    def test_all_market_fundamental_sheet_uses_the_same_formula_pool(self) -> None:
        """Catches the exported fundamentals sheet showing PB>=1 legacy candidates."""
        self.assertTrue(
            hasattr(instock_tech_picker_data, "select_fundamental_rows"),
            "the payload must expose a strategy-aware fundamentals view",
        )
        config = instock_tech_picker_data.load_config(
            ROOT / "engine" / "all_market_config.json"
        )
        rows = [
            {
                "代码": "000001", "名称": "公式合格", "行业": "银行",
                "PE": 8.0, "PB": 0.8, "历史最低ROE%": 20.0,
                "股息率%": 5.0, "净利同比%": 5.0,
                "三年净利增速%": 3.0, "综合评分": 10.0,
            },
            {
                "代码": "000002", "名称": "旧基本面高分", "行业": "银行",
                "PE": 10.0, "PB": 1.5, "历史最低ROE%": 20.0, "ROE%": 20.0,
                "股息率%": 5.0,
                "净利同比%": 20.0, "三年净利增速%": 20.0,
                "综合评分": 99.0,
            },
        ]

        fundamentals = instock_tech_picker_data.select_fundamental_rows(rows, config)

        self.assertEqual([row["代码"] for row in fundamentals], ["000001"])
        self.assertIn("市赚率3.0", fundamentals[0])


if __name__ == "__main__":
    unittest.main()
