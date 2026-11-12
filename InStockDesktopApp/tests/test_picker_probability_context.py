from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from instock_app.settings import STRATEGIES
from instock_app.ui.picker_page import format_market_regime
from workers.picker_worker import (
    HEAD_RERANK_POLICIES,
    apply_head_reranking,
    apply_probability_context,
    normalize_picker_payload,
    probability_context_from_payload,
)


class PickerProbabilityContextTests(unittest.TestCase):
    def test_market_earnings_all_market_policy_preserves_formula_order(self):
        """Catches probability reranking replacing PB/quality-qualified PR3 stocks."""
        spec = STRATEGIES["all_market"]
        rows = [
            {
                "代码": f"{index:06d}", "名称": f"市赚率候选{index}",
                "行业": "银行", "综合评分": 100 - index,
                "PB": 0.8, "市赚率3.0": 0.5 + index / 100,
            }
            for index in range(1, 21)
        ]
        payload = {
            "sheets": {spec.top_sheet: rows, "全部候选样本": rows},
            "summary": {},
        }
        probability_context = {
            "selected_rows": [
                {
                    "代码": "600360", "名称": "公式池外核心",
                    "行业": "计算机", "综合评分": 999,
                    "通过高胜率门槛": True,
                }
            ]
        }

        apply_head_reranking(
            spec,
            payload,
            probability_context,
            {},
            all_sheet_name="全部候选样本",
            policy=HEAD_RERANK_POLICIES["all_market"],
        )

        self.assertEqual(
            [f"{index:06d}" for index in range(1, 21)],
            [row["代码"] for row in payload["sheets"][spec.top_sheet]],
        )
        self.assertFalse(payload["summary"]["head_rerank_enabled"])

    def test_disabled_reranking_preserves_top_sheet_not_broad_source_sheet(self):
        """Catches a disabled reranker replacing PR3 picks from the broad score pool."""
        spec = STRATEGIES["all_market"]
        formula_top = [
            {"代码": f"{index:06d}", "综合评分": 50, "排名": index}
            for index in range(1, 21)
        ]
        broad_source = [
            {"代码": f"9{index:05d}", "综合评分": 100, "排名": index}
            for index in range(1, 21)
        ]
        payload = {
            "sheets": {
                spec.top_sheet: formula_top,
                "全部候选样本": broad_source,
            }
        }

        apply_head_reranking(
            spec,
            payload,
            {},
            {},
            all_sheet_name="全部候选样本",
            policy=HEAD_RERANK_POLICIES["all_market"],
        )

        self.assertEqual(
            [f"{index:06d}" for index in range(1, 21)],
            [row["代码"] for row in payload["sheets"][spec.top_sheet]],
        )

    def test_probability_context_is_indexed_by_stock_code(self):
        payload = {
            "latest_dates": {"综合选股": "2026-07-17"},
            "summary": {
                "market_regime": "poor",
                "probability_threshold": 0.61,
                "unavailable_probability_codes": ["603980"],
            },
            "sheets": {
                "高胜率精选": [],
                "全部概率候选": [
                    {"代码": "601016", "5日上涨概率": 0.6432},
                    {"代码": "000703", "5日上涨概率": 0.5871},
                ],
            },
        }

        context = probability_context_from_payload(payload, {})

        self.assertEqual(
            {"601016": 0.6432, "000703": 0.5871},
            context["probabilities_by_code"],
        )
        self.assertEqual(["603980"], context["unavailable_codes"])
        self.assertGreater(
            context["precision_scores_by_code"]["601016"],
            context["precision_scores_by_code"]["000703"],
        )
        self.assertEqual([], context["selected_rows"])

    def test_base_strategy_uses_shortlist_for_top_five_reranking(self):
        spec = STRATEGIES["all_market"]
        shortlist = [
            {
                "代码": f"{index:06d}",
                "名称": f"股票{index}",
                "行业": f"行业{index % 5}",
                "综合评分": 200 - index,
            }
            for index in range(1, 31)
        ]
        payload = {
            "sheets": {
                spec.top_sheet: shortlist[:20],
                "全部候选样本": shortlist,
            }
        }
        context = {
            "probabilities_by_code": {row["代码"]: 0.5 for row in shortlist},
            "precision_scores_by_code": {
                **{row["代码"]: 0.1 for row in shortlist},
                "000006": 1.0,
            },
        }
        apply_head_reranking(
            spec,
            payload,
            context,
            {},
            all_sheet_name="全部候选样本",
        )
        top = payload["sheets"][spec.top_sheet]
        self.assertEqual(len(top), 20)
        self.assertIn("000006", [row["代码"] for row in top[:5]])

    def test_strategy_without_oos_improvement_keeps_original_top_five(self):
        spec = STRATEGIES["tech_attack"]
        rows = [
            {
                "代码": f"{index:06d}",
                "名称": f"股票{index}",
                "行业": f"行业{index % 5}",
                "综合评分": 200 - index,
            }
            for index in range(1, 21)
        ]
        payload = {"sheets": {spec.top_sheet: rows}}
        apply_head_reranking(
            spec,
            payload,
            {"precision_scores_by_code": {"000006": 1.0}},
            {},
            all_sheet_name=spec.top_sheet,
            policy={"enabled": False, "reason": "oos_gate_failed"},
        )
        top = payload["sheets"][spec.top_sheet]
        self.assertEqual([f"{index:06d}" for index in range(1, 6)], [row["代码"] for row in top[:5]])
        self.assertFalse(payload["summary"]["head_rerank_enabled"])

    def test_all_market_can_pin_a_validated_core_stock_outside_base_shortlist(self):
        spec = STRATEGIES["all_market"]
        rows = [
            {
                "代码": f"{index:06d}",
                "名称": f"股票{index}",
                "行业": f"行业{index % 5}",
                "综合评分": 200 - index,
            }
            for index in range(1, 21)
        ]
        core = {
            "代码": "600360",
            "名称": "高胜率核心",
            "行业": "计算机软件",
            "综合评分": 88,
            "通过高胜率门槛": True,
        }
        payload = {"sheets": {spec.top_sheet: rows}}
        apply_head_reranking(
            spec,
            payload,
            {"selected_rows": [core]},
            {},
            all_sheet_name=spec.top_sheet,
            policy={
                "enabled": True,
                "use_global_core": True,
                "candidate_limit": 10,
                "precision_weight": 0.0,
                "probability_weight": 0.0,
                "sector_weight_scale": 1.0,
                "industry_cap": 2,
            },
        )
        top = payload["sheets"][spec.top_sheet]
        self.assertEqual("600360", top[0]["代码"])
        self.assertEqual(0, top[0]["原策略排名"])
        self.assertEqual(20, len(top))

    def test_live_normal_strategy_rows_receive_per_code_probabilities(self):
        spec = STRATEGIES["rebound_trend"]
        payload = {
            "generated_at": "2026-07-18 13:39:22",
            "latest_dates": {"综合选股": "2026-07-17"},
            "strategy_gate_passed": False,
            "summary": {},
            "sheets": {
                spec.top_sheet: [
                    {
                        "代码": "601016",
                        "名称": "节能风电",
                        "行业": "电力",
                        "最新价": 3.5,
                        "涨跌幅%": 6.06,
                        "综合评分": 144.1,
                        "今日主力净流入_万元": 10940.22,
                        "头部重排": True,
                        "头部重排评分": 88.5,
                        "原策略排名": 6,
                        "热门板块": "煤化工概念",
                        "板块因子权重": 0.15,
                    },
                    {
                        "代码": "603980",
                        "名称": "吉华集团",
                        "行业": "化学制品",
                        "最新价": 6.17,
                        "涨跌幅%": 2.49,
                        "综合评分": 141.3,
                        "今日主力净流入_万元": 3215.86,
                    },
                ]
            },
        }
        context = {
            "trade_date": "2026-07-17",
            "market_regime": "poor",
            "probability_threshold": 0.61,
            "probabilities_by_code": {"601016": 0.6432},
            "gate_passed_by_code": {"601016": True},
            "unavailable_codes": ["603980"],
        }

        apply_probability_context(spec, payload, context)
        result = normalize_picker_payload(spec, payload)
        first, second = result["rows"]

        self.assertEqual(0.6432, first["win_probability"])
        self.assertEqual("poor", first["market_regime"])
        self.assertEqual(0.61, first["probability_threshold"])
        self.assertTrue(first["high_precision_gate_passed"])
        self.assertTrue(first["head_reranked"])
        self.assertEqual(88.5, first["head_rerank_score"])
        self.assertEqual(6, first["original_rank"])
        self.assertEqual("煤化工概念", first["hot_sector"])
        self.assertEqual(0.15, first["sector_factor_weight"])
        self.assertIsNone(second["win_probability"])

    def test_probability_context_rejects_a_different_trade_date(self):
        spec = STRATEGIES["rebound_trend"]
        payload = {
            "latest_dates": {"综合选股": "2026-07-17"},
            "sheets": {spec.top_sheet: []},
        }
        context = {
            "trade_date": "2026-07-16",
            "market_regime": "poor",
            "probability_threshold": 0.57,
            "probabilities_by_code": {},
            "unavailable_codes": [],
        }

        with self.assertRaises(ValueError):
            apply_probability_context(spec, payload, context)

    def test_v5_market_regimes_are_displayed_in_chinese(self):
        self.assertEqual("良好", format_market_regime("good"))
        self.assertEqual("一般", format_market_regime("normal"))
        self.assertEqual("糟糕", format_market_regime("poor"))


if __name__ == "__main__":
    unittest.main()
