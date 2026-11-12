from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from workers.top5_reranker import rerank_top_five, sector_context_from_payload


def make_rows(count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "代码": f"{index:06d}",
            "名称": f"股票{index}",
            "行业": f"行业{index % 5}",
            "概念": "普通概念",
            "综合评分": 200 - index,
            "风险提示": "",
            "新闻重大风险": False,
        }
        for index in range(1, count + 1)
    ]


class TopFiveRerankerTests(unittest.TestCase):
    def test_high_precision_signal_can_promote_a_shortlist_stock_into_top_five(self) -> None:
        rows = make_rows(30)
        context = {
            "probabilities_by_code": {row["代码"]: 0.50 for row in rows},
            "precision_scores_by_code": {
                **{row["代码"]: 0.1 for row in rows},
                "000006": 1.0,
            },
        }
        result = rerank_top_five(rows, context, {})
        self.assertEqual(len(result), 20)
        self.assertIn("000006", [row["代码"] for row in result[:5]])
        self.assertEqual([1, 2, 3, 4, 5], [row["排名"] for row in result[:5]])

    def test_top_five_never_contains_more_than_two_stocks_from_one_industry(self) -> None:
        rows = make_rows(20)
        for row in rows[:8]:
            row["行业"] = "同一行业"
        context = {
            "probabilities_by_code": {row["代码"]: 0.5 for row in rows},
            "precision_scores_by_code": {row["代码"]: 1.0 for row in rows},
        }
        result = rerank_top_five(rows, context, {})
        industries = [row["行业"] for row in result[:5]]
        self.assertLessEqual(industries.count("同一行业"), 2)

    def test_hard_risk_stock_is_excluded_from_top_five_but_not_deleted(self) -> None:
        rows = make_rows(20)
        rows[0]["新闻重大风险"] = True
        result = rerank_top_five(rows, {}, {})
        self.assertNotIn("000001", [row["代码"] for row in result[:5]])
        self.assertIn("000001", [row["代码"] for row in result])

    def test_head_ready_sector_uses_fifteen_percent_only_for_first_five(self) -> None:
        payload = {
            "as_of": "2026-08-21",
            "universes": {
                "concept": {
                    "status": "head_ready",
                    "rows": [
                        {"rank": rank, "sector_name": f"概念{rank}", "score": 100 - rank}
                        for rank in range(1, 11)
                    ],
                }
            },
        }
        context = sector_context_from_payload(payload, "2026-08-21")
        self.assertEqual(context["concepts"]["概念1"]["weight"], 0.15)
        self.assertNotIn("概念6", context["concepts"])

    def test_forced_fallback_sector_is_only_a_five_percent_tiebreaker(self) -> None:
        payload = {
            "as_of": "2026-08-21",
            "universes": {
                "industry": {
                    "status": "forced_fallback",
                    "rows": [{"rank": 1, "sector_name": "通信设备", "score": 98}],
                }
            },
        }
        context = sector_context_from_payload(payload, "2026-08-21")
        self.assertEqual(context["industries"]["通信设备"]["weight"], 0.05)


if __name__ == "__main__":
    unittest.main()
