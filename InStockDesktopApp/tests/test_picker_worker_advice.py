from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from instock_app.settings import STRATEGIES
from workers.picker_worker import normalize_picker_payload


class PickerWorkerAdviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = STRATEGIES["tech_attack"]

    def payload(
        self,
        *,
        gate: bool,
        policy: dict | None = None,
        atr: object = 0.5,
    ) -> dict:
        return {
            "order_flow_schema": "grouped-order-flow-v2",
            "generated_at": "2026-07-14 16:00:00",
            "latest_dates": {"每日行情": "2026-07-14"},
            "strategy_gate_passed": gate,
            "trade_policy": policy,
            "summary": {},
            "sheets": {
                self.spec.top_sheet: [
                    {
                        "代码": "000001",
                        "综合评分": 88.0,
                        "名称": "测试股票",
                        "行业": "测试行业",
                        "最新价": 10.0,
                        "涨跌幅": 1.2,
                        "今日主力净流入_万元": 100.0,
                        "3日主力净流入_万元": 240.0,
                        "5日主力净流入_万元": 360.0,
                        "股东结构评分": 11.5,
                        "机构持股比例%": 28.0,
                        "机构持股家数": 35,
                        "股东户数变化%": -6.0,
                        "十大股东持股比例%": 55.0,
                        "近3月机构调研数": 12,
                        "股东结构说明": "机构持股28.00%（35家）；股东户数变化-6.00%",
                        "今日大单净额_万元": 1234.5,
                        "今日散户净额_万元": -456.7,
                        "订单资金日期": "2026-07-14",
                        "订单资金来源": "数据库分项",
                        "PE": 20.0,
                        "ROE%": 12.0,
                        "净利同比": 15.0,
                        "加分理由": "测试理由",
                        "风险提示": "测试风险",
                        "概念": "测试概念",
                        "ATR14": atr,
                    }
                ]
            },
        }

    def test_unpromoted_strategy_uses_atr_reference_policy(self) -> None:
        result = normalize_picker_payload(self.spec, self.payload(gate=False))
        row = result["rows"][0]

        self.assertEqual(row["advice_status"], "reference_only")
        self.assertEqual(row["advice_holding_days"], 7)
        self.assertEqual(row["policy_version"], "atr-reference-v1")
        self.assertEqual(row["stop_loss_price"], 9.25)
        self.assertEqual(row["take_profit_price"], 11.50)

    def test_order_flow_groups_are_normalized(self) -> None:
        result = normalize_picker_payload(self.spec, self.payload(gate=False))
        row = result["rows"][0]

        self.assertEqual(row["large_order_flow"], 1234.5)
        self.assertEqual(row["retail_flow"], -456.7)
        self.assertEqual(row["order_flow_as_of"], "2026-07-14")
        self.assertEqual(row["order_flow_source"], "数据库分项")
        self.assertEqual(row["fund_flow_1d"], 100.0)
        self.assertEqual(row["fund_flow_3d"], 240.0)
        self.assertEqual(row["fund_flow_5d"], 360.0)
        self.assertEqual(row["shareholder_score"], 11.5)
        self.assertEqual(row["institutional_holding_ratio"], 28.0)
        self.assertEqual(row["shareholder_count_change"], -6.0)
        self.assertIn("机构持股28.00%", row["shareholder_review"])

    def test_market_earnings_metrics_are_preserved_for_result_review(self) -> None:
        """Catches the 3.0 evidence disappearing between engine and desktop UI."""
        payload = self.payload(gate=False)
        item = payload["sheets"][self.spec.top_sheet][0]
        item.update(
            {
                "PB": 0.8,
                "历史最低ROE%": 12.0,
                "隐含ROE%": 10.0,
                "股息率%": 5.0,
                "估算分红率%": 40.0,
                "分红修正系数": 1.25,
                "市赚率3.0": 1.0,
            }
        )

        row = normalize_picker_payload(self.spec, payload)["rows"][0]

        self.assertIn("pb", row)
        self.assertEqual(row["pb"], 0.8)
        self.assertEqual(row["historical_roe"], 12.0)
        self.assertEqual(row["implied_roe"], 10.0)
        self.assertEqual(row["dividend_yield"], 5.0)
        self.assertEqual(row["payout_ratio"], 40.0)
        self.assertEqual(row["dividend_correction"], 1.25)
        self.assertEqual(row["market_earnings_ratio"], 1.0)
        self.assertIn("三年最低ROE 12%", row["performance_review"])
        self.assertNotIn("隐含ROE", row["performance_review"])
        self.assertIn("市赚率3.0 1", row["performance_review"])

    def test_same_day_database_derived_grouped_flows_are_normalized(self) -> None:
        payload = self.payload(gate=False)
        item = payload["sheets"][self.spec.top_sheet][0]
        item["订单资金来源"] = "数据库汇总推导"

        row = normalize_picker_payload(self.spec, payload)["rows"][0]

        self.assertEqual(row["large_order_flow"], 1234.5)
        self.assertEqual(row["retail_flow"], -456.7)
        self.assertEqual(row["order_flow_as_of"], "2026-07-14")
        self.assertEqual(row["order_flow_source"], "数据库汇总推导")

    def test_legacy_grouped_snapshot_does_not_masquerade_as_direct_flow(self) -> None:
        payload = self.payload(gate=False)
        payload.pop("order_flow_schema")
        item = payload["sheets"][self.spec.top_sheet][0]
        item["今日大单资金_万元"] = 999.0
        item["今日散户资金_万元"] = -999.0

        row = normalize_picker_payload(self.spec, payload)["rows"][0]

        self.assertIsNone(row["large_order_flow"])
        self.assertIsNone(row["retail_flow"])
        self.assertIsNone(row["order_flow_as_of"])
        self.assertEqual(row["order_flow_source"], "历史口径不兼容")

    def test_promoted_policy_still_has_precedence(self) -> None:
        policy = {
            "policy_version": "validated-v1",
            "holding_days": 10,
            "atr_multiplier": 1.0,
            "reward_risk": 1.5,
            "stop_loss_floor": 0.04,
            "stop_loss_cap": 0.08,
            "take_profit_floor": 0.08,
            "take_profit_cap": 0.16,
        }

        result = normalize_picker_payload(
            self.spec,
            self.payload(gate=True, policy=policy),
        )
        row = result["rows"][0]

        self.assertEqual(row["advice_status"], "ready")
        self.assertEqual(row["advice_holding_days"], 10)
        self.assertEqual(row["policy_version"], "validated-v1")

    def test_invalid_atr_remains_blank(self) -> None:
        result = normalize_picker_payload(
            self.spec,
            self.payload(gate=False, atr=None),
        )
        row = result["rows"][0]

        self.assertEqual(row["advice_status"], "insufficient_data")
        self.assertIsNone(row["advice_holding_days"])
        self.assertIsNone(row["take_profit_price"])
        self.assertIsNone(row["stop_loss_price"])

    def test_v3_fixed_policy_does_not_require_atr(self) -> None:
        policy = {
            "policy_version": "high-precision-5d-v3",
            "holding_days": 5,
            "atr_multiplier": 1.5,
            "reward_risk": 2.0,
            "fixed_take_profit_pct": 0.03,
            "fixed_stop_loss_pct": 0.06,
            "position_fraction": 0.20,
        }

        result = normalize_picker_payload(
            self.spec,
            self.payload(gate=True, policy=policy, atr=None),
        )
        row = result["rows"][0]

        self.assertEqual(row["advice_status"], "ready")
        self.assertEqual(row["advice_holding_days"], 5)
        self.assertEqual(row["take_profit_pct"], 0.03)
        self.assertEqual(row["stop_loss_pct"], 0.06)
        self.assertEqual(row["take_profit_price"], 10.30)
        self.assertEqual(row["stop_loss_price"], 9.40)
        self.assertEqual(row["position_fraction"], 0.20)


if __name__ == "__main__":
    unittest.main()
