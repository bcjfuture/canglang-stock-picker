from __future__ import annotations

import json
import requests
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from picker_order_flow import (
    OrderFlowComponents,
    enrich_selected_rows,
    fetch_order_flow_components,
    parse_daily_flow_payload,
    read_order_flow_cache,
    write_order_flow_cache,
)


class PickerOrderFlowTests(unittest.TestCase):
    def payload(self, *rows: str) -> dict:
        return {"data": {"klines": list(rows)}}

    def test_remote_payload_groups_super_large_and_medium_small(self) -> None:
        parsed = parse_daily_flow_payload(
            "600900",
            "2026-07-30",
            self.payload(
                "2026-07-29,1,2,3,4,5",
                "2026-07-30,636393264,-282765216,-353628016,506381024,130012240",
            ),
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.large_order_wan, 63639.33)
        self.assertEqual(parsed.retail_order_wan, -63639.32)
        self.assertEqual(parsed.trade_date, "2026-07-30")

    def test_remote_payload_rejects_wrong_date(self) -> None:
        parsed = parse_daily_flow_payload(
            "600900",
            "2026-07-30",
            self.payload("2026-07-29,1,2,3,4,5"),
        )
        self.assertIsNone(parsed)

    def test_remote_payload_rejects_incomplete_component_pairs(self) -> None:
        parsed = parse_daily_flow_payload(
            "600900",
            "2026-07-30",
            self.payload("2026-07-30,636393264,-282765216,-353628016,-,130012240"),
        )

        self.assertIsNotNone(parsed)
        self.assertIsNone(parsed.large_order_wan)
        self.assertEqual(parsed.retail_order_wan, -63639.32)

    def test_remote_request_uses_accessible_h5_history_proxy_first(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = self.payload(
            "2026-07-30,636393264,-282765216,-353628016,506381024,130012240"
        )
        with patch("picker_order_flow.requests.get", return_value=response) as get:
            parsed = fetch_order_flow_components("600900", "2026-07-30")

        self.assertIsNotNone(parsed)
        self.assertEqual(
            get.call_args.args[0],
            "https://emdatah5.eastmoney.com/dc/ZJLX/getDBHistoryData",
        )
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["ut"], "b2884a393a59ad64002292a3e90d46a5")
        self.assertEqual(params["fields1"], "f1,f2,f3")
        self.assertEqual(params["fields2"], "f51,f52,f53,f54,f55,f56,f62,f63")

    def test_remote_request_falls_back_to_direct_history_endpoint(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = self.payload(
            "2026-07-30,636393264,-282765216,-353628016,506381024,130012240"
        )
        with patch(
            "picker_order_flow.requests.get",
            side_effect=[requests.ConnectionError("h5 unavailable"), response],
        ) as get:
            parsed = fetch_order_flow_components("600900", "2026-07-30")

        self.assertIsNotNone(parsed)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(
            get.call_args_list[0].args[0],
            "https://emdatah5.eastmoney.com/dc/ZJLX/getDBHistoryData",
        )
        self.assertEqual(
            get.call_args_list[1].args[0],
            "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
        )

    def test_exact_date_cache_round_trip_and_stale_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "2026-07-30.json"
            row = OrderFlowComponents(
                code="600900",
                trade_date="2026-07-30",
                large_order_wan=50638.1,
                retail_order_wan=-28276.52,
                source="实时分项",
                fetched_at="2026-07-30T20:00:00",
            )
            write_order_flow_cache(path, "2026-07-30", {"600900": row})

            loaded = read_order_flow_cache(path, "2026-07-30")
            stale = read_order_flow_cache(path, "2026-07-29")

        self.assertEqual(loaded["600900"].large_order_wan, 50638.1)
        self.assertEqual(loaded["600900"].source, "当日分项缓存")
        self.assertEqual(stale, {})

    def test_enrichment_prefers_database_then_cache_and_keeps_order(self) -> None:
        rows = [
            {
                "代码": "600900",
                "综合评分": 145.6,
                "今日大单净额_万元": 3000.0,
                "今日散户净额_万元": -2000.0,
                "订单资金日期": "2026-07-30",
                "订单资金来源": "数据库分项",
            },
            {"代码": "600036", "综合评分": 142.6},
        ]
        cache = {
            "600900": OrderFlowComponents(
                "600900", "2026-07-30", 999.0, -999.0, "当日分项缓存", ""
            ),
            "600036": OrderFlowComponents(
                "600036", "2026-07-30", 4200.0, -1800.0, "当日分项缓存", ""
            ),
        }

        enriched = enrich_selected_rows(
            rows,
            "2026-07-30",
            cache=cache,
            fetcher=lambda *_args, **_kwargs: self.fail("remote should not run"),
        )

        self.assertEqual([row["代码"] for row in enriched], ["600900", "600036"])
        self.assertEqual([row["综合评分"] for row in enriched], [145.6, 142.6])
        self.assertEqual(enriched[0]["今日大单净额_万元"], 3000.0)
        self.assertEqual(enriched[1]["今日大单净额_万元"], 4200.0)
        self.assertEqual(enriched[1]["今日散户净额_万元"], -1800.0)

    def test_enrichment_accepts_same_day_database_derived_components_without_remote(self) -> None:
        rows = [
            {
                "代码": "600875",
                "今日大单净额_万元": 24328.17,
                "今日散户净额_万元": -24328.17,
                "订单资金日期": "2026-08-04",
                "订单资金来源": "数据库汇总推导",
            }
        ]

        enriched = enrich_selected_rows(
            rows,
            "2026-08-04",
            cache={},
            fetcher=lambda *_args, **_kwargs: self.fail("remote should not run"),
        )

        self.assertEqual(enriched[0]["今日大单净额_万元"], 24328.17)
        self.assertEqual(enriched[0]["今日散户净额_万元"], -24328.17)
        self.assertEqual(enriched[0]["订单资金来源"], "数据库汇总推导")

    def test_cache_document_contains_schema_and_exact_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flow.json"
            write_order_flow_cache(path, "2026-07-30", {})
            document = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(document["schema"], "grouped-order-flow-v2")
        self.assertEqual(document["trade_date"], "2026-07-30")

    def test_old_direct_component_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flow.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "direct-large-small-v1",
                        "trade_date": "2026-07-30",
                        "items": {
                            "600900": {
                                "trade_date": "2026-07-30",
                                "large_order_wan": 50638.1,
                                "retail_order_wan": -28276.52,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(read_order_flow_cache(path, "2026-07-30"), {})


if __name__ == "__main__":
    unittest.main()
