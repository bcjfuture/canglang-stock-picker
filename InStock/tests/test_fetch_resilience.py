import unittest
from datetime import datetime
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from instock.core.eastmoney_fetcher import eastmoney_fetcher
from instock.core.fund_flow_fallback import _fetch_xuangu_fund_flow_rank
import instock.core.stockfetch as stockfetch
import instock.core.tablestructure as tablestructure


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.trust_env = True
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return _FakeResponse(
            {
                "success": True,
                "result": {
                    "nextpage": False,
                    "data": [
                        {
                            "SECURITY_CODE": "600000",
                            "SECURITY_NAME_ABBR": "PF Bank",
                            "NEW_PRICE": 8.61,
                            "CHANGE_RATE": -1.37,
                            "NET_INFLOW": 123456,
                            "NETINFLOW_3DAYS": 654321,
                            "NETINFLOW_5DAYS": 777777,
                        }
                    ],
                },
            }
        )


class _DisconnectingSession:
    def __init__(self):
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        raise requests.ConnectionError("remote closed connection")


class FetchResilienceTests(unittest.TestCase):
    def test_eastmoney_fetcher_ignores_environment_proxy_by_default(self):
        fetcher = eastmoney_fetcher()

        self.assertFalse(fetcher.session.trust_env)

    def test_eastmoney_fetcher_does_not_multiply_adapter_and_outer_retries(self):
        fetcher = eastmoney_fetcher()

        self.assertEqual(0, fetcher.session.adapters["http://"].max_retries.total)

    def test_eastmoney_connection_failures_retry_twice_without_console_spam(self):
        fetcher = eastmoney_fetcher()
        session = _DisconnectingSession()
        fetcher.session = session
        fetcher.proxies = {}

        with patch("instock.core.eastmoney_fetcher.time.sleep"), patch(
            "builtins.print"
        ) as console:
            with self.assertRaises(requests.ConnectionError):
                fetcher.make_request("http://push2.example.test", retry=2)

        self.assertEqual(2, session.calls)
        console.assert_not_called()

    def test_stock_fetch_prefers_quote_snapshot_before_unstable_push2(self):
        columns = list(tablestructure.TABLE_CN_STOCK_SPOT["columns"])[1:]
        row = {column: np.nan for column in columns}
        row.update({"code": "600000", "name": "PF Bank", "new_price": 8.61})
        refreshed = pd.DataFrame([row])

        with patch(
            "instock.core.fallback_quote.load_latest_snapshot",
            return_value=refreshed,
        ), patch(
            "instock.core.fallback_quote.refresh_quotes",
            return_value=refreshed,
        ), patch.object(
            stockfetch.she,
            "stock_zh_a_spot_em",
            return_value=None,
        ) as primary:
            result = stockfetch.fetch_stocks(datetime(2026, 7, 7))

        primary.assert_not_called()
        self.assertEqual(len(result.index), 1)
        self.assertEqual(result.iloc[0]["date"], "2026-07-07")

    def test_xuangu_fund_flow_uses_direct_session_and_maps_rows(self):
        fake_session = _FakeSession()

        with patch("instock.core.fund_flow_fallback.requests.Session", return_value=fake_session):
            result = _fetch_xuangu_fund_flow_rank(0)

        self.assertFalse(fake_session.trust_env)
        self.assertEqual(result.iloc[0]["code"], "600000")
        self.assertEqual(result.iloc[0]["fund_amount"], 123456)

    def test_stock_fund_flow_prefers_xuangu_before_unstable_push2(self):
        columns = list(tablestructure.CN_STOCK_FUND_FLOW[0]["columns"])
        row = {column: None for column in columns}
        row.update({"code": "600000", "name": "PF Bank", "new_price": 8.61, "fund_amount": 123456})
        fallback = pd.DataFrame([row], columns=columns)

        with patch(
            "instock.core.fund_flow_fallback.fetch_individual_rank",
            return_value=fallback,
        ), patch.object(
            stockfetch.sff,
            "stock_individual_fund_flow_rank",
            return_value=None,
        ) as primary:
            result = stockfetch.fetch_stocks_fund_flow(0)

        primary.assert_not_called()
        self.assertEqual(len(result.index), 1)
        self.assertEqual(result.iloc[0]["fund_amount"], 123456)

    def test_stock_history_network_failures_are_rate_limited_by_error_type(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            stockfetch,
            "stock_hist_cache_path",
            temp_dir,
        ), patch.object(
            stockfetch.she,
            "stock_zh_a_hist",
            side_effect=RuntimeError("upstream json parse failed"),
        ), patch(
            "instock.core.history_fallback.fetch_tencent_stock_history",
            side_effect=RuntimeError("fallback unavailable"),
        ):
            if hasattr(stockfetch, "_eastmoney_history_unavailable"):
                stockfetch._eastmoney_history_unavailable.clear()
            if hasattr(stockfetch, "_hist_failure_log_counts"):
                stockfetch._hist_failure_log_counts.clear()
            with self.assertLogs(level="ERROR") as logs:
                for offset in range(6):
                    stockfetch.stock_hist_cache(
                        str(600000 + offset),
                        "20260707",
                        is_cache=False,
                        adjust="qfq",
                    )

        self.assertEqual(len(logs.output), 4)
        self.assertIn("further similar history failures suppressed", logs.output[-1])

    def test_stock_history_uses_tencent_when_eastmoney_disconnects(self):
        class HistoryResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "code": 0,
                    "data": {
                        "sz000001": {
                            "qfqday": [
                                ["2026-08-14", "10.00", "10.20", "10.30", "9.90", "1234.000"],
                                ["2026-08-17", "10.20", "10.10", "10.25", "10.00", "1100.000"],
                            ]
                        }
                    },
                }

        class HistorySession:
            trust_env = True

            def get(self, *_args, **_kwargs):
                return HistoryResponse()

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            stockfetch,
            "stock_hist_cache_path",
            temp_dir,
        ), patch.object(
            stockfetch.she,
            "stock_zh_a_hist",
            side_effect=requests.ConnectionError("remote closed connection"),
        ), patch(
            "instock.core.history_fallback.requests.Session",
            return_value=HistorySession(),
        ):
            stockfetch._eastmoney_history_unavailable.clear()
            result = stockfetch.stock_hist_cache(
                "000001",
                "20230819",
                is_cache=False,
                adjust="qfq",
            )

        self.assertEqual(result["date"].tolist(), ["2026-08-14", "2026-08-17"])
        self.assertEqual(result.iloc[-1]["close"], 10.10)

    def test_stock_history_stops_probing_eastmoney_after_disconnect(self):
        calls = []
        fallback = pd.DataFrame(
            [["2026-08-17", 10.0, 10.1, 10.2, 9.9, 1000.0, 1010000.0, 3.0, 1.0, 0.1, 0.0]],
            columns=[
                "date", "open", "close", "high", "low", "volume", "amount",
                "amplitude", "quote_change", "ups_downs", "turnover",
            ],
        )

        def disconnected(*_args, **_kwargs):
            calls.append(1)
            raise requests.ConnectionError("remote closed connection")

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            stockfetch,
            "stock_hist_cache_path",
            temp_dir,
        ), patch.object(
            stockfetch.she,
            "stock_zh_a_hist",
            side_effect=disconnected,
        ), patch(
            "instock.core.history_fallback.fetch_tencent_stock_history",
            return_value=fallback,
        ):
            stockfetch._eastmoney_history_unavailable.clear()
            first = stockfetch.stock_hist_cache("000001", "20230819", is_cache=False, adjust="qfq")
            second = stockfetch.stock_hist_cache("000002", "20230819", is_cache=False, adjust="qfq")

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(first.index), 1)
        self.assertEqual(len(second.index), 1)


if __name__ == "__main__":
    unittest.main()
