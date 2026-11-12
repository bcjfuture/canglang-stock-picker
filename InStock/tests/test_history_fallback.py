import datetime
import unittest

import pandas as pd
import requests

from instock.core.history_fallback import fetch_tencent_stock_history


class _FakeResponse:
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


class HistoryFallbackTests(unittest.TestCase):
    def test_tencent_history_maps_rows_and_appends_latest_spot(self):
        latest_spot = pd.Series(
            {
                "date": datetime.date(2026, 8, 18),
                "open_price": 10.15,
                "new_price": 10.30,
                "high_price": 10.40,
                "low_price": 10.05,
                "volume": 150000,
                "deal_amount": 1550000,
                "amplitude": 3.47,
                "change_rate": 1.98,
                "ups_downs": 0.20,
                "turnoverrate": 1.25,
            }
        )

        result = fetch_tencent_stock_history(
            "000001",
            "20230819",
            "20260818",
            latest_spot=latest_spot,
            request_get=lambda *_args, **_kwargs: _FakeResponse(),
        )

        self.assertEqual(
            list(result.columns),
            [
                "date",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "amplitude",
                "quote_change",
                "ups_downs",
                "turnover",
            ],
        )
        self.assertEqual(result["date"].tolist(), ["2026-08-14", "2026-08-17", "2026-08-18"])
        self.assertAlmostEqual(result.iloc[0]["amount"], 1258680.0)
        self.assertAlmostEqual(result.iloc[1]["quote_change"], -0.9803921569)
        self.assertEqual(result.iloc[-1]["volume"], 1500.0)
        self.assertEqual(result.iloc[-1]["amount"], 1550000.0)

    def test_tencent_history_uses_secondary_endpoint_after_501(self):
        attempts = []

        def first_blocked_then_success(url, *_args, **_kwargs):
            attempts.append(url)
            if len(attempts) == 1:
                raise requests.HTTPError("501 Server Error")
            return _FakeResponse()

        result = fetch_tencent_stock_history(
            "000001",
            "20230819",
            "20260818",
            request_get=first_blocked_then_success,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(len(result.index), 2)
        self.assertNotEqual(attempts[0], attempts[1])


if __name__ == "__main__":
    unittest.main()
