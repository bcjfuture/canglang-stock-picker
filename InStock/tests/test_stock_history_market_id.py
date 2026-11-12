import unittest
from unittest.mock import patch

from instock.core.crawling import stock_hist_em


class _EmptyResponse:
    def json(self):
        return {"data": None}


class StockHistoryMarketIdTests(unittest.TestCase):
    def test_shanghai_fund_uses_shanghai_market_id(self):
        captured = {}

        def fake_request(_url, params=None, **_kwargs):
            captured.update(params or {})
            return _EmptyResponse()

        with patch.object(stock_hist_em.fetcher, "make_request", side_effect=fake_request):
            stock_hist_em.stock_zh_a_hist(
                symbol="510300",
                period="daily",
                start_date="20260701",
                end_date="20260706",
                adjust="qfq",
            )

        self.assertEqual(captured["secid"], "1.510300")


if __name__ == "__main__":
    unittest.main()
