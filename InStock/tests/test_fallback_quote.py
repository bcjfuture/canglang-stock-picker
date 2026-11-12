import unittest
from datetime import datetime
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from instock.core.fallback_quote import refresh_quotes
import instock.core.fallback_quote as fallback_quote
import instock.core.stockfetch as stockfetch
import instock.core.tablestructure as tablestructure


class FallbackQuoteTests(unittest.TestCase):
    def test_quote_coverage_gate_rejects_half_market_and_accepts_near_full_market(self):
        self.assertFalse(fallback_quote.quote_coverage_is_sufficient(1906, 4584))
        self.assertTrue(fallback_quote.quote_coverage_is_sufficient(4400, 4584))

    def test_merge_quote_universe_restores_missing_codes_without_overwriting_existing_rows(self):
        columns = ["code", "name", "new_price", "industry", "basic_eps"]
        previous = pd.DataFrame(
            [{"code": "600000", "name": "浦发银行", "new_price": 8.61,
              "industry": "银行", "basic_eps": 1.23}],
            columns=columns,
        )
        universe = pd.DataFrame(
            [
                {"code": "600000", "name": "弱字段名称", "new_price": 8.60},
                {"code": "000001", "name": "平安银行", "new_price": 12.34},
            ]
        )

        result = fallback_quote.merge_quote_universe(previous, universe, columns)

        self.assertEqual(result["code"].tolist(), ["600000", "000001"])
        existing = result.loc[result["code"] == "600000"].iloc[0]
        added = result.loc[result["code"] == "000001"].iloc[0]
        self.assertEqual(existing["name"], "浦发银行")
        self.assertEqual(existing["industry"], "银行")
        self.assertEqual(existing["basic_eps"], 1.23)
        self.assertEqual(added["name"], "平安银行")
        self.assertEqual(added["new_price"], 12.34)
        self.assertTrue(pd.isna(added["industry"]))

    def test_refreshes_market_fields_and_preserves_fundamentals(self):
        previous = pd.DataFrame(
            [
                {
                    "code": "600000",
                    "name": "旧名称",
                    "new_price": 8.73,
                    "change_rate": 0.0,
                    "ups_downs": 0.0,
                    "volume": 1,
                    "deal_amount": 1,
                    "open_price": 1.0,
                    "high_price": 1.0,
                    "low_price": 1.0,
                    "pre_close_price": 1.0,
                    "turnoverrate": 0.0,
                    "total_market_cap": 1,
                    "free_cap": 1,
                    "basic_eps": 1.23,
                    "industry": "银行",
                }
            ]
        )
        quote = (
            'v_sh600000="1~浦发银行~600000~8.61~8.73~8.69~684341~292475~391866~'
            '8.60~377~8.59~81~8.58~2218~8.57~561~8.56~473~8.61~2521~8.62~'
            '12555~8.63~1824~8.64~2012~8.65~3249~~20260630161405~-0.12~-1.37~'
            '8.70~8.49~8.61/684341/586217640~684341~58622~0.21~5.70~~8.70~8.49~'
            '2.41~2867.63~2867.63~0.38~9.60~7.86~0.91~-18451~8.57~4.01~5.73~'
            '~~~0.19~58621.7640~0.0000~0~ ~GP-A~-30.79~-7.22~4.33~6.03~0.49~'
            '14.39~8.49~-9.65~-7.62~-15.42~33305838300~33305838300";'
        )

        result = refresh_quotes(previous, lambda symbols: quote)

        row = result.iloc[0]
        self.assertEqual(row["name"], "浦发银行")
        self.assertEqual(row["new_price"], 8.61)
        self.assertEqual(row["volume"], 68434100)
        self.assertEqual(row["deal_amount"], 586217640)
        self.assertEqual(row["total_market_cap"], 286763000000)
        self.assertEqual(row["basic_eps"], 1.23)
        self.assertEqual(row["industry"], "银行")

    def test_stock_fetch_uses_fallback_when_primary_source_disconnects(self):
        columns = list(tablestructure.TABLE_CN_STOCK_SPOT["columns"])[1:]
        values = {column: np.nan for column in columns}
        values.update({"code": "600000", "name": "浦发银行", "new_price": 8.61})
        previous = pd.DataFrame([values])

        with patch.object(
            stockfetch.she,
            "stock_zh_a_spot_em",
            side_effect=ConnectionError("primary disconnected"),
        ), patch(
            "instock.core.fallback_quote.load_latest_snapshot",
            return_value=previous,
        ), patch(
            "instock.core.fallback_quote.load_latest_stock_universe",
            return_value=previous,
        ), patch(
            "instock.core.fallback_quote.refresh_quotes",
            return_value=previous,
        ):
            result = stockfetch.fetch_stocks(datetime(2026, 6, 30))

        self.assertEqual(len(result.index), 1)
        self.assertEqual(result.iloc[0]["date"], "2026-06-30")

    def test_stock_quote_override_seeds_snapshot_from_full_universe(self):
        columns = list(tablestructure.TABLE_CN_STOCK_SPOT["columns"])[1:]
        existing = {column: np.nan for column in columns}
        existing.update({"code": "600000", "name": "浦发银行", "new_price": 8.61})
        added = {column: np.nan for column in columns}
        added.update({"code": "000001", "name": "平安银行", "new_price": 11.44})
        previous = pd.DataFrame([existing], columns=columns)
        universe = pd.DataFrame([added], columns=columns)

        with patch(
            "instock.core.fallback_quote.load_latest_snapshot", return_value=previous
        ), patch(
            "instock.core.fallback_quote.load_latest_stock_universe", return_value=universe
        ), patch(
            "instock.core.fallback_quote.refresh_quotes", side_effect=lambda frame: frame
        ):
            result = stockfetch.fetch_stocks(datetime(2026, 8, 4))

        self.assertEqual(set(result["code"]), {"600000", "000001"})

    def test_history_cache_rolls_forward_previous_cache_without_network(self):
        previous = pd.DataFrame(
            [["2026-06-29", 8.7, 8.73, 8.8, 8.6, 1000, 870000, 2.3, 0.2, 0.02, 0.1]],
            columns=list(tablestructure.CN_STOCK_HIST_DATA["columns"]),
        )
        latest_spot = [
            (datetime(2026, 6, 30).date(), 8.69, 8.61, 8.70, 8.49, 68434100,
             586217640, 2.41, -1.37, -0.12, 0.21)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            prior_dir = os.path.join(temp_dir, "202306", "20230630")
            os.makedirs(prior_dir)
            previous.to_pickle(os.path.join(prior_dir, "600000qfq.gzip.pickle"), compression="gzip")

            with patch.object(stockfetch, "stock_hist_cache_path", temp_dir), patch.object(
                stockfetch.she, "stock_zh_a_hist", side_effect=AssertionError("network should not be used")
            ) as network, patch(
                "instock.lib.database.executeSqlFetch", return_value=latest_spot
            ):
                result = stockfetch.stock_hist_cache("600000", "20230701", adjust="qfq")

        network.assert_not_called()
        self.assertEqual(len(result.index), 2)
        self.assertEqual(str(result.iloc[-1]["date"]), "2026-06-30")
        self.assertEqual(result.iloc[-1]["close"], 8.61)
        self.assertEqual(result.iloc[-1]["volume"], 684341)

    def test_history_cache_reuses_loaded_spot_row_without_per_stock_query(self):
        previous = pd.DataFrame(
            [["2026-06-29", 8.7, 8.73, 8.8, 8.6, 1000, 870000, 2.3, 0.2, 0.02, 0.1]],
            columns=list(tablestructure.CN_STOCK_HIST_DATA["columns"]),
        )
        latest_spot = pd.Series(
            {
                "date": datetime(2026, 6, 30).date(),
                "open_price": 8.69,
                "new_price": 8.61,
                "high_price": 8.70,
                "low_price": 8.49,
                "volume": 68434100,
                "deal_amount": 586217640,
                "amplitude": 2.41,
                "change_rate": -1.37,
                "ups_downs": -0.12,
                "turnoverrate": 0.21,
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            prior_dir = os.path.join(temp_dir, "202306", "20230630")
            os.makedirs(prior_dir)
            previous.to_pickle(os.path.join(prior_dir, "600000qfq.gzip.pickle"), compression="gzip")

            with patch.object(stockfetch, "stock_hist_cache_path", temp_dir), patch.object(
                stockfetch.she, "stock_zh_a_hist", side_effect=AssertionError("network should not be used")
            ), patch(
                "instock.lib.database.executeSqlFetch",
                side_effect=AssertionError("per-stock SQL should not be used"),
            ) as database_query:
                result = stockfetch.stock_hist_cache(
                    "600000", "20230701", adjust="qfq", latest_spot=latest_spot
                )

        database_query.assert_not_called()
        self.assertEqual(len(result.index), 2)
        self.assertEqual(result.iloc[-1]["close"], 8.61)
        self.assertEqual(result.iloc[-1]["volume"], 684341)


if __name__ == "__main__":
    unittest.main()
