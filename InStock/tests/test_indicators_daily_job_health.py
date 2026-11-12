import datetime
import unittest

from instock.job import indicators_data_daily_job as job


class IndicatorsDailyJobHealthTests(unittest.TestCase):
    def test_indicator_health_rejects_partial_trade_date_partition(self):
        counts = {
            "cn_stock_spot": 4597,
            "cn_stock_indicators": 1915,
        }

        with self.assertRaisesRegex(RuntimeError, "coverage is incomplete"):
            job.require_indicator_partition(
                datetime.date(2026, 8, 18),
                lambda table_name, _date: counts[table_name],
            )

    def test_indicator_health_accepts_at_least_95_percent_coverage(self):
        counts = {
            "cn_stock_spot": 4597,
            "cn_stock_indicators": 4400,
        }

        job.require_indicator_partition(
            datetime.date(2026, 8, 18),
            lambda table_name, _date: counts[table_name],
        )


if __name__ == "__main__":
    unittest.main()
