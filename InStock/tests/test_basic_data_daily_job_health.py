import datetime
import unittest

from instock.job import basic_data_daily_job as job


class BasicDataDailyJobHealthTests(unittest.TestCase):
    def test_snapshot_health_rejects_missing_trade_date_partition(self):
        counts = {
            "cn_stock_spot": 0,
            "cn_etf_spot": 800,
        }

        with self.assertRaisesRegex(RuntimeError, "cn_stock_spot has no rows"):
            job.require_snapshot_partitions(
                datetime.date(2026, 8, 17),
                lambda table_name, _date: counts[table_name],
            )

    def test_snapshot_health_accepts_complete_trade_date_partitions(self):
        counts = {
            "cn_stock_spot": 4597,
            "cn_etf_spot": 800,
        }

        job.require_snapshot_partitions(
            datetime.date(2026, 8, 17),
            lambda table_name, _date: counts[table_name],
        )


if __name__ == "__main__":
    unittest.main()
