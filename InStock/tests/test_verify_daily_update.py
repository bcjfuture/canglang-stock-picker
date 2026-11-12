import unittest
from unittest.mock import MagicMock

import instock.job.verify_daily_update as verifier


class DailyUpdateVerifierTests(unittest.TestCase):
    def test_stock_coverage_rejects_partial_partition_and_accepts_full_partition(self):
        partial = {
            "cn_stock_spot": {"rows": 1906},
            "cn_stock_fund_flow": {"rows": 4584},
        }
        complete = {
            "cn_stock_spot": {"rows": 4400},
            "cn_stock_fund_flow": {"rows": 4584},
        }

        with self.assertRaisesRegex(RuntimeError, "coverage"):
            verifier.verify_stock_coverage(partial)
        verifier.verify_stock_coverage(complete)

    def test_indicator_coverage_rejects_partial_partition_and_accepts_full_partition(self):
        partial = {
            "cn_stock_spot": {"rows": 4597},
            "cn_stock_indicators": {"rows": 1915},
        }
        complete = {
            "cn_stock_spot": {"rows": 4597},
            "cn_stock_indicators": {"rows": 4400},
        }

        with self.assertRaisesRegex(RuntimeError, "indicator coverage"):
            verifier.verify_indicator_coverage(partial)
        verifier.verify_indicator_coverage(complete)

    def test_verify_table_accepts_positive_unique_partition(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (90, 90, 0)

        result = verifier.verify_table(
            cursor,
            "cn_stock_fund_flow_industry",
            "2026-07-03",
            "name",
        )

        self.assertEqual(result["rows"], 90)
        self.assertEqual(result["unique_keys"], 90)
        self.assertEqual(result["null_keys"], 0)

    def test_verify_table_rejects_empty_duplicate_or_null_key_partition(self):
        cases = (
            (0, 0, 0),
            (90, 89, 0),
            (90, 90, 1),
        )

        for row in cases:
            cursor = MagicMock()
            cursor.fetchone.return_value = row
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                verifier.verify_table(
                    cursor,
                    "cn_stock_fund_flow_industry",
                    "2026-07-03",
                    "name",
                )

    def test_verify_table_rejects_unsafe_identifiers(self):
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "unsafe SQL identifier"):
            verifier.verify_table(cursor, "bad;drop", "2026-07-03", "name")

        cursor.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
