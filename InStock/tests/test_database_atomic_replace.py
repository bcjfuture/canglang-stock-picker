import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

import instock.lib.database as database


class AtomicReplaceTests(unittest.TestCase):
    def test_rejects_duplicate_keys_before_opening_database(self):
        data = pd.DataFrame([
            {"date": "2026-07-03", "code": "000001"},
            {"date": "2026-07-03", "code": "000001"},
        ])

        with patch.object(database, "engine") as db_engine:
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                database.replace_partition_from_df(
                    data,
                    "sample",
                    "date",
                    "2026-07-03",
                    ["date", "code"],
                    None,
                    primary_keys="`date`,`code`",
                )

        db_engine.assert_not_called()

    def test_rejects_invalid_partitions_before_opening_database(self):
        cases = {
            "empty": pd.DataFrame(columns=["date", "code"]),
            "missing": pd.DataFrame([{"date": "2026-07-03"}]),
            "null": pd.DataFrame([{"date": "2026-07-03", "code": None}]),
            "mixed": pd.DataFrame([
                {"date": "2026-07-03", "code": "000001"},
                {"date": "2026-07-02", "code": "000002"},
            ]),
        }

        for name, data in cases.items():
            with self.subTest(name=name), patch.object(database, "engine") as db_engine:
                with self.assertRaises(ValueError):
                    database.replace_partition_from_df(
                        data,
                        "sample",
                        "date",
                        datetime(2026, 7, 3),
                        ["date", "code"],
                        None,
                        primary_keys="`date`,`code`",
                    )
                db_engine.assert_not_called()

    def test_uses_one_transaction_for_delete_append_and_count(self):
        data = pd.DataFrame([{"date": "2026-07-03", "code": "000001"}])
        db_engine = MagicMock()
        connection = db_engine.begin.return_value.__enter__.return_value
        connection.execute.return_value.scalar_one.return_value = 1

        with patch.object(database, "checkTableIsExist", return_value=True), patch.object(
            database, "engine", return_value=db_engine
        ), patch.object(pd.DataFrame, "to_sql") as to_sql:
            count = database.replace_partition_from_df(
                data,
                "sample",
                "date",
                "2026-07-03",
                ["date", "code"],
                None,
                primary_keys="`date`,`code`",
            )

        self.assertEqual(count, 1)
        self.assertIs(to_sql.call_args.kwargs["con"], connection)
        self.assertEqual(connection.execute.call_count, 2)
        db_engine.begin.assert_called_once_with()
        db_engine.dispose.assert_called_once_with()

    def test_insert_failure_is_reraised(self):
        data = pd.DataFrame([{"date": "2026-07-03", "code": "000001"}])
        db_engine = MagicMock()

        with patch.object(database, "checkTableIsExist", return_value=True), patch.object(
            database, "engine", return_value=db_engine
        ), patch.object(
            pd.DataFrame,
            "to_sql",
            side_effect=RuntimeError("insert failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "insert failed"):
                database.replace_partition_from_df(
                    data,
                    "sample",
                    "date",
                    "2026-07-03",
                    ["date", "code"],
                    None,
                    primary_keys="`date`,`code`",
                )

        db_engine.dispose.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
