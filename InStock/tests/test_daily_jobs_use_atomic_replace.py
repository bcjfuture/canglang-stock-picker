import unittest
from pathlib import Path


class DailyJobAtomicWriteTests(unittest.TestCase):
    JOB_FILES = (
        "basic_data_daily_job.py",
        "basic_data_other_daily_job.py",
        "selection_data_daily_job.py",
        "indicators_data_daily_job.py",
        "strategy_data_daily_job.py",
    )

    def test_desktop_jobs_do_not_split_delete_and_insert(self):
        job_dir = Path(__file__).parents[1] / "instock" / "job"

        for filename in self.JOB_FILES:
            source = (job_dir / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertNotIn("mdb.executeSql(del_sql)", source)
                self.assertNotIn("mdb.insert_db_from_df(", source)
                self.assertIn("mdb.replace_partition_from_df(", source)


if __name__ == "__main__":
    unittest.main()
