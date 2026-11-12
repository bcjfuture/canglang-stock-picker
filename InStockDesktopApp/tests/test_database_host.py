from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import instock_tech_picker_data
from instock_app.settings import AppPaths
from workers import update_worker


class DatabaseHostTests(unittest.TestCase):
    """Catches "localhost", which tries IPv6 first and costs ~2 s per connection."""

    def test_update_jobs_connect_over_ipv4(self) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured.update(kwargs["env"])
            return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch.object(update_worker.subprocess, "run", fake_run):
            update_worker.subprocess_job_runner(AppPaths.default())("verify_daily_update.py", 5)
        self.assertEqual("127.0.0.1", captured["db_host"])

    def test_picker_engine_connects_over_ipv4(self) -> None:
        self.assertEqual("127.0.0.1", instock_tech_picker_data.DB["host"])


if __name__ == "__main__":
    unittest.main()