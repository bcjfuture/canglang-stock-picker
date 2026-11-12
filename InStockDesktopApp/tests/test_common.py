from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workers.common import write_json_atomic


class AtomicJsonWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_write_json_atomic_retries_transient_permission_error(self) -> None:
        target = self.root / "result.json"
        target.write_text('{"state": "old"}', encoding="utf-8")
        real_replace = os.replace
        attempts = 0

        def flaky_replace(source, destination):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise PermissionError("destination temporarily locked")
            real_replace(source, destination)

        with patch(
            "workers.common.os.replace", side_effect=flaky_replace
        ), patch("time.sleep") as sleep:
            try:
                write_json_atomic(target, {"state": "new"})
            except PermissionError as exc:
                self.fail(f"transient destination lock was not retried: {exc}")

        self.assertEqual(attempts, 3)
        self.assertEqual(
            json.loads(target.read_text(encoding="utf-8")), {"state": "new"}
        )
        self.assertEqual([call(0.05), call(0.1)], sleep.call_args_list)
        self.assertEqual([], list(self.root.glob("result.json.*.tmp")))

    def test_write_json_atomic_stops_retrying_and_cleans_temp_file(self) -> None:
        target = self.root / "result.json"
        target.write_text('{"state": "old"}', encoding="utf-8")

        with patch(
            "workers.common.os.replace",
            side_effect=PermissionError("destination remains locked"),
        ) as replace, patch("time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                write_json_atomic(target, {"state": "new"})

        self.assertEqual(replace.call_count, 7)
        self.assertEqual(
            [
                call(0.05),
                call(0.1),
                call(0.2),
                call(0.4),
                call(0.8),
                call(1.6),
            ],
            sleep.call_args_list,
        )
        self.assertEqual(
            json.loads(target.read_text(encoding="utf-8")), {"state": "old"}
        )
        self.assertEqual([], list(self.root.glob("result.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
