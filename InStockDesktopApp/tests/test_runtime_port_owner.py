from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from instock_app import runtime
from instock_app.runtime import InStockRuntime, listener_pid

LOOPBACK = 0x0100007F


class ListenerPidTests(unittest.TestCase):
    def test_listener_reported_with_state_zero_is_found(self) -> None:
        """Catches Windows 26300 hiding a running MariaDB from the app."""
        rows = [
            (11, 3306, LOOPBACK, 52254, 0),  # TIME_WAIT client leftovers
            (0, 3306, 0, 0, 32384),
        ]
        self.assertEqual(32384, listener_pid(rows, 3306))

    def test_classic_listen_state_still_works(self) -> None:
        self.assertEqual(2280, listener_pid([(2, 3306, 0, 0, 2280)], 3306))

    def test_connections_and_other_ports_are_ignored(self) -> None:
        rows = [(5, 3306, LOOPBACK, 50000, 4000), (2, 9988, 0, 0, 77)]
        self.assertIsNone(listener_pid(rows, 3306))

    def test_netstat_is_used_when_the_table_is_unavailable(self) -> None:
        with patch.object(runtime, "tcp_table_rows", side_effect=OSError("no iphlpapi")), patch.object(
            InStockRuntime, "_netstat_port_owner", return_value=1234
        ):
            self.assertEqual(1234, InStockRuntime._find_port_owner(3306))

    def test_live_table_finds_a_real_listener(self) -> None:
        import socket

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen()
        try:
            port = server.getsockname()[1]
            import os

            self.assertEqual(os.getpid(), InStockRuntime._find_port_owner(port))
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
