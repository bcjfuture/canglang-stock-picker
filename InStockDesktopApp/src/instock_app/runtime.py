from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Callable

from .settings import AppPaths


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    name: str
    command_line: str


@dataclass(frozen=True)
class RuntimeStatus:
    db_running: bool
    web_running: bool
    db_pid: int | None
    web_pid: int | None


TcpRow = tuple[int, int, int, int, int]  # state, local port, remote address, remote port, pid


def tcp_table_rows() -> list[TcpRow]:
    """IPv4 TCP table straight from the OS, independent of netstat's text output."""
    import ctypes
    import socket
    import struct
    from ctypes import wintypes

    get_table = ctypes.windll.iphlpapi.GetExtendedTcpTable
    size = wintypes.DWORD(0)
    get_table(None, ctypes.byref(size), False, socket.AF_INET, 5, 0)  # TCP_TABLE_OWNER_PID_ALL
    for _attempt in range(3):
        buffer = ctypes.create_string_buffer(size.value)
        result = get_table(buffer, ctypes.byref(size), False, socket.AF_INET, 5, 0)
        if result == 0:
            break
        if result != 122:  # ERROR_INSUFFICIENT_BUFFER: the table grew, retry
            raise OSError(f"GetExtendedTcpTable failed: {result}")
    else:
        raise OSError("GetExtendedTcpTable kept growing")
    count = struct.unpack_from("I", buffer, 0)[0]
    rows: list[TcpRow] = []
    for index in range(count):
        state, _local_addr, local_port, remote_addr, remote_port, pid = struct.unpack_from(
            "IIIIII", buffer, 4 + index * 24
        )
        rows.append(
            (state, socket.ntohs(local_port & 0xFFFF), remote_addr, socket.ntohs(remote_port & 0xFFFF), pid)
        )
    return rows


def listener_pid(rows: list[TcpRow], port: int) -> int | None:
    """Process listening on ``port``.

    A listener has no remote endpoint.  Windows 11 build 26300 reports such
    sockets with state 0 instead of LISTEN (2), and netstat stopped printing
    them, so the state value is not relied on.
    """
    for state, local_port, remote_addr, remote_port, pid in rows:
        if local_port == port and remote_addr == 0 and remote_port == 0 and pid > 0 and state in (0, 2):
            return pid
    return None


PortOwner = Callable[[int], int | None]
ProcessLookup = Callable[[int], ProcessInfo | None]
ProcessStopper = Callable[[int], None]


class InStockRuntime:
    def __init__(
        self,
        paths: AppPaths,
        port_owner: PortOwner | None = None,
        process_info: ProcessLookup | None = None,
        stop_process: ProcessStopper | None = None,
    ) -> None:
        self.paths = paths
        self._port_owner = port_owner or self._find_port_owner
        self._process_info = process_info or self._lookup_process
        self._stop_process = stop_process or self._kill_process_tree

    def status(self) -> RuntimeStatus:
        db_pid = self._port_owner(3306)
        web_pid = self._port_owner(9988)
        return RuntimeStatus(
            db_running=db_pid is not None,
            web_running=web_pid is not None,
            db_pid=db_pid,
            web_pid=web_pid,
        )

    def start_database(self, timeout: float = 45.0) -> int:
        existing = self._port_owner(3306)
        if existing is not None:
            return existing
        log_path = self.paths.logs_dir.parent / "mariadb.err"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            [
                str(self.paths.mariadb_exe),
                f"--datadir={self.paths.mariadb_data_dir}",
                "--port=3306",
                "--bind-address=127.0.0.1",
                f"--log-error={log_path}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            owner = self._port_owner(3306)
            if owner is not None:
                return owner
            if process.poll() is not None:
                break
            time.sleep(0.25)
        raise RuntimeError("MariaDB 未在限定时间内启动，请查看 mariadb.err")

    def stop_instock(self) -> list[int]:
        stopped: list[int] = []
        web_pid = self._port_owner(9988)
        if web_pid is not None:
            self._stop_process(web_pid)
            stopped.append(web_pid)

        db_pid = self._port_owner(3306)
        if db_pid is None:
            return stopped
        info = self._process_info(db_pid)
        expected_data_dir = str(self.paths.mariadb_data_dir).lower()
        if (
            info is not None
            and (info.name.lower().startswith("mariadb") or info.name.lower().startswith("mysql"))
            and expected_data_dir in info.command_line.lower()
        ):
            self._stop_process(db_pid)
            stopped.append(db_pid)
        return stopped

    @staticmethod
    def _find_port_owner(port: int) -> int | None:
        try:
            owner = listener_pid(tcp_table_rows(), port)
        except (AttributeError, OSError, ValueError):
            owner = None
        return owner if owner is not None else InStockRuntime._netstat_port_owner(port)

    @staticmethod
    def _netstat_port_owner(port: int) -> int | None:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        completed = subprocess.run(
            ["netstat.exe", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            check=False,
        )
        pattern = re.compile(rf"^\s*TCP\s+\S*:{port}\s+\S+\s+LISTENING\s+(\d+)\s*$", re.I)
        for line in completed.stdout.splitlines():
            match = pattern.match(line)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _lookup_process(pid: int) -> ProcessInfo | None:
        script = (
            f"Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" | "
            "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
        )
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        data = json.loads(completed.stdout)
        return ProcessInfo(
            pid=int(data["ProcessId"]),
            name=str(data.get("Name", "")),
            command_line=str(data.get("CommandLine", "")),
        )

    @staticmethod
    def _kill_process_tree(pid: int) -> None:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            check=False,
        )
