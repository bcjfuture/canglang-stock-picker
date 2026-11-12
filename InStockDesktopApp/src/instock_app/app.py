from __future__ import annotations

import argparse
import sys
import tkinter as tk
from datetime import datetime

import customtkinter as ctk

from .controller import AppController
from .result_store import ResultStore
from .runtime import InStockRuntime
from .settings import AppPaths
from .task_runner import TaskRunner
from .ui.main_window import MainWindow


def build_app(root: tk.Tk) -> MainWindow:
    paths = AppPaths.default()
    controller = AppController(
        paths=paths,
        runner=TaskRunner(),
        store=ResultStore(paths.data_dir / "store"),
        runtime=InStockRuntime(paths),
    )
    return MainWindow(root, controller)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="沧浪之水智能选股")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--test-auto-close-ms", type=int)
    args = parser.parse_args(argv)
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    if args.smoke_test:
        root.withdraw()
    window = build_app(root)
    if args.smoke_test:
        root.update_idletasks()
        message = f"APP_SMOKE=PASS pages={len(window.page_names)}"
        paths = AppPaths.default()
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        (paths.logs_dir / "smoke-test.log").write_text(
            f"{datetime.now().isoformat(timespec='seconds')} {message}\n", encoding="utf-8"
        )
        print(message, flush=True)
        root.destroy()
        return 0
    if args.test_auto_close_ms:
        root.after(args.test_auto_close_ms, root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
