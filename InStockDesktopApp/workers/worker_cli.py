from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    source_root = project_root / "src"
    if source_root.exists():
        sys.path.insert(0, str(source_root))

from instock_app.settings import AppPaths
from workers.common import emit, write_json_atomic
from workers.backtest_worker import run_backtest
from workers.export_worker import export_workbook
from workers.picker_worker import run_picker
from workers.recent_backtest_worker import run_recent_backtest
from workers.sector_forecast_worker import run_sector_forecast
from workers.update_worker import UpdateFailed, run_update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="沧浪之水 Smart Picker worker")
    sub = parser.add_subparsers(dest="command", required=True)
    update = sub.add_parser("update")
    update.add_argument("--result", type=Path, required=True)
    pick = sub.add_parser("pick")
    pick.add_argument("--strategy", required=True)
    pick.add_argument("--result", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--start", default="2022-01-01")
    backtest.add_argument("--end")
    backtest.add_argument("--result", type=Path, required=True)
    backtest.add_argument("--refresh", action="store_true")
    backtest.add_argument("--source", type=Path)
    recent_backtest = sub.add_parser("recent-backtest")
    recent_backtest.add_argument("--result", type=Path, required=True)
    forecast = sub.add_parser("sector-forecast")
    forecast.add_argument("--result", type=Path, required=True)
    forecast.add_argument("--refresh-history", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = AppPaths.default()
    try:
        if args.command == "update":
            result = run_update(paths, emit, result_path=args.result)
            write_json_atomic(args.result, result)
            return 0
        if args.command == "pick":
            result = run_picker(args.strategy, paths, emit)
            write_json_atomic(args.result, result)
            return 0
        if args.command == "export":
            import json

            result = json.loads(args.input.read_text(encoding="utf-8"))
            path = export_workbook(result, args.output)
            emit("result", f"Excel 已导出：{path}", 100, {"path": str(path)})
            return 0
        if args.command == "backtest":
            result = run_backtest(
                paths, emit, start=args.start, end=args.end, refresh=args.refresh,
                source_path=args.source,
            )
            write_json_atomic(args.result, result)
            return 0
        if args.command == "recent-backtest":
            result = run_recent_backtest(paths, emit)
            write_json_atomic(args.result, result)
            return 0
        if args.command == "sector-forecast":
            result = run_sector_forecast(
                paths, emit, refresh_history=args.refresh_history
            )
            write_json_atomic(args.result, result)
            return 0
    except UpdateFailed as exc:
        emit("error", str(exc), 100, {"error": str(exc)})
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
