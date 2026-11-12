# -*- coding: utf-8 -*-
"""Post-update data maintenance for the strategy engines.

Runs after the InStock daily jobs: refreshes quarterly statements that are in
their disclosure season and repairs missing sessions in the daily history.
Prints a single JSON summary on the last line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import daily_history as dh
from fundamentals_refresh import refresh_fundamentals
from picker_backtest import direct_network_environment
from repair_history import fetch_adjusted


MAX_REPAIRS_PER_RUN = 200
FUNDAMENTALS_PATH =Path(__file__).resolve().parents[1] / "data" / "fundamentals" / "quarterly-point-in-time.pkl"


def main() -> int:
    summary: dict[str, object] = {}
    with direct_network_environment():
        try:
            import akshare as ak

            summary["fundamentals"] = refresh_fundamentals(
                FUNDAMENTALS_PATH, ak, log=lambda text: print(text, flush=True)
            )
        except Exception as exc:
            summary["fundamentals"] = {"error": str(exc)[:300]}

        try:
            calendar = dh.trading_calendar()
            hist_dir = dh.latest_hist_dir()
            if hist_dir is None:
                raise RuntimeError("沧浪之水 history cache not found")
            end = calendar.max()
            codes = dh.codes_needing_repair(hist_dir, calendar, end)
            print(f"history: {len(codes)} codes with missing sessions", flush=True)
            # Small batches per update: the free quote sources throttle bulk
            # downloads, and factor windows already tolerate a few old gaps.
            summary["history"] = dh.repair_histories(
                codes[:MAX_REPAIRS_PER_RUN], end, calendar, fetcher=fetch_adjusted, max_workers=2
            )
            summary["history"]["remaining"] = max(0, len(codes) - MAX_REPAIRS_PER_RUN)
            summary["history"]["incomplete"] = summary["history"]["incomplete"][:50]
            summary["history"]["failed"] = summary["history"]["failed"][:50]
        except Exception as exc:
            summary["history"] = {"error": str(exc)[:300]}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
