# -*- coding: utf-8 -*-
"""Fill missing sessions in the daily history used by strategy factors.

usage: repair_history.py [end_date] [--max N]
Prints one JSON line per progress step and a final JSON summary.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

import daily_history as dh
from picker_backtest import _fetch_akshare, _fetch_tencent, _normalize_bars, direct_network_environment


def fetch_adjusted(code: str, start: str, end: str) -> pd.DataFrame:
    errors = []
    for fetcher in (_fetch_tencent, _fetch_akshare):
        try:
            bars = _normalize_bars(fetcher(code, start, end))
            if not bars.empty:
                return bars
        except Exception as exc:  # fall through to the next source
            errors.append(f"{fetcher.__name__}: {exc}")
    raise RuntimeError(" | ".join(errors) or "no bars")


def main(argv: list[str]) -> int:
    limit = None
    if "--max" in argv:
        index = argv.index("--max")
        limit = int(argv[index + 1])
        argv = argv[:index] + argv[index + 2:]
    calendar = dh.trading_calendar()
    hist_dir = dh.latest_hist_dir()
    if hist_dir is None:
        print(json.dumps({"error": "沧浪之水 history cache not found"}, ensure_ascii=False))
        return 1
    end = pd.Timestamp(argv[0]) if argv else calendar.max()
    codes = dh.codes_needing_repair(hist_dir, calendar, end)
    if limit is not None:
        codes = codes[:limit]
    print(json.dumps({"stage": "scan", "codes_with_gaps": len(codes)}, ensure_ascii=False), flush=True)

    def progress(done: int, total: int) -> None:
        print(json.dumps({"stage": "repair", "done": done, "total": total}), flush=True)

    with direct_network_environment():
        outcome = dh.repair_histories(codes, end, calendar, fetcher=fetch_adjusted, progress=progress)
    outcome["end"] = end.date().isoformat()
    print(json.dumps(outcome, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
