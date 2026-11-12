# -*- coding: utf-8 -*-
"""Keep the point-in-time quarterly statements current.

The statement cache only fetched quarters it had never seen, so a quarter
fetched early in its disclosure season stayed partial and later quarters were
never added.  Quarters whose disclosure window is open (or closed within the
grace period) are re-fetched and replace their earlier rows.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from high_precision_v2 import _fetch_with_retries, normalize_quarterly_statements, save_frame_atomic


# Statutory disclosure deadline per period end (month, day) -> (months later, day).
DEADLINES = {(3, 31): (4, 30), (6, 30): (8, 31), (9, 30): (10, 31), (12, 31): (4, 30)}
GRACE_DAYS = 20
HISTORY_YEARS = 2


def quarter_ends(today: date, years: int = HISTORY_YEARS) -> list[date]:
    ends = []
    for year in range(today.year - years, today.year + 1):
        for month, day in DEADLINES:
            end = date(year, month, day)
            if end < today:
                ends.append(end)
    return ends


def disclosure_deadline(period_end: date) -> date:
    month, day = DEADLINES[(period_end.month, period_end.day)]
    year = period_end.year + (1 if period_end.month == 12 else 0)
    return date(year, month, day)


def quarters_to_refresh(today: date, loaded: set[str]) -> list[str]:
    """Missing quarters plus any whose disclosure season is still running."""
    result = []
    for end in quarter_ends(today):
        key = end.strftime("%Y%m%d")
        open_window = end < today <= disclosure_deadline(end) + timedelta(days=GRACE_DAYS)
        if key not in loaded or open_window:
            result.append(key)
    return result


def refresh_fundamentals(cache_path: Path, ak_module, today: date | None = None, log=print) -> dict:
    today = today or date.today()
    cache_path = Path(cache_path)
    existing = pd.read_pickle(cache_path) if cache_path.exists() else pd.DataFrame()
    loaded = set()
    if not existing.empty:
        loaded = set(pd.to_datetime(existing["report_date"]).dt.strftime("%Y%m%d"))
    requested = quarters_to_refresh(today, loaded)
    refreshed, failures = [], []
    frame = existing
    for key in requested:
        log(f"fundamentals: fetching {key}")
        try:
            income = _fetch_with_retries(ak_module.stock_lrb_em, key, attempts=5)
            balance = _fetch_with_retries(ak_module.stock_zcfz_em, key, attempts=5)
            cashflow = _fetch_with_retries(ak_module.stock_xjll_em, key, attempts=5)
            fresh = normalize_quarterly_statements(key, income, balance, cashflow)
        except TypeError:
            # akshare returns no table before the first report of a quarter is filed.
            log(f"fundamentals: {key} has no published reports yet")
            continue
        except Exception as exc:
            failures.append({"report_date": key, "error": str(exc)[:300]})
            log(f"fundamentals: failed {key}: {exc}")
            continue
        if fresh.empty:
            failures.append({"report_date": key, "error": "empty"})
            continue
        if not frame.empty:
            keep = pd.to_datetime(frame["report_date"]).dt.strftime("%Y%m%d") != key
            frame = frame.loc[keep]
        frame = pd.concat([frame, fresh], ignore_index=True)
        frame = frame.sort_values(["code", "announcement_date", "report_date"]).reset_index(drop=True)
        save_frame_atomic(cache_path, frame)
        refreshed.append({"report_date": key, "rows": int(len(fresh))})
        log(f"fundamentals: cached {key} rows={len(fresh)}")
    return {"requested": requested, "refreshed": refreshed, "failures": failures}
