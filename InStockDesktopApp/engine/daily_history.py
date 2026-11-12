"""Point-in-time daily bars for strategy factors.

The InStock update job keeps a forward-adjusted history for the whole market,
but it rolls the file forward each day by appending the raw spot bar.  After an
ex-rights date the appended raw prices no longer line up with the adjusted
history, and on days the update did not run a stock silently misses sessions.
This module re-adjusts those jumps using each row's own change amount and
reports missing sessions against an independent index calendar, so momentum,
drawdown and volatility factors are computed over the windows they claim.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INSTOCK_HIST_ROOT = Path(r"C:\InStockLocal\InStock\instock\cache\hist")
FALLBACK_BAR_CACHE = Path(r"C:\InStockLocal\backtest_cache\daily_bars")
CALENDAR_PATHS = (
    Path(r"C:\InStockLocal\backtest_cache\indices\000300.csv"),
    Path(r"C:\InStockLocal\backtest_cache\indices\000905.csv"),
)
BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]
EX_RIGHTS_TOLERANCE = 0.003


def latest_hist_dir(root: Path = INSTOCK_HIST_ROOT) -> Path | None:
    """Return the newest InStock day directory that holds history pickles."""
    root = Path(root)
    if not root.exists():
        return None
    days = sorted(
        (day for month in root.iterdir() if month.is_dir() for day in month.iterdir() if day.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for day in days:
        if any(day.glob("*qfq.gzip.pickle")):
            return day
    return None


def normalize_volume_units(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert per-row volume to shares when a source reported board lots.

    Sources disagree (lots vs shares) and caches mix them.  ``amount`` is in
    yuan everywhere, so amount / (volume * close) is ~1 for shares and ~100 for
    lots; rows near 100 are rescaled.
    """
    data = frame.copy()
    volume = pd.to_numeric(data["volume"], errors="coerce")
    ratio = pd.to_numeric(data["amount"], errors="coerce") / (
        volume * pd.to_numeric(data["close"], errors="coerce")
    )
    lots = ratio.between(30.0, 300.0)
    data.loc[lots, "volume"] = volume[lots] * 100.0
    return data


def trading_calendar(paths: Iterable[Path] = CALENDAR_PATHS) -> pd.DatetimeIndex:
    """Exchange sessions from locally cached index bars (empty when unavailable)."""
    dates: set[pd.Timestamp] = set()
    for path in paths:
        try:
            frame = pd.read_csv(path, usecols=["date"])
        except (OSError, ValueError):
            continue
        dates.update(pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize())
    return _warm(pd.DatetimeIndex(sorted(dates)))


def _warm(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # pandas builds the lookup table lazily and not thread-safely; build it
    # before the calendar is shared with worker threads.
    calendar.get_indexer(calendar[:1])
    return calendar


def extend_calendar(calendar: pd.DatetimeIndex, observed: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    """Append sessions observed after the cached calendar ends (stale index file)."""
    end = calendar.max() if len(calendar) else pd.Timestamp.min
    later = {pd.Timestamp(value).normalize() for value in observed if pd.Timestamp(value) > end}
    return _warm(calendar.append(pd.DatetimeIndex(sorted(later))) if later else calendar)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "amount", "ups_downs"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "open", "close"])
    data = data[(data["open"] > 0) & (data["close"] > 0)]
    return data.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _previous_session_map(calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    return {calendar[i]: calendar[i - 1] for i in range(1, len(calendar))}


def adjust_ex_rights(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, int]:
    """Forward-adjust history across ex-rights jumps detected from ``ups_downs``.

    The exchange pre-close (close - change amount) differs from the stored
    previous close only on ex-rights days or when sessions are missing.  Only
    contiguous days are adjusted; a jump across a missing session is left for
    the gap check to reject.
    """
    if "ups_downs" not in frame or frame.empty or not len(calendar):
        return frame, 0
    data = frame.copy()
    positions = pd.Series(calendar.get_indexer(data["date"]), index=data.index)
    contiguous = (positions >= 0) & (positions.diff() == 1)
    pre_close = data["close"] - data["ups_downs"]
    ratio = pre_close / data["close"].shift()
    # Ratios above 1 occur where raw bars follow an already-adjusted segment.
    mask = (
        contiguous
        & ratio.between(0.3, 3.0)
        & ((ratio - 1.0).abs() > EX_RIGHTS_TOLERANCE)
    )
    factor = ratio.where(mask, 1.0)
    events = int(mask.sum())
    if events:
        # Every bar before an ex-rights day is scaled by that day's factor.
        cumulative = factor[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
        for column in ("open", "high", "low", "close"):
            data[column] = data[column] * cumulative
    return data, events


def missing_sessions(dates: pd.Series, calendar: pd.DatetimeIndex, end: pd.Timestamp, sessions: int) -> int:
    """Count exchange sessions absent from ``dates`` in the last ``sessions`` up to ``end``."""
    if not len(calendar):
        return 0
    window = calendar[calendar <= pd.Timestamp(end).normalize()][-sessions:]
    present = set(pd.to_datetime(dates).dt.normalize())
    return int(sum(day not in present for day in window))


def _read_instock(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_pickle(path, compression="gzip")
    except Exception:
        return None


def _read_fallback(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def default_repair_dir() -> Path:
    # Shared by sandboxed historical re-runs, so not tied to INSTOCK_DATA_DIR.
    return Path(__file__).resolve().parents[1] / "data" / "cache" / "history_repair"


def _read_repair(code: str, repair_dir: Path | None) -> pd.DataFrame | None:
    path = Path(repair_dir or default_repair_dir()) / f"{code}.csv"
    if not path.exists():
        return None
    try:
        frame = _prepare(pd.read_csv(path))
    except Exception:
        return None
    if frame.empty:
        return None
    # A single consistent download: day-over-day change is the true change.
    frame["ups_downs"] = frame["close"].diff()
    return frame


def _splice_repair(repaired: pd.DataFrame, instock: pd.DataFrame | None) -> pd.DataFrame:
    """Repaired history, then InStock bars newer than it (raw, carrying ``ups_downs``)."""
    if instock is None or instock.empty:
        return repaired
    newer = _prepare(instock)
    newer = newer[newer["date"] > repaired["date"].max()]
    return pd.concat([repaired, newer], ignore_index=True)


def codes_needing_repair(
    hist_dir: Path,
    calendar: pd.DatetimeIndex,
    end: pd.Timestamp,
    *,
    sessions: int = 130,
    repair_dir: Path | None = None,
) -> list[str]:
    """Codes whose InStock history misses sessions in the recent window."""
    result = []
    for path in sorted(Path(hist_dir).glob("*qfq.gzip.pickle")):
        code = path.name[:6]
        bars, _meta = load_code_history(code, end, calendar=calendar, hist_dir=hist_dir, repair_dir=repair_dir)
        if len(bars) < sessions:
            continue
        if missing_sessions(bars["date"], calendar, bars["date"].max(), sessions):
            result.append(code)
    return result


def repair_histories(
    codes: Iterable[str],
    end: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    *,
    fetcher,
    repair_dir: Path | None = None,
    lookback_days: int = 420,
    max_workers: int = 4,
    max_consecutive_failures: int = 20,
    progress=None,
) -> dict:
    """Download one consistent adjusted history per code into the repair folder.

    InStock's own cache is left untouched.  A download is kept only when it
    covers every session in its range, so a repair never adds new gaps.
    """
    from concurrent.futures import ThreadPoolExecutor

    target = Path(repair_dir or default_repair_dir())
    target.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(end).normalize()
    start = (end - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    codes = list(codes)
    outcome = {"requested": len(codes), "repaired": 0, "incomplete": [], "failed": []}

    def one(code):
        try:
            bars = _prepare(fetcher(code, start, end.strftime("%Y-%m-%d")))
        except Exception as exc:
            return code, "failed", str(exc)[:200]
        if bars.empty:
            return code, "failed", "empty"
        window = calendar[(calendar >= bars["date"].min()) & (calendar <= bars["date"].max())]
        if missing_sessions(bars["date"], window, bars["date"].max(), len(window)):
            return code, "incomplete", ""
        bars = normalize_volume_units(bars)
        temporary = target / f"{code}.csv.tmp"
        bars.loc[:, BAR_COLUMNS].to_csv(temporary, index=False, date_format="%Y-%m-%d")
        os.replace(temporary, target / f"{code}.csv")
        return code, "repaired", ""

    consecutive_failures = 0
    done = 0
    chunk = max(1, max_workers * 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for offset in range(0, len(codes), chunk):
            for code, status, detail in executor.map(one, codes[offset : offset + chunk]):
                done += 1
                if status == "repaired":
                    outcome["repaired"] += 1
                    consecutive_failures = 0
                elif status == "incomplete":
                    outcome["incomplete"].append(code)
                    consecutive_failures = 0
                else:
                    outcome["failed"].append({"code": code, "error": detail})
                    consecutive_failures += 1
                if progress and done % 50 == 0:
                    progress(done, len(codes))
            if consecutive_failures >= max_consecutive_failures:
                # The source is throttling us; stop and let the next update continue.
                outcome["stopped_early"] = True
                break
    return outcome


def load_code_history(
    code: str,
    end: pd.Timestamp | str,
    *,
    calendar: pd.DatetimeIndex,
    hist_dir: Path | None,
    fallback_dir: Path = FALLBACK_BAR_CACHE,
    repair_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Bars up to ``end`` for one code plus quality metadata.

    Prefers the full-market InStock history (with any repaired history laid
    over it); falls back to the backtest cache.
    """
    code = str(code).zfill(6)
    cutoff = pd.Timestamp(end).normalize()
    raw = None
    source = "missing"
    if hist_dir is not None:
        raw = _read_instock(Path(hist_dir) / f"{code}qfq.gzip.pickle")
        source = "instock_hist" if raw is not None else source
        repaired = _read_repair(code, repair_dir)
        if repaired is not None:
            raw = _splice_repair(repaired, raw)
            source = "repaired"
    if raw is None:
        raw = _read_fallback(Path(fallback_dir) / f"{code}.csv")
        source = "backtest_cache" if raw is not None else source
    if raw is None or raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS), {"source": source, "ex_rights_events": 0}
    data = _prepare(raw)
    data = data[data["date"] <= cutoff]
    data, events = adjust_ex_rights(data, calendar)
    if "amount" not in data:
        data["amount"] = data["close"] * data["volume"]
    data = normalize_volume_units(data)
    return data.loc[:, BAR_COLUMNS].reset_index(drop=True), {
        "source": source,
        "ex_rights_events": events,
    }


def append_spot_bar(bars: pd.DataFrame, row: dict, selection_date: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Append the signal-day spot bar when history stops one session earlier.

    The spot pre-close tells us whether the history needs an ex-rights
    adjustment before the raw spot price can be appended.
    """
    selection_date = pd.Timestamp(selection_date).normalize()
    if not bars.empty and bars["date"].max() >= selection_date:
        return bars

    def number(key, default=None):
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    price = number("new_price")
    volume = number("volume")
    if not price or not volume:
        return bars
    data = bars.copy()
    pre_close = number("pre_close_price")
    previous = _previous_session_map(calendar).get(selection_date)
    if (
        not data.empty
        and pre_close
        and previous is not None
        and data["date"].iloc[-1] == previous
    ):
        ratio = pre_close / float(data["close"].iloc[-1])
        if abs(ratio - 1.0) > EX_RIGHTS_TOLERANCE and 0.3 < ratio < 3.0:
            for column in ("open", "high", "low", "close"):
                data[column] = data[column] * ratio
    spot = pd.DataFrame(
        [
            {
                "date": selection_date,
                "open": number("open_price", pre_close or price),
                "high": number("high_price", price),
                "low": number("low_price", price),
                "close": price,
                "volume": volume,
                "amount": number("deal_amount", price * volume),
            }
        ]
    )
    return pd.concat([data, spot], ignore_index=True)


RECENT_SESSIONS = 20
MAX_MISSING_SESSIONS = 6


def history_window_is_complete(
    bars: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    end: pd.Timestamp,
    sessions: int = 61,
    *,
    max_missing: int = MAX_MISSING_SESSIONS,
) -> bool:
    """True when the factor window is usable once aligned to the calendar.

    The last 20 sessions must all be present (short-horizon factors and the
    rebound shape read them directly); older sessions may miss a few days,
    which ``align_to_calendar`` carries forward.
    """
    if bars.empty or pd.Timestamp(bars["date"].max()).normalize() != pd.Timestamp(end).normalize():
        return False
    if len(bars) < sessions - max_missing:
        return False
    if missing_sessions(bars["date"], calendar, end, RECENT_SESSIONS):
        return False
    return missing_sessions(bars["date"], calendar, end, sessions) <= max_missing


def align_to_calendar(bars: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Insert missing sessions so rolling windows span the sessions they claim.

    A missing day carries the previous close with no traded volume; without
    this, ``iloc[-61]`` silently reaches further back than 60 sessions.
    """
    if bars.empty or not len(calendar):
        return bars
    start, end = bars["date"].min(), bars["date"].max()
    sessions = calendar[(calendar >= start) & (calendar <= end)]
    aligned = bars.set_index("date").reindex(sessions.union(pd.DatetimeIndex(bars["date"])))
    aligned.index.name = "date"
    missing = aligned["close"].isna()
    aligned["close"] = aligned["close"].ffill()
    for column in ("open", "high", "low"):
        aligned.loc[missing, column] = aligned.loc[missing, "close"]
    aligned.loc[missing, ["volume", "amount"]] = np.nan
    return aligned.reset_index()


def finite_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")
