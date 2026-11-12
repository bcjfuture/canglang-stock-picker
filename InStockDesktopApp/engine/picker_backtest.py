# -*- coding: utf-8 -*-
"""Leakage-safe price/volume backtest helpers for the desktop pickers."""

from __future__ import annotations

import json
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from market_policy import (
    MarketPolicy,
    classify_regime,
    neighbor_stability,
    participation_checks,
    pick_count_for_regime,
)
from trade_policy import TradePolicy, simulate_trade


FACTOR_COLUMNS = (
    "mom_5",
    "mom_20",
    "mom_60",
    "ma_gap_5_20",
    "volume_ratio_20",
    "volatility_20",
    "drawdown_60",
    "distance_high_60",
    "liquidity_20",
    "trend_20_60",
    "overheat",
    "rebound_setup",
    "ma_20",
    "atr_14",
    "atr_pct_14",
)

BAR_COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount")
TECH_KEYWORDS = (
    "半导体", "芯片", "人工智能", "AI", "算力", "服务器", "数据中心", "云计算", "软件", "信创",
    "网络安全", "通信", "5G", "6G", "光模块", "光通信", "CPO", "PCB", "电子", "机器人",
    "机器视觉", "工业母机", "智能制造", "智能驾驶", "无人驾驶", "车联网", "物联网", "鸿蒙",
    "华为", "英伟达", "低空经济", "商业航天",
)
_TRADE_SIM_CACHE: dict[tuple, tuple[dict | None, dict | None, dict | None, dict | None]] = {}


def _cached_trade_caps(code, signal_date, bars, signal_pos, atr, policy, cost):
    key = (
        str(code), str(signal_date)[:10], round(float(atr), 8),
        policy.holding_days, policy.atr_multiplier, policy.reward_risk, float(cost),
    )
    cached = _TRADE_SIM_CACHE.get(key)
    if cached is not None:
        return cached
    value = (
        simulate_trade(bars, signal_pos, atr, policy, cost=cost),
        simulate_trade(bars, signal_pos, atr, policy, cost=cost, hard_cap=5),
        simulate_trade(bars, signal_pos, atr, policy, cost=cost, hard_cap=10),
        simulate_trade(bars, signal_pos, atr, policy, cost=cost, hard_cap=20),
    )
    _TRADE_SIM_CACHE[key] = value
    return value


def _market_code(code: str) -> tuple[str, str]:
    clean = str(code).strip().zfill(6)
    is_sh = clean.startswith(("5", "6", "9"))
    return ("1" if is_sh else "0") + "." + clean, ("sh" if is_sh else "sz") + clean


def _normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    data = frame.rename(columns=aliases).copy()
    for column in BAR_COLUMNS:
        if column not in data:
            data[column] = np.nan
    data = data.loc[:, BAR_COLUMNS]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in BAR_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "open", "close"])
    data = data[(data["open"] > 0) & (data["close"] > 0)]
    return data.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def validate_daily_bars(frame: pd.DataFrame) -> None:
    """Reject malformed daily bars before normalization can hide source defects."""
    aliases = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close"}
    data = frame.rename(columns=aliases).copy()
    required = ("date", "open", "high", "low", "close")
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"daily bars missing columns: {','.join(missing)}")
    dates = pd.to_datetime(data["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("daily bar dates must be valid, unique and ascending")
    numeric = data.loc[:, required[1:]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or (numeric <= 0).any().any():
        raise ValueError("daily OHLC values must be finite and positive")
    if (numeric["high"] < numeric[["open", "close"]].max(axis=1)).any():
        raise ValueError("daily high is below open or close")
    if (numeric["low"] > numeric[["open", "close"]].min(axis=1)).any():
        raise ValueError("daily low is above open or close")


def _fetch_eastmoney(code: str, start: str, end: str) -> pd.DataFrame:
    secid, _ = _market_code(code)
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
    }
    session = requests.Session()
    session.trust_env = False
    response = session.get(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        params=params,
        timeout=18,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
    )
    response.raise_for_status()
    rows = ((response.json().get("data") or {}).get("klines") or [])
    parsed = []
    for row in rows:
        fields = row.split(",")
        if len(fields) >= 7:
            parsed.append(fields[:7])
    return pd.DataFrame(parsed, columns=BAR_COLUMNS)


def _fetch_akshare(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    _, symbol = _market_code(code)
    return ak.stock_zh_a_daily(
        symbol=symbol,
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust="qfq",
    )


def _fetch_tencent(code: str, start: str, end: str, session=None) -> pd.DataFrame:
    _, symbol = _market_code(code)
    session = session or requests.Session()
    session.trust_env = False
    parsed = []
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    for year in range(start_ts.year, end_ts.year + 1):
        chunk_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1)).strftime("%Y-%m-%d")
        chunk_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31)).strftime("%Y-%m-%d")
        response = session.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"{symbol},day,{chunk_start},{chunk_end},640,qfq"},
            timeout=18,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        node = ((response.json().get("data") or {}).get(symbol) or {})
        rows = node.get("qfqday") or node.get("day") or []
        for row in rows:
            if len(row) >= 6:
                day, open_, close, high, low, volume = row[:6]
                try:
                    amount = float(close) * float(volume) * 100.0
                except (TypeError, ValueError):
                    amount = np.nan
                parsed.append([day, open_, high, low, close, volume, amount])
    return pd.DataFrame(parsed, columns=BAR_COLUMNS)


def _network_fetch(code: str, start: str, end: str) -> pd.DataFrame:
    errors = []
    for fetcher in (_fetch_akshare, _fetch_tencent, _fetch_eastmoney):
        for attempt in range(3):
            try:
                result = _normalize_bars(fetcher(code, start, end))
                if not result.empty:
                    return result
            except Exception as exc:  # source failover is deliberate
                errors.append(f"{fetcher.__name__}: {exc}")
                time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"No historical bars for {code}: {' | '.join(errors[-4:])}")


def load_local_spot_bars(code: str, start: str, end: str) -> pd.DataFrame:
    """Read exact-date OHLC rows already committed by the local daily update."""
    import pymysql

    connection = pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root",
        database="instockdb",
        connect_timeout=2,
        read_timeout=4,
        write_timeout=2,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT `date`, `open_price`, `high_price`, `low_price`,
                       `new_price`, `volume`, `deal_amount`
                FROM `cn_stock_spot`
                WHERE `code`=%s AND `date` BETWEEN %s AND %s
                ORDER BY `date`
                """,
                (str(code).zfill(6), start, end),
            )
            rows = cursor.fetchall()
    finally:
        connection.close()
    frame = pd.DataFrame(rows, columns=BAR_COLUMNS)
    clean = _normalize_bars(frame)
    if clean.empty:
        return clean
    valid = (
        (clean["high"] >= clean[["open", "close"]].max(axis=1))
        & (clean["low"] <= clean[["open", "close"]].min(axis=1))
        & (clean["high"] >= clean["low"])
    )
    return clean.loc[valid].reset_index(drop=True)


def _merge_bars(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return _normalize_bars(pd.concat(usable, ignore_index=True))


@contextmanager
def direct_network_environment():
    """Temporarily prevent local proxy inheritance during bulk market downloads."""
    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    original = {name: os.environ.get(name) for name in names}
    original_no_proxy = os.environ.get("NO_PROXY")
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ["NO_PROXY"] = "*"
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if original_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = original_no_proxy


def fetch_daily_bars(
    code: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    fetcher=None,
    snapshot_loader=None,
) -> pd.DataFrame:
    """Load adjusted bars from a per-symbol cache, fetching atomically when needed."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{str(code).zfill(6)}.csv"
    requested_end = pd.Timestamp(end)
    cached = pd.DataFrame(columns=BAR_COLUMNS)
    if cache_path.exists():
        cached = _normalize_bars(pd.read_csv(cache_path))
        if not cached.empty and cached["date"].max() >= requested_end:
            return cached[(cached["date"] >= pd.Timestamp(start)) & (cached["date"] <= requested_end)].reset_index(drop=True)

    if snapshot_loader is None and fetcher is None:
        snapshot_loader = load_local_spot_bars
    local = pd.DataFrame(columns=BAR_COLUMNS)
    if snapshot_loader is not None:
        try:
            local = _normalize_bars(snapshot_loader(str(code).zfill(6), start, end))
        except Exception:
            local = pd.DataFrame(columns=BAR_COLUMNS)

    combined_cache = _merge_bars(cached, local)
    if not combined_cache.empty and combined_cache["date"].max() >= requested_end:
        temp_path = cache_path.with_suffix(".csv.tmp")
        combined_cache.to_csv(temp_path, index=False, date_format="%Y-%m-%d")
        os.replace(temp_path, cache_path)
        return combined_cache[
            (combined_cache["date"] >= pd.Timestamp(start))
            & (combined_cache["date"] <= requested_end)
        ].reset_index(drop=True)

    loader = fetcher or _network_fetch
    data = _merge_bars(loader(str(code).zfill(6), start, end), local)
    if data.empty:
        raise RuntimeError(f"Historical source returned no valid bars for {code}")
    temp_path = cache_path.with_suffix(".csv.tmp")
    data.to_csv(temp_path, index=False, date_format="%Y-%m-%d")
    os.replace(temp_path, cache_path)
    return data[(data["date"] >= pd.Timestamp(start)) & (data["date"] <= requested_end)].reset_index(drop=True)


def load_universe_from_db(limit: int | None = None) -> pd.DataFrame:
    """Read the latest non-ST liquid A-share universe from the local InStock DB."""
    import pymysql

    conn = pymysql.connect(
        host="127.0.0.1",
        user="root",
        password="root",
        database="instockdb",
        port=3306,
        charset="utf8mb4",
    )
    sql = """
        SELECT code, name, industry, concept, deal_amount
        FROM cn_stock_selection
        WHERE date=(SELECT MAX(date) FROM cn_stock_selection)
          AND name NOT LIKE '%%ST%%'
          AND name NOT LIKE '%%退%%'
          AND new_price > 0
        ORDER BY deal_amount DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [item[0] for item in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


def forward_return(
    bars: pd.DataFrame,
    signal_pos: int,
    hold: int = 10,
    cost: float = 0.003,
) -> float | None:
    """Return next-open to hold-session close net return."""
    entry_pos = signal_pos + 1
    exit_pos = entry_pos + hold
    if signal_pos < 0 or hold < 1 or exit_pos >= len(bars):
        return None
    entry = float(bars.iloc[entry_pos]["open"])
    exit_price = float(bars.iloc[exit_pos]["close"])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_price) or exit_price <= 0:
        return None
    return float(exit_price / entry - 1.0 - cost)


def calculate_forward_returns(
    bars: pd.DataFrame,
    hold: int = 10,
    cost: float = 0.003,
) -> pd.Series:
    """Vectorized equivalent of ``forward_return`` for every signal row."""
    if hold < 1:
        raise ValueError("hold must be positive")
    entry = pd.to_numeric(bars["open"], errors="coerce").shift(-1)
    exit_price = pd.to_numeric(bars["close"], errors="coerce").shift(-(hold + 1))
    result = exit_price / entry - 1.0 - float(cost)
    result = result.mask(~np.isfinite(entry) | (entry <= 0) | ~np.isfinite(exit_price) | (exit_price <= 0))
    return result.astype("float64")


def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from sequential period returns."""
    values = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if values.empty:
        return 0.0
    wealth = (1.0 + values).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def compute_factors(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing-only factors; row t never observes row t+1."""
    data = bars.copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    amount = pd.to_numeric(data.get("amount", close * volume), errors="coerce")
    daily_return = close.pct_change(fill_method=None)
    high_60 = close.rolling(60, min_periods=20).max()
    ma_5 = close.rolling(5, min_periods=5).mean()
    ma_20 = close.rolling(20, min_periods=20).mean()
    ma_60 = close.rolling(60, min_periods=40).mean()
    volume_ma_20 = volume.rolling(20, min_periods=10).mean()

    out = pd.DataFrame(index=data.index)
    out["mom_5"] = close.pct_change(5, fill_method=None)
    out["mom_20"] = close.pct_change(20, fill_method=None)
    out["mom_60"] = close.pct_change(60, fill_method=None)
    out["ma_gap_5_20"] = ma_5 / ma_20 - 1.0
    out["volume_ratio_20"] = volume / volume_ma_20
    out["volatility_20"] = daily_return.rolling(20, min_periods=15).std()
    out["drawdown_60"] = close / close.rolling(60, min_periods=20).max() - 1.0
    out["distance_high_60"] = close / high_60 - 1.0
    out["liquidity_20"] = np.log1p(amount.rolling(20, min_periods=10).mean())
    out["trend_20_60"] = ma_20 / ma_60 - 1.0
    out["overheat"] = (out["mom_5"] - 0.08).clip(lower=0.0) + 0.05 * (
        out["volume_ratio_20"] - 2.5
    ).clip(lower=0.0)
    out["rebound_setup"] = (
        -(out["mom_5"] - 0.02).abs()
        + out["trend_20_60"].clip(lower=-0.2, upper=0.2)
        - out["overheat"]
    )
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    out["atr_pct_14"] = out["atr_14"] / close.replace(0, np.nan)
    out["ma_20"] = ma_20
    return out.loc[:, FACTOR_COLUMNS]


def build_symbol_panel(
    code: str,
    bars: pd.DataFrame,
    metadata: dict | None = None,
    cost: float = 0.003,
) -> pd.DataFrame:
    """Build a stock-date factor panel with separately calculated future labels."""
    validate_daily_bars(bars)
    clean = _normalize_bars(bars)
    factors = compute_factors(clean)
    panel = factors.copy()
    panel.insert(0, "date", clean["date"].to_numpy())
    panel.insert(1, "code", str(code).zfill(6))
    for column in BAR_COLUMNS[1:]:
        panel[column] = clean[column].to_numpy()
    for key, value in (metadata or {}).items():
        if key not in panel:
            panel[key] = value
    for hold in (5, 10, 20):
        panel[f"forward_{hold}"] = calculate_forward_returns(clean, hold=hold, cost=cost).to_numpy()
    return panel


def filter_profile_universe(panel: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Apply the same broad technology direction gate used by the desktop profile."""
    if universe != "tech" or panel.empty:
        return panel.copy()
    industry = panel.get("industry", pd.Series("", index=panel.index)).fillna("").astype(str)
    concept = panel.get("concept", pd.Series("", index=panel.index)).fillna("").astype(str)
    haystack = industry + " " + concept
    mask = pd.Series(False, index=panel.index)
    upper = haystack.str.upper()
    for keyword in TECH_KEYWORDS:
        mask |= upper.str.contains(keyword.upper(), regex=False)
    return panel[mask].copy()


def replacement_passes(
    baseline: dict,
    candidate: dict,
    fold_improvements: list[bool],
    drawdown_tolerance: float = 0.15,
) -> bool:
    """Conservative gate for allowing optimized parameters into live selection."""
    if not fold_improvements or sum(bool(v) for v in fold_improvements) <= len(fold_improvements) / 2:
        return False
    if float(candidate.get("mean_return_10", float("-inf"))) <= float(baseline.get("mean_return_10", 0.0)):
        return False
    base_dd = abs(float(baseline.get("max_drawdown_10", 0.0)))
    candidate_dd = abs(float(candidate.get("max_drawdown_10", 0.0)))
    allowed_dd = max(0.02, base_dd * (1.0 + drawdown_tolerance))
    if candidate_dd > allowed_dd:
        return False
    worse_5 = float(candidate.get("mean_return_5", 0.0)) < float(baseline.get("mean_return_5", 0.0))
    worse_20 = float(candidate.get("mean_return_20", 0.0)) < float(baseline.get("mean_return_20", 0.0))
    return not (worse_5 and worse_20)


def select_profile_candidate(
    panel: pd.DataFrame,
    candidates: list[dict],
    folds: list[dict],
    baseline_name: str,
    top_n: int = 20,
    rebalance_every: int = 5,
) -> dict:
    """Select the most repeatedly training-chosen candidate and validate it."""
    by_name = {candidate["name"]: candidate for candidate in candidates}
    if baseline_name not in by_name:
        raise ValueError(f"baseline {baseline_name!r} is not in candidates")
    walk = walk_forward_optimize(panel, candidates, folds, top_n, rebalance_every)
    chosen_names = [fold["chosen_profile"] for fold in walk["folds"]]
    if not chosen_names:
        raise ValueError("at least one fold is required")
    counts = Counter(chosen_names)
    last_position = {name: max(i for i, value in enumerate(chosen_names) if value == name) for name in counts}
    chosen_name = max(counts, key=lambda name: (counts[name], last_position[name]))
    chosen = by_name[chosen_name]
    baseline = by_name[baseline_name]

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    valid_mask = pd.Series(False, index=data.index)
    for fold in folds:
        valid_mask |= data["date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
    validation_panel = data[valid_mask]
    baseline_metrics = evaluate_profile(validation_panel, baseline, rebalance_every, top_n)
    candidate_metrics = evaluate_profile(validation_panel, chosen, rebalance_every, top_n)

    improvements = []
    for fold in folds:
        part = data[data["date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))]
        base_fold = evaluate_profile(part, baseline, rebalance_every, top_n)
        chosen_fold = evaluate_profile(part, chosen, rebalance_every, top_n)
        improvements.append(chosen_fold["mean_return_10"] > base_fold["mean_return_10"])
    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "accepted": replacement_passes(baseline_metrics, candidate_metrics, improvements),
        "fold_improvements": improvements,
        "walk_forward": walk,
    }


def default_folds() -> list[dict]:
    return [
        {"train_end": "2023-12-31", "valid_start": "2024-01-01", "valid_end": "2024-06-30"},
        {"train_end": "2024-06-30", "valid_start": "2024-07-01", "valid_end": "2024-12-31"},
        {"train_end": "2024-12-31", "valid_start": "2025-01-01", "valid_end": "2025-06-30"},
        {"train_end": "2025-06-30", "valid_start": "2025-07-01", "valid_end": "2025-12-31"},
        {"train_end": "2025-12-31", "valid_start": "2026-01-01", "valid_end": "2026-06-30"},
    ]


def build_market_panel(
    universe: pd.DataFrame,
    start: str,
    end: str,
    cache_dir: str | Path,
    workers: int = 16,
    cost: float = 0.003,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch symbols concurrently and assemble one reproducible panel."""
    records = universe.to_dict("records")
    panels = []
    errors = []

    def load(record):
        code = str(record["code"]).zfill(6)
        frame = fetch_daily_bars(code, start, end, cache_dir)
        if len(frame) < 500:
            raise RuntimeError(f"only {len(frame)} valid rows")
        metadata = {
            "name": record.get("name", ""),
            "industry": record.get("industry", ""),
            "concept": record.get("concept", ""),
        }
        return build_symbol_panel(code, frame, metadata, cost=cost)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_map = {executor.submit(load, record): record for record in records}
        total = len(future_map)
        completed = 0
        for future in as_completed(future_map):
            record = future_map[future]
            completed += 1
            try:
                panels.append(future.result())
            except Exception as exc:
                errors.append({"code": str(record.get("code", "")), "error": str(exc)[:300]})
            if completed % 25 == 0 or completed == total:
                print(f"Historical cache: {completed}/{total}, usable={len(panels)}, failed={len(errors)}", flush=True)
    if not panels:
        raise RuntimeError("No symbols produced a valid historical panel")
    panel = pd.concat(panels, ignore_index=True)
    panel["above_ma20"] = (panel["close"] > panel["ma_20"]).astype(float)
    grouped = panel.groupby("date")
    panel["breadth_above_ma20"] = grouped["above_ma20"].transform("mean")
    panel["market_median_mom20"] = grouped["mom_20"].transform("median")
    panel["market_median_drawdown60"] = grouped["drawdown_60"].transform("median")
    return panel, errors


def _compact_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"selections", "period_returns_10"}
    }


def run_optimization(
    start: str,
    end: str,
    universe_limit: int,
    workers: int,
    cache_dir: Path,
    profiles_path: Path,
    output_json: Path,
    output_csv: Path,
    cost: float = 0.003,
    top_n: int = 20,
    search_space_path: Path | None = None,
    panel_cache_path: Path | None = None,
) -> dict:
    universe = load_universe_from_db(limit=universe_limit)
    panel = None
    fetch_errors = []
    if panel_cache_path is not None and panel_cache_path.exists():
        cached = pd.read_pickle(panel_cache_path)
        if (
            cached.get("start") == start and cached.get("end") == end
            and int(cached.get("universe_limit", 0)) == int(universe_limit)
        ):
            panel = cached["panel"]
            fetch_errors = cached.get("fetch_errors", [])
            print(f"Loaded policy panel cache: {panel_cache_path}", flush=True)
    if panel is None:
        with direct_network_environment():
            panel, fetch_errors = build_market_panel(universe, start, end, cache_dir, workers, cost)
        if panel_cache_path is not None:
            panel_cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(
                {
                    "start": start, "end": end, "universe_limit": universe_limit,
                    "panel": panel, "fetch_errors": fetch_errors,
                },
                panel_cache_path,
            )
            print(f"Saved policy panel cache: {panel_cache_path}", flush=True)
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    folds = default_folds()
    if search_space_path is not None:
        search_space = json.loads(search_space_path.read_text(encoding="utf-8"))
        return run_policy_optimization(
            panel=panel,
            profiles=profiles,
            folds=folds,
            search_space=search_space,
            start=start,
            end=end,
            universe=universe,
            fetch_errors=fetch_errors,
            output_json=output_json,
            output_csv=output_csv,
            cost=cost,
        )
    results = {}
    summary_rows = []
    for profile_name, spec in profiles.items():
        scoped = filter_profile_universe(panel, spec.get("universe", "all"))
        scoped = scoped.dropna(subset=["forward_10"])
        evidence = select_profile_candidate(
            scoped,
            spec["candidates"],
            folds,
            baseline_name=spec["baseline"],
            top_n=top_n,
            rebalance_every=5,
        )
        chosen_name = evidence["candidate"]["profile"]
        chosen_spec = next(item for item in spec["candidates"] if item["name"] == chosen_name)
        evidence["chosen_weights"] = chosen_spec["weights"]
        evidence["baseline_weights"] = next(
            item["weights"] for item in spec["candidates"] if item["name"] == spec["baseline"]
        )
        results[profile_name] = evidence
        summary_rows.append(
            {
                "profile": profile_name,
                "baseline": evidence["baseline"]["profile"],
                "candidate": chosen_name,
                "accepted": evidence["accepted"],
                "baseline_mean_10": evidence["baseline"]["mean_return_10"],
                "candidate_mean_10": evidence["candidate"]["mean_return_10"],
                "baseline_max_drawdown_10": evidence["baseline"]["max_drawdown_10"],
                "candidate_max_drawdown_10": evidence["candidate"]["max_drawdown_10"],
                "improved_folds": sum(evidence["fold_improvements"]),
                "total_folds": len(evidence["fold_improvements"]),
            }
        )
        print(
            f"{profile_name}: {evidence['baseline']['mean_return_10']:.4%} -> "
            f"{evidence['candidate']['mean_return_10']:.4%}, accepted={evidence['accepted']}",
            flush=True,
        )

    dates = pd.to_datetime(panel["date"])
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": {
            "signal": "close",
            "entry": "next_open",
            "holding_period_primary": 10,
            "holding_period_checks": [5, 20],
            "round_trip_cost": cost,
            "rebalance_every": 5,
            "top_n": top_n,
            "walk_forward_folds": folds,
        },
        "coverage": {
            "requested_start": start,
            "requested_end": end,
            "actual_start": dates.min().strftime("%Y-%m-%d"),
            "actual_end": dates.max().strftime("%Y-%m-%d"),
            "requested_symbols": int(len(universe)),
            "usable_symbols": int(panel["code"].nunique()),
            "stock_dates": int(len(panel)),
            "fetch_failures": fetch_errors,
        },
        "profiles": results,
        "caveats": [
            "The universe is based on currently available InStock records and therefore has survivorship bias.",
            "Historical fundamentals and fund flow were not backfilled with current values.",
            "Backtest results are evidence about historical behavior, not a guarantee of future returns.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temp = output_json.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, output_json)
    pd.DataFrame(summary_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward optimization for InStock desktop pickers")
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize")
    optimize.add_argument("--start", default="2022-01-01")
    optimize.add_argument("--end", default=date.today().isoformat())
    optimize.add_argument("--universe-limit", type=int, default=600)
    optimize.add_argument("--workers", type=int, default=16)
    optimize.add_argument("--cost", type=float, default=0.003)
    optimize.add_argument("--top-n", type=int, default=20)
    optimize.add_argument("--cache-dir", type=Path, default=Path(r"C:\InStockLocal\backtest_cache\daily_bars"))
    optimize.add_argument("--profiles", type=Path, default=Path(__file__).with_name("picker_factor_profiles.json"))
    optimize.add_argument("--search-space", type=Path)
    optimize.add_argument("--panel-cache", type=Path)
    optimize.add_argument("--output", type=Path, default=Path(r"C:\InStockLocal\reports\picker_backtest_results.json"))
    optimize.add_argument("--summary", type=Path, default=Path(r"C:\InStockLocal\reports\picker_backtest_summary.csv"))
    args = parser.parse_args(argv)
    if args.command == "optimize":
        run_optimization(
            args.start,
            args.end,
            args.universe_limit,
            args.workers,
            args.cache_dir,
            args.profiles,
            args.output,
            args.summary,
            args.cost,
            args.top_n,
            args.search_space,
            args.panel_cache,
        )
        return 0
    return 2


def score_factors(panel: pd.DataFrame, profile: dict) -> pd.Series:
    """Score a panel using within-date percentile ranks."""
    if "date" not in panel:
        raise ValueError("panel must contain date")
    score = pd.Series(0.0, index=panel.index, dtype="float64")
    for factor, weight in profile.get("weights", {}).items():
        if factor not in panel:
            continue
        numeric = pd.to_numeric(panel[factor], errors="coerce")
        ranked = numeric.groupby(panel["date"]).rank(pct=True, method="average")
        score = score.add(ranked.fillna(0.5) * float(weight), fill_value=0.0)
    return score


def _objective(metrics: dict) -> float:
    if not metrics.get("periods"):
        return float("-inf")
    return float(
        metrics["median_return_10"]
        + 0.02 * (metrics["win_rate_10"] - 0.5)
        - 0.35 * abs(metrics["max_drawdown_10"])
    )


def evaluate_profile(
    panel: pd.DataFrame,
    profile: dict,
    rebalance_every: int = 5,
    top_n: int = 20,
) -> dict:
    """Evaluate equal-weight selections on deterministic rebalance dates."""
    if panel.empty:
        return {
            "profile": profile.get("name", "unnamed"),
            "periods": 0,
            "observations": 0,
            "mean_return_5": 0.0,
            "mean_return_10": 0.0,
            "mean_return_20": 0.0,
            "median_return_10": 0.0,
            "win_rate_10": 0.0,
            "max_drawdown_10": 0.0,
            "worst_decile_10": 0.0,
            "period_returns_10": [],
        }
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["score"] = score_factors(data, profile)
    dates = sorted(data["date"].dropna().unique())[:: max(1, int(rebalance_every))]
    period_rows = []
    observations = 0
    for signal_date in dates:
        cross = data[data["date"] == signal_date].dropna(subset=["forward_10"])
        picks = cross.sort_values(["score", "code"], ascending=[False, True]).head(top_n)
        if picks.empty:
            continue
        observations += len(picks)
        period_rows.append(
            {
                "date": pd.Timestamp(signal_date),
                "return_5": float(pd.to_numeric(picks.get("forward_5"), errors="coerce").mean()),
                "return_10": float(pd.to_numeric(picks["forward_10"], errors="coerce").mean()),
                "return_20": float(pd.to_numeric(picks.get("forward_20"), errors="coerce").mean()),
                "codes": picks["code"].astype(str).tolist(),
            }
        )
    periods = pd.DataFrame(period_rows)
    if periods.empty:
        return evaluate_profile(pd.DataFrame(), profile, rebalance_every, top_n)
    returns_10 = periods["return_10"]
    return {
        "profile": profile.get("name", "unnamed"),
        "periods": int(len(periods)),
        "observations": int(observations),
        "mean_return_5": float(periods["return_5"].mean()),
        "mean_return_10": float(returns_10.mean()),
        "mean_return_20": float(periods["return_20"].mean()),
        "median_return_10": float(returns_10.median()),
        "win_rate_10": float((returns_10 > 0).mean()),
        "max_drawdown_10": max_drawdown(returns_10),
        "worst_decile_10": float(returns_10.quantile(0.1)),
        "period_returns_10": [float(value) for value in returns_10],
        "selection_dates": [row["date"].strftime("%Y-%m-%d") for row in period_rows],
        "selections": {row["date"].strftime("%Y-%m-%d"): row["codes"] for row in period_rows},
    }


def walk_forward_optimize(
    panel: pd.DataFrame,
    candidates: list[dict],
    folds: list[dict],
    top_n: int = 20,
    rebalance_every: int = 5,
) -> dict:
    """Choose on past rows and report only subsequent validation rows."""
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    outputs = []
    for fold in folds:
        train_end = pd.Timestamp(fold["train_end"])
        valid_start = pd.Timestamp(fold["valid_start"])
        valid_end = pd.Timestamp(fold["valid_end"])
        if not train_end < valid_start <= valid_end:
            raise ValueError("fold dates must be chronological")
        train = data[data["date"] <= train_end]
        valid = data[(data["date"] >= valid_start) & (data["date"] <= valid_end)]
        train_metrics = [evaluate_profile(train, c, rebalance_every, top_n) for c in candidates]
        best_index = max(range(len(candidates)), key=lambda idx: _objective(train_metrics[idx]))
        chosen = candidates[best_index]
        validation = evaluate_profile(valid, chosen, rebalance_every, top_n)
        outputs.append(
            {
                "train_end": train_end.strftime("%Y-%m-%d"),
                "train_max_date": train["date"].max().strftime("%Y-%m-%d") if not train.empty else None,
                "validation_min_date": valid["date"].min().strftime("%Y-%m-%d") if not valid.empty else None,
                "validation_max_date": valid["date"].max().strftime("%Y-%m-%d") if not valid.empty else None,
                "chosen_profile": chosen.get("name", f"candidate_{best_index}"),
                "train_metrics": train_metrics[best_index],
                "validation_metrics": validation,
            }
        )
    return {"folds": outputs}


def _empty_policy_metrics(profile: dict, trade_policy: TradePolicy, market_policy: MarketPolicy) -> dict:
    return {
        "annualized_return": 0.0,
        "mean_return_5": 0.0,
        "mean_return_10": 0.0,
        "mean_return_20": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "top_contributor_share": 1.0,
        "best_regime_share": 1.0,
        "periods": 0,
        "scheduled_dates": 0,
        "active_dates": 0,
        "activity_rate": 0.0,
        "fold_active_dates": [],
        "selection_dates": [],
        "selections": {},
        "trades": [],
        "policy": trade_policy.to_dict(),
        "market_policy": market_policy.to_dict(),
    }


def evaluate_policy_profile(
    panel: pd.DataFrame,
    profile: dict,
    trade_policy: TradePolicy,
    market_policy: MarketPolicy,
    folds: list[dict] | None = None,
    cost: float = 0.003,
    rebalance_every: int = 5,
) -> dict:
    """Evaluate sequential equal-weight cohorts with dynamic market participation.

    New signals are considered only on deterministic rebalance dates and only after
    the previous cohort's reserved holding window has ended. An early exit therefore
    becomes cash, not permission to create overlapping exposure.
    """
    if panel.empty:
        return _empty_policy_metrics(profile, trade_policy, market_policy)
    policy_columns = {
        "date", "code", "open", "high", "low", "close", "volume", "amount",
        "atr_14", "breadth_above_ma20", "market_median_mom20",
        "market_median_drawdown60", "_policy_score",
    }
    if "_policy_score" in panel.columns:
        data = panel.loc[:, [column for column in panel.columns if column in policy_columns]].copy()
    else:
        data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    required = {
        "date", "code", "open", "high", "low", "close", "atr_14",
        "breadth_above_ma20", "market_median_mom20", "market_median_drawdown60",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"policy panel missing columns: {','.join(missing)}")
    data = data.dropna(subset=["date"]).sort_values(["date", "code"]).reset_index(drop=True)
    if "_policy_score" in data:
        data["score"] = pd.to_numeric(data["_policy_score"], errors="coerce")
    else:
        data["score"] = score_factors(data, profile)
    trading_dates = list(pd.DatetimeIndex(sorted(data["date"].unique())))
    date_positions = {day: index for index, day in enumerate(trading_dates)}
    proposed_dates = trading_dates[:: max(1, int(rebalance_every))]
    symbol_bars = {
        str(code): part.sort_values("date").reset_index(drop=True)
        for code, part in data.groupby(data["code"].astype(str), sort=False)
    }
    symbol_positions = {
        code: {day: index for index, day in enumerate(part["date"])}
        for code, part in symbol_bars.items()
    }

    busy_until = -1
    scheduled_dates = 0
    active_dates: list[pd.Timestamp] = []
    selections: dict[str, list[str]] = {}
    trades: list[dict] = []
    cohorts: list[dict] = []

    for signal_date in proposed_dates:
        signal_index = date_positions[signal_date]
        if signal_index < busy_until:
            continue
        scheduled_dates += 1
        cross = data[data["date"] == signal_date]
        if cross.empty:
            continue
        row = cross.iloc[0]
        regime = classify_regime(
            float(row["breadth_above_ma20"]),
            float(row["market_median_mom20"]),
            float(row["market_median_drawdown60"]),
            market_policy,
        )
        requested_count = pick_count_for_regime(regime, market_policy)
        key = signal_date.date().isoformat()
        if requested_count == 0:
            selections[key] = []
            cash_end = min(signal_index + max(1, int(rebalance_every)), len(trading_dates) - 1)
            busy_until = cash_end
            cohorts.append(
                {
                    "signal_date": signal_date,
                    "end_date": trading_dates[cash_end],
                    "duration": max(1, cash_end - (signal_index + 1)),
                    "regime": "weak",
                    "return": 0.0,
                    "cap_5": 0.0,
                    "cap_10": 0.0,
                    "cap_20": 0.0,
                }
            )
            continue

        candidates = cross.dropna(subset=["atr_14", "score"])
        picks = candidates.sort_values(["score", "code"], ascending=[False, True]).head(requested_count)
        cohort_end_index = signal_index + 1 + trade_policy.holding_days
        if picks.empty or cohort_end_index >= len(trading_dates):
            continue
        cohort_end = trading_dates[cohort_end_index]
        cohort_trades = []
        for pick in picks.to_dict("records"):
            code = str(pick["code"])
            bars = symbol_bars[code]
            signal_pos = symbol_positions[code].get(signal_date)
            if signal_pos is None:
                continue
            actual, cap_5, cap_10, cap_20 = _cached_trade_caps(
                code, signal_date, bars, signal_pos, pick["atr_14"], trade_policy, cost
            )
            if not all((actual, cap_5, cap_10, cap_20)):
                continue
            trade = {
                **actual,
                "signal_date": key,
                "cohort_end_date": cohort_end.date().isoformat(),
                "code": code,
                "atr_14": float(pick["atr_14"]),
                "trade_policy": trade_policy.to_dict(),
                "regime": regime,
                "return_cap_5": float(cap_5["net_return"]),
                "return_cap_10": float(cap_10["net_return"]),
                "return_cap_20": float(cap_20["net_return"]),
            }
            cohort_trades.append(trade)
            trades.append(trade)
        if not cohort_trades:
            continue
        codes = [item["code"] for item in cohort_trades]
        selections[key] = codes
        active_dates.append(signal_date)
        busy_until = cohort_end_index
        cohorts.append(
            {
                "signal_date": signal_date,
                "end_date": cohort_end,
                "duration": trade_policy.holding_days,
                "regime": regime,
                "return": float(np.mean([item["net_return"] for item in cohort_trades])),
                "cap_5": float(np.mean([item["return_cap_5"] for item in cohort_trades])),
                "cap_10": float(np.mean([item["return_cap_10"] for item in cohort_trades])),
                "cap_20": float(np.mean([item["return_cap_20"] for item in cohort_trades])),
            }
        )

    if not cohorts:
        empty = _empty_policy_metrics(profile, trade_policy, market_policy)
        empty["scheduled_dates"] = scheduled_dates
        return empty
    cohort_frame = pd.DataFrame(cohorts)
    actual_returns = pd.to_numeric(cohort_frame["return"], errors="coerce").fillna(0.0)
    total_days = max(1, int(pd.to_numeric(cohort_frame["duration"]).sum()))
    wealth = float(np.prod(1.0 + actual_returns.to_numpy(dtype="float64")))
    annualized = float(wealth ** (252.0 / total_days) - 1.0) if wealth > 0 else -1.0

    positive_trades = [item for item in trades if item["net_return"] > 0]
    positive_total = sum(item["net_return"] for item in positive_trades)
    by_code: dict[str, float] = {}
    by_regime: dict[str, float] = {}
    for item in positive_trades:
        by_code[item["code"]] = by_code.get(item["code"], 0.0) + item["net_return"]
        by_regime[item["regime"]] = by_regime.get(item["regime"], 0.0) + item["net_return"]
    top_share = max(by_code.values(), default=positive_total) / positive_total if positive_total > 0 else 1.0
    regime_share = max(by_regime.values(), default=positive_total) / positive_total if positive_total > 0 else 1.0
    fold_active_dates = []
    for fold in folds or []:
        start = pd.Timestamp(fold["valid_start"])
        end = pd.Timestamp(fold["valid_end"])
        fold_active_dates.append(sum(start <= day <= end for day in active_dates))
    return {
        "annualized_return": annualized,
        "mean_return_5": float(cohort_frame["cap_5"].mean()),
        "mean_return_10": float(cohort_frame["cap_10"].mean()),
        "mean_return_20": float(cohort_frame["cap_20"].mean()),
        "max_drawdown": max_drawdown(actual_returns),
        "win_rate": float((actual_returns > 0).mean()),
        "top_contributor_share": float(top_share),
        "best_regime_share": float(regime_share),
        "periods": int(len(cohort_frame)),
        "scheduled_dates": int(scheduled_dates),
        "active_dates": int(len(active_dates)),
        "activity_rate": float(len(active_dates) / scheduled_dates) if scheduled_dates else 0.0,
        "fold_active_dates": fold_active_dates,
        "selection_dates": [day.date().isoformat() for day in active_dates],
        "selections": selections,
        "trades": trades,
        "policy": trade_policy.to_dict(),
        "market_policy": market_policy.to_dict(),
    }


def _policy_benchmark_annualized(metrics: dict, index_bars: pd.DataFrame | None) -> float:
    if index_bars is None or index_bars.empty:
        return 0.0
    bars = index_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    bars = bars.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    positions = {day: index for index, day in enumerate(bars["date"])}
    trades = metrics.get("trades", [])
    by_signal: dict[str, list[dict]] = {}
    for trade in trades:
        by_signal.setdefault(str(trade["signal_date"]), []).append(trade)
    returns = []
    durations = []
    for signal_text in metrics.get("selections", {}):
        signal = pd.Timestamp(signal_text).normalize()
        signal_pos = positions.get(signal)
        if signal_pos is None:
            continue
        matching = by_signal.get(signal_text, [])
        if matching:
            entry_date = pd.Timestamp(matching[0]["entry_date"]).normalize()
            exit_date = pd.Timestamp(matching[0]["cohort_end_date"]).normalize()
            entry_pos = positions.get(entry_date)
            exit_pos = positions.get(exit_date)
        else:
            entry_pos = signal_pos + 1
            exit_pos = min(signal_pos + 5, len(bars) - 1)
        if entry_pos is None or exit_pos is None or exit_pos <= entry_pos:
            continue
        entry = float(bars.iloc[entry_pos]["open"])
        exit_price = float(bars.iloc[exit_pos]["close"])
        returns.append(exit_price / entry - 1.0)
        durations.append(exit_pos - entry_pos)
    if not returns:
        return 0.0
    wealth = float(np.prod(1.0 + np.asarray(returns, dtype="float64")))
    total_days = max(1, int(sum(durations)))
    return float(wealth ** (252.0 / total_days) - 1.0) if wealth > 0 else -1.0


def _policy_objective(metrics: dict, index_bars: pd.DataFrame | None = None) -> float:
    """Balanced training-only objective; hard gates are checked after validation."""
    if not metrics.get("periods"):
        return float("-inf")
    activity_penalty = max(0.0, 0.40 - float(metrics.get("activity_rate", 0.0))) * 2.0
    concentration_penalty = max(0.0, float(metrics.get("top_contributor_share", 1.0)) - 0.35)
    regime_penalty = max(0.0, float(metrics.get("best_regime_share", 1.0)) - 0.60)
    benchmark_annualized = _policy_benchmark_annualized(metrics, index_bars)
    return float(
        metrics["annualized_return"]
        + 0.80 * (metrics["annualized_return"] - benchmark_annualized)
        + 1.5 * metrics["mean_return_10"]
        - 0.70 * abs(metrics["max_drawdown"])
        - activity_penalty
        - concentration_penalty
        - regime_penalty
        + 0.0040 * float(metrics.get("active_dates", 0))
    )


def _prepare_policy_score(panel: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Precompute one factor ranking and retain only columns needed by policy trials."""
    required = [
        "date", "code", "open", "high", "low", "close", "volume", "amount",
        "atr_14", "breadth_above_ma20", "market_median_mom20",
        "market_median_drawdown60",
    ]
    factors = [name for name in profile.get("weights", {}) if name in panel.columns]
    columns = list(dict.fromkeys(required + factors))
    slim = panel.loc[:, columns].copy()
    slim["_policy_score"] = score_factors(slim, profile)
    return slim.loc[:, required + ["_policy_score"]]


def _best_variant(panel, profile, trade_policy, market_policy, variants, kind, cost, index_bars=None):
    best = None
    best_metrics = None
    best_score = float("-inf")
    panel_dates = pd.to_datetime(panel["date"], errors="coerce")
    date_span = max(1, (panel_dates.max() - panel_dates.min()).days)
    half_years = max(1, int(np.ceil(date_span / 183.0)))
    for variant in variants:
        trade = variant if kind == "trade" else trade_policy
        market = variant if kind == "market" else market_policy
        metrics = evaluate_policy_profile(panel, profile, trade, market, cost=cost)
        score = _policy_objective(metrics, index_bars)
        activity_deficit = max(0, 8 * half_years - int(metrics.get("active_dates", 0)))
        score -= 0.20 * activity_deficit
        score -= 2.0 * max(0.0, 0.40 - float(metrics.get("activity_rate", 0.0)))
        if score > best_score:
            best, best_metrics, best_score = variant, metrics, score
    return best, best_metrics


def _coordinate_trade_search(panel, profile, market_policy, search_space, cost, version, index_bars=None):
    panel = _prepare_policy_score(panel, profile)
    current = TradePolicy(version, 10, 1.5, 2.0)
    axes = (
        ("holding_days", search_space["holding_days"]),
        ("atr_multiplier", search_space["atr_multipliers"]),
        ("reward_risk", search_space["reward_risk"]),
    )
    for _ in range(1):
        for field_name, values in axes:
            variants = []
            for value in values:
                data = current.to_dict()
                data[field_name] = value
                variants.append(TradePolicy.from_dict(data))
            current, _ = _best_variant(
                panel, profile, current, market_policy, variants, "trade", cost, index_bars
            )
    return current


def _coordinate_market_search(panel, profile, trade_policy, search_space, cost, index_bars=None):
    panel = _prepare_policy_score(panel, profile)
    current = MarketPolicy(0.35, -0.03, -0.30, 15, 10)
    axes = (
        ("breadth_floor", search_space["breadth_floors"]),
        ("momentum_floor", search_space["momentum_floors"]),
        ("drawdown_floor", search_space["drawdown_floors"]),
        ("strong_pick_count", search_space["strong_pick_counts"]),
        ("normal_pick_count", search_space["normal_pick_counts"]),
    )
    for _ in range(1):
        for field_name, values in axes:
            variants = []
            for value in values:
                data = current.to_dict()
                data[field_name] = value
                variants.append(MarketPolicy.from_dict(data))
            current, _ = _best_variant(
                panel, profile, trade_policy, current, variants, "market", cost, index_bars
            )
    return current


def _modal_dict(values: list[dict]) -> dict:
    encoded = [json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values]
    counts = Counter(encoded)
    last = {value: max(i for i, item in enumerate(encoded) if item == value) for value in counts}
    return json.loads(max(counts, key=lambda value: (counts[value], last[value])))


def _aggregate_fold_metrics(
    fold_outputs: list[dict], live_policy: TradePolicy, live_market: MarketPolicy
) -> dict:
    """Stitch only the chronological validation cohorts chosen by preceding training."""
    selections = {}
    trades = []
    cohorts = []
    selection_dates = []
    fold_active_dates = []
    scheduled_dates = 0
    for fold in fold_outputs:
        metrics = fold["validation_metrics"]
        fold_trades = metrics.get("trades", [])
        trades.extend(fold_trades)
        selections.update(metrics.get("selections", {}))
        selection_dates.extend(metrics.get("selection_dates", []))
        fold_active_dates.append(int(metrics.get("active_dates", 0)))
        scheduled_dates += int(metrics.get("scheduled_dates", 0))
        by_signal: dict[str, list[dict]] = {}
        for trade in fold_trades:
            by_signal.setdefault(str(trade["signal_date"]), []).append(trade)
        for signal_date, codes in metrics.get("selections", {}).items():
            items = by_signal.get(signal_date, [])
            if items:
                policy_data = items[0].get("trade_policy") or metrics.get("policy", {})
                duration = int(policy_data.get("holding_days", 5))
                cohorts.append(
                    {
                        "date": pd.Timestamp(signal_date),
                        "duration": duration,
                        "return": float(np.mean([item["net_return"] for item in items])),
                        "cap_5": float(np.mean([item["return_cap_5"] for item in items])),
                        "cap_10": float(np.mean([item["return_cap_10"] for item in items])),
                        "cap_20": float(np.mean([item["return_cap_20"] for item in items])),
                    }
                )
            elif not codes:
                cohorts.append(
                    {
                        "date": pd.Timestamp(signal_date), "duration": 4, "return": 0.0,
                        "cap_5": 0.0, "cap_10": 0.0, "cap_20": 0.0,
                    }
                )
    if not cohorts:
        result = _empty_policy_metrics({}, live_policy, live_market)
        result["fold_active_dates"] = fold_active_dates
        result["scheduled_dates"] = scheduled_dates
        return result
    cohort_frame = pd.DataFrame(cohorts).sort_values("date")
    actual = pd.to_numeric(cohort_frame["return"], errors="coerce").fillna(0.0)
    total_days = max(1, int(pd.to_numeric(cohort_frame["duration"]).sum()))
    wealth = float(np.prod(1.0 + actual.to_numpy(dtype="float64")))
    annualized = float(wealth ** (252.0 / total_days) - 1.0) if wealth > 0 else -1.0
    positives = [trade for trade in trades if float(trade.get("net_return", 0.0)) > 0]
    positive_total = sum(float(trade["net_return"]) for trade in positives)
    by_code: dict[str, float] = {}
    by_regime: dict[str, float] = {}
    for trade in positives:
        value = float(trade["net_return"])
        by_code[str(trade["code"])] = by_code.get(str(trade["code"]), 0.0) + value
        regime = str(trade.get("regime", "unknown"))
        by_regime[regime] = by_regime.get(regime, 0.0) + value
    top_share = max(by_code.values(), default=positive_total) / positive_total if positive_total else 1.0
    regime_share = max(by_regime.values(), default=positive_total) / positive_total if positive_total else 1.0
    active_dates = len(selection_dates)
    return {
        "annualized_return": annualized,
        "mean_return_5": float(cohort_frame["cap_5"].mean()),
        "mean_return_10": float(cohort_frame["cap_10"].mean()),
        "mean_return_20": float(cohort_frame["cap_20"].mean()),
        "max_drawdown": max_drawdown(actual),
        "win_rate": float((actual > 0).mean()),
        "top_contributor_share": float(top_share),
        "best_regime_share": float(regime_share),
        "periods": int(len(cohort_frame)),
        "scheduled_dates": scheduled_dates,
        "active_dates": active_dates,
        "activity_rate": float(active_dates / scheduled_dates) if scheduled_dates else 0.0,
        "fold_active_dates": fold_active_dates,
        "selection_dates": selection_dates,
        "selections": selections,
        "trades": trades,
        "policy": live_policy.to_dict(),
        "market_policy": live_market.to_dict(),
    }


def _adjacent_trade_policies(policy: TradePolicy, search_space: dict) -> list[TradePolicy]:
    mapping = {
        "holding_days": search_space["holding_days"],
        "atr_multiplier": search_space["atr_multipliers"],
        "reward_risk": search_space["reward_risk"],
    }
    output = []
    for field_name, values in mapping.items():
        position = values.index(getattr(policy, field_name))
        for neighbor_index in (position - 1, position + 1):
            if 0 <= neighbor_index < len(values):
                data = policy.to_dict()
                data[field_name] = values[neighbor_index]
                output.append(TradePolicy.from_dict(data))
    return output


def optimize_policy_profile(
    panel, spec, folds, search_space, profile_key, cost=0.003, index_bars=None
) -> dict:
    """Chronological three-layer coordinate search with validation-only reporting."""
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    fold_outputs = []
    chosen_profiles = []
    chosen_trades = []
    chosen_markets = []
    policy_version = f"risk-gated-20260714-{profile_key}"
    default_market = MarketPolicy(0.35, -0.03, -0.30, 15, 10)
    for fold in folds:
        _TRADE_SIM_CACHE.clear()
        train_end = pd.Timestamp(fold["train_end"])
        valid_start = pd.Timestamp(fold["valid_start"])
        valid_end = pd.Timestamp(fold["valid_end"])
        if not train_end < valid_start <= valid_end:
            raise ValueError("fold dates must be chronological")
        train = data[data["date"] <= train_end]
        valid = data[data["date"].between(valid_start, valid_end)]
        baseline_profile = next(item for item in spec["candidates"] if item["name"] == spec["baseline"])
        trade = _coordinate_trade_search(
            train, baseline_profile, default_market, search_space, cost, policy_version,
            index_bars,
        )
        market = _coordinate_market_search(
            train, baseline_profile, trade, search_space, cost, index_bars
        )
        weight_metrics = [
            evaluate_policy_profile(
                _prepare_policy_score(train, candidate), candidate, trade, market, cost=cost
            )
            for candidate in spec["candidates"]
        ]
        best_index = max(
            range(len(weight_metrics)),
            key=lambda index: _policy_objective(weight_metrics[index], index_bars),
        )
        chosen_profile = spec["candidates"][best_index]
        validation = evaluate_policy_profile(
            _prepare_policy_score(valid, chosen_profile), chosen_profile, trade, market, cost=cost
        )
        chosen_profiles.append(chosen_profile)
        chosen_trades.append(trade.to_dict())
        chosen_markets.append(market.to_dict())
        fold_outputs.append(
            {
                "train_end": train_end.date().isoformat(),
                "train_max_date": train["date"].max().date().isoformat() if not train.empty else None,
                "validation_min_date": valid["date"].min().date().isoformat() if not valid.empty else None,
                "validation_max_date": valid["date"].max().date().isoformat() if not valid.empty else None,
                "chosen_profile": chosen_profile["name"],
                "chosen_weights": chosen_profile["weights"],
                "trade_policy": trade.to_dict(),
                "market_policy": market.to_dict(),
                "validation_metrics": validation,
            }
        )
        print(
            f"{profile_key} fold {valid_start.date()}..{valid_end.date()}: "
            f"{chosen_profile['name']} annual={validation['annualized_return']:.2%} "
            f"dd={validation['max_drawdown']:.2%}",
            flush=True,
        )

    modal_profile = _modal_dict(chosen_profiles)
    modal_trade = TradePolicy.from_dict(_modal_dict(chosen_trades))
    modal_market = MarketPolicy.from_dict(_modal_dict(chosen_markets))
    validation_mask = pd.Series(False, index=data.index)
    for fold in folds:
        validation_mask |= data["date"].between(pd.Timestamp(fold["valid_start"]), pd.Timestamp(fold["valid_end"]))
    validation_panel = data[validation_mask]
    candidate = _aggregate_fold_metrics(fold_outputs, modal_trade, modal_market)
    baseline_profile = next(item for item in spec["candidates"] if item["name"] == spec["baseline"])
    baseline = evaluate_policy_profile(
        _prepare_policy_score(validation_panel, baseline_profile),
        baseline_profile,
        TradePolicy(f"baseline-{profile_key}", 10, 1.5, 2.0),
        default_market,
        folds=folds,
        cost=cost,
    )
    neighbor_items = []
    for neighbor in _adjacent_trade_policies(modal_trade, search_space):
        metrics = evaluate_policy_profile(
            _prepare_policy_score(validation_panel, modal_profile), modal_profile,
            neighbor, modal_market, folds=folds, cost=cost
        )
        neighbor_items.append(
            {
                "policy": neighbor.to_dict(),
                "mean_return": metrics["mean_return_10"],
                "max_drawdown": metrics["max_drawdown"],
            }
        )
    live_metrics = evaluate_policy_profile(
        _prepare_policy_score(validation_panel, modal_profile), modal_profile,
        modal_trade, modal_market, folds=folds, cost=cost,
    )
    neighbor_result = neighbor_stability(live_metrics["max_drawdown"], neighbor_items)
    active_counts = [len(value) for value in candidate["selections"].values() if value]
    participation = participation_checks(
        candidate["scheduled_dates"], active_counts, candidate["fold_active_dates"]
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "accepted": bool(participation["passed"] and neighbor_result["passed"]),
        "chosen_weights": modal_profile["weights"],
        "participation": participation,
        "neighbor_stability": {**neighbor_result, "neighbors": neighbor_items},
        "walk_forward": {"folds": fold_outputs},
    }


def run_policy_optimization(
    panel,
    profiles,
    folds,
    search_space,
    start,
    end,
    universe,
    fetch_errors,
    output_json,
    output_csv,
    cost,
):
    index_path = Path(r"C:\InStockLocal\backtest_cache\indices\000300.csv")
    index_bars = pd.read_csv(index_path) if index_path.exists() else None
    results = {}
    summary_rows = []
    for profile_key, spec in profiles.items():
        scoped = filter_profile_universe(panel, spec.get("universe", "all"))
        evidence = optimize_policy_profile(
            scoped, spec, folds, search_space, profile_key, cost, index_bars=index_bars
        )
        results[profile_key] = evidence
        summary_rows.append(
            {
                "profile": profile_key,
                "candidate": evidence["candidate"]["policy"]["policy_version"],
                "annualized_return": evidence["candidate"]["annualized_return"],
                "mean_return_10": evidence["candidate"]["mean_return_10"],
                "max_drawdown": evidence["candidate"]["max_drawdown"],
                "activity_rate": evidence["candidate"]["activity_rate"],
                "participation_passed": evidence["participation"]["passed"],
                "neighbor_passed": evidence["neighbor_stability"]["passed"],
            }
        )
    dates = pd.to_datetime(panel["date"])
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": {
            "signal": "close",
            "entry": "next_open",
            "round_trip_cost": cost,
            "non_overlapping_cohorts": True,
            "weak_regime_cash": True,
            "walk_forward_folds": folds,
            "search_space": search_space,
        },
        "coverage": {
            "requested_start": start,
            "requested_end": end,
            "actual_start": dates.min().date().isoformat(),
            "actual_end": dates.max().date().isoformat(),
            "requested_symbols": int(len(universe)),
            "usable_symbols": int(panel["code"].nunique()),
            "stock_dates": int(len(panel)),
            "fetch_failures": fetch_errors,
        },
        "profiles": results,
        "caveats": [
            "Current-universe construction introduces survivorship bias.",
            "Rolling out-of-sample history does not guarantee future returns.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_json)
    pd.DataFrame(summary_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
