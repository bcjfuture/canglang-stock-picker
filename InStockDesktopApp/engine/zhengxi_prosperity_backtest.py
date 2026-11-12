# -*- coding: utf-8 -*-
"""Point-in-time evaluation helpers for the Zhengxi-inspired strategy."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def forward_returns(
    selected: pd.DataFrame,
    bars_by_code: Mapping[str, pd.DataFrame],
    horizons: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Measure close-to-close returns from the signal session to future sessions."""
    records = []
    for source in selected.to_dict("records"):
        code = str(source.get("code") or "").zfill(6)
        signal_date = pd.Timestamp(source["signal_date"]).normalize()
        bars = bars_by_code.get(code)
        if bars is None or bars.empty:
            continue
        prices = bars.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        prices = prices.dropna(subset=["date", "close"]).sort_values("date")
        prices = prices.drop_duplicates("date", keep="last").reset_index(drop=True)
        matching = prices.index[prices["date"] == signal_date]
        if len(matching) != 1:
            continue
        entry_index = int(matching[0])
        entry_close = float(prices.loc[entry_index, "close"])
        if entry_close <= 0:
            continue
        record = dict(source)
        record["code"] = code
        record["entry_date"] = signal_date
        record["entry_close"] = entry_close
        for horizon in horizons:
            target_index = entry_index + int(horizon)
            record[f"{horizon}d_return"] = (
                float(prices.loc[target_index, "close"]) / entry_close - 1.0
                if target_index < len(prices)
                else float("nan")
            )
        records.append(record)
    return pd.DataFrame(records)


def _stat(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "mean": None, "median": None, "win_rate": None}
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "win_rate": float((numeric > 0).mean()),
    }


def _turnover(observations: pd.DataFrame) -> float | None:
    dates = sorted(pd.to_datetime(observations["signal_date"]).dt.normalize().unique())
    if len(dates) < 2:
        return None
    sets = {
        date: set(
            observations.loc[
                pd.to_datetime(observations["signal_date"]).dt.normalize() == date,
                "code",
            ].astype(str)
        )
        for date in dates
    }
    values = []
    for previous, current in zip(dates, dates[1:]):
        denominator = max(len(sets[previous]), len(sets[current]), 1)
        values.append(1.0 - len(sets[previous] & sets[current]) / denominator)
    return float(pd.Series(values).mean())


def summarize_backtest(
    observations: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> dict:
    """Summarize horizons and keep strict/fallback tiers separately visible."""
    frame = observations.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    benchmark_frame = benchmark.copy()
    benchmark_frame["signal_date"] = pd.to_datetime(
        benchmark_frame["signal_date"]
    ).dt.normalize()
    horizon_columns = sorted(
        (column for column in frame if column.endswith("d_return")),
        key=lambda column: int(column.removesuffix("d_return")),
    )
    horizons = {}
    for column in horizon_columns:
        label = column.removesuffix("_return")
        stats = _stat(frame[column])
        benchmark_column = column if column in benchmark_frame else None
        if benchmark_column:
            merged = frame[["signal_date", column]].merge(
                benchmark_frame[["signal_date", benchmark_column]],
                on="signal_date",
                how="left",
                suffixes=("", "_benchmark"),
            )
            excess = merged[column] - merged[f"{benchmark_column}_benchmark"]
            stats["excess_mean"] = (
                float(excess.dropna().mean()) if excess.notna().any() else None
            )
        else:
            stats["excess_mean"] = None
        horizons[label] = stats

    tier_performance = {}
    for tier, group in frame.groupby("selection_tier", sort=False):
        tier_performance[str(tier)] = {
            column.removesuffix("_return"): _stat(group[column])
            for column in horizon_columns
        }

    industry_max = []
    for _signal_date, group in frame.groupby("signal_date"):
        shares = group["industry"].value_counts(normalize=True)
        if not shares.empty:
            industry_max.append(float(shares.max()))

    max_drawdown = None
    if horizon_columns:
        periodic = frame.groupby("signal_date")[horizon_columns[0]].mean().dropna()
        if not periodic.empty:
            wealth = (1.0 + periodic).cumprod()
            max_drawdown = float((wealth / wealth.cummax() - 1.0).min())

    return {
        "horizons": horizons,
        "tier_performance": tier_performance,
        "max_drawdown_proxy": max_drawdown,
        "turnover": _turnover(frame),
        "industry_concentration": (
            float(pd.Series(industry_max).mean()) if industry_max else None
        ),
    }
