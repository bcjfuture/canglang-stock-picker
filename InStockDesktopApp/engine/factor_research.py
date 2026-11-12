# -*- coding: utf-8 -*-
"""Factor IC research over the full A-share daily history.

usage: factor_research.py [output.json] [--start YYYY-MM-DD] [--min-amount 5e7]

For every factor the cross-sectional rank IC against forward returns is
computed per signal date.  Forward returns enter at the next session's open
(skipped when that open is at the limit-up price, i.e. not buyable) and exit
at the close ``h`` sessions after the signal.  Multi-day horizons are sampled
every ``h`` sessions so the t-statistics use non-overlapping periods.

Universe: stocks in InStock's history cache.  Delisted names are absent, so
results carry survivorship bias; liquidity filtering keeps it to tradable
names.
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import daily_history as dh
from selection_refinement import EVIDENCE_FACTORS


HORIZONS = (1, 5, 10, 20)
QUANTILES = 5
REPORT_DIR = Path(r"C:\InStockLocal\reports")


def limit_ratio(code: str) -> float:
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("4", "8", "92")):
        return 0.30
    return 0.10


def load_panel(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    calendar = dh.trading_calendar()
    hist_dir = dh.latest_hist_dir()
    if hist_dir is None:
        raise RuntimeError("沧浪之水 history cache not found")
    codes = sorted(path.name[:6] for path in hist_dir.glob("*qfq.gzip.pickle"))
    sessions = calendar[(calendar >= start) & (calendar <= end)]
    # Research needs the long history; repaired files only cover recent months,
    # so gaps are left as missing values and windows touching them become NaN.
    no_repair = Path(hist_dir) / "__no_repair__"

    def one(code):
        bars, _meta = dh.load_code_history(
            code, end, calendar=calendar, hist_dir=hist_dir, repair_dir=no_repair
        )
        if bars.empty:
            return code, None
        return code, bars.set_index("date").reindex(sessions)

    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded = dict(executor.map(one, codes))
    loaded = {code: frame for code, frame in loaded.items() if frame is not None}
    panel = {
        field: pd.DataFrame({code: frame[field] for code, frame in loaded.items()}, index=sessions)
        for field in ("open", "high", "low", "close", "amount")
    }
    return panel


def evidence_composite(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Same composite the strategies use, ranked per date across the market."""
    total = None
    count = None
    for name, sign in EVIDENCE_FACTORS.items():
        ranked = parts[name].rank(axis=1, pct=True)
        contribution = ranked if sign > 0 else 1.0 - ranked
        present = contribution.notna().astype(float)
        contribution = contribution.fillna(0.0)
        total = contribution if total is None else total + contribution
        count = present if count is None else count + present
    return (total / count).where(count >= len(EVIDENCE_FACTORS) - 1)


def compute_factors(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close, high, low, amount = panel["close"], panel["high"], panel["low"], panel["amount"]
    daily = close.pct_change(fill_method=None)

    def mean(frame, window):
        return frame.rolling(window, min_periods=window).mean()

    ma20, ma60 = mean(close, 20), mean(close, 60)
    amount5, amount20, amount60 = mean(amount, 5), mean(amount, 20), mean(amount, 60)
    max60 = close.rolling(60, min_periods=60).max()
    low20 = low.rolling(20, min_periods=20).min()
    recent_low3 = low.rolling(3, min_periods=3).min()
    preceding_low5 = low.shift(3).rolling(5, min_periods=5).min()
    drawdown = close / close.rolling(60, min_periods=20).max() - 1
    prior_drawdown20 = drawdown.shift(1).rolling(20, min_periods=20).min()
    evidence = evidence_composite(
        {
            "mom_20": close / close.shift(20) - 1,
            "mom_5": close / close.shift(5) - 1,
            "volatility_20": daily.rolling(20, min_periods=20).std(),
            "abnormal_amount": amount5 / amount60,
            "recovery_from_low": close / low20 - 1,
            "ma20_gap": close / ma20 - 1,
        }
    )
    return {
        "证据组合(v2)": evidence,
        "当日涨幅": daily,
        "5日反转(-5日涨幅)": -(close / close.shift(5) - 1),
        "20日动量": close / close.shift(20) - 1,
        "60日动量": close / close.shift(60) - 1,
        "中期动量(60日去近20日)": close.shift(20) / close.shift(60) - 1,
        "20日波动率": daily.rolling(20, min_periods=20).std(),
        "20日成交额(对数)": np.log(amount20.where(amount20 > 0)),
        "异常成交(5日/60日)": amount5 / amount60,
        "量比(当日/前5日)": amount / amount5.shift(1),
        "偏离20日线": close / ma20 - 1,
        "20/60日均线趋势": ma20 / ma60 - 1,
        "距60日高点": close / max60 - 1,
        "距20日低点反弹": close / low20 - 1,
        "反弹形态": (
            (prior_drawdown20 <= -0.05)
            & (recent_low3 >= preceding_low5)
            & (close > mean(close, 10))
            & ((close / low20 - 1) <= 0.15)
        ).astype(float).where(prior_drawdown20.notna()),
        "过热(20日涨幅>25%或偏离>15%)": (
            ((close / close.shift(20) - 1) > 0.25) | ((close / ma20 - 1) > 0.15)
        ).astype(float).where(ma20.notna()),
    }


def forward_returns(panel: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    close, open_ = panel["close"], panel["open"]
    entry = open_.shift(-1)
    limits = pd.Series({code: limit_ratio(code) for code in close.columns})
    limit_up = entry >= close * (1 + limits - 0.005)
    exit_ = close.shift(-horizon)
    result = exit_ / entry - 1
    return result.where(~limit_up)


def rank_ic(factor: pd.DataFrame, returns: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    f = factor.where(mask)
    r = returns.where(mask & f.notna())
    f = f.where(r.notna())
    fr = f.rank(axis=1)
    rr = r.rank(axis=1)
    count = fr.notna().sum(axis=1)
    fr = fr.sub(fr.mean(axis=1), axis=0)
    rr = rr.sub(rr.mean(axis=1), axis=0)
    ic = (fr * rr).sum(axis=1) / np.sqrt((fr**2).sum(axis=1) * (rr**2).sum(axis=1))
    return ic.where(count >= 200)


def quantile_spread(factor: pd.DataFrame, returns: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    f = factor.where(mask)
    r = returns.where(mask & f.notna())
    f = f.where(r.notna())
    pct = f.rank(axis=1, pct=True)
    top = r.where(pct > 1 - 1 / QUANTILES).mean(axis=1)
    bottom = r.where(pct <= 1 / QUANTILES).mean(axis=1)
    market = r.mean(axis=1)
    return top - bottom, top - market


def summarize(series: pd.Series, step: int) -> dict:
    sampled = series.dropna().iloc[::step]
    n = len(sampled)
    if n < 5:
        return {"n": n}
    mean = float(sampled.mean())
    std = float(sampled.std(ddof=1))
    return {
        "n": n,
        "mean": mean,
        "t": mean / (std / math.sqrt(n)) if std > 0 else None,
        "positive_share": float((sampled > 0).mean()),
    }


def regime_study(panel: dict[str, pd.DataFrame], mask: pd.DataFrame, horizon: int = 5) -> list[dict]:
    """Does market breadth at the signal predict the next equal-weight return?"""
    close = panel["close"]
    ma20 = close.rolling(20, min_periods=20).mean()
    breadth = (close > ma20).where(mask & ma20.notna()).mean(axis=1)
    median_mom20 = (close / close.shift(20) - 1).where(mask).median(axis=1)
    returns = forward_returns(panel, horizon).where(mask).mean(axis=1)
    frame = pd.DataFrame({"breadth": breadth, "mom20": median_mom20, "ret": returns}).dropna()
    frame = frame.iloc[::horizon]
    rows = []
    for label, condition in (
        ("弱势(宽度<35%且20日中位动量<0)", (frame.breadth < 0.35) & (frame.mom20 < 0)),
        ("强势(宽度>60%)", frame.breadth > 0.60),
        ("其他", ~(((frame.breadth < 0.35) & (frame.mom20 < 0)) | (frame.breadth > 0.60))),
    ):
        part = frame.loc[condition, "ret"]
        rows.append(
            {
                "regime": label,
                "periods": int(len(part)),
                "mean_next_5d_market_return": float(part.mean()) if len(part) else None,
                "win_share": float((part > 0).mean()) if len(part) else None,
            }
        )
    return rows


def snapshot_factors(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Fundamental and fund-flow factors from the daily DB snapshots (short history)."""
    import pymysql

    from selection_refinement import operating_cash_flow_positive, ttm_roe_pct

    connection = pymysql.connect(
        host="127.0.0.1", port=3306, user="root", password="root", database="instockdb", charset="utf8mb4"
    )
    try:
        selection = pd.read_sql(
            "SELECT date, code, pe9, pbnewmrq, netprofit_yoy_ratio, pcfjyxjl9, per_netcash_operate,"
            " debt_asset_ratio, roe_weight FROM cn_stock_selection",
            connection,
        )
        flows = pd.read_sql(
            "SELECT date, code, fund_amount, fund_amount_3, fund_amount_5 FROM cn_stock_fund_flow",
            connection,
        )
    finally:
        connection.close()
    selection["date"] = pd.to_datetime(selection["date"])
    flows["date"] = pd.to_datetime(flows["date"])
    records = selection.to_dict("records")
    selection["roe_ttm"] = [ttm_roe_pct(row) for row in records]
    selection["cash_positive"] = [
        {True: 1.0, False: 0.0}.get(operating_cash_flow_positive(row), np.nan) for row in records
    ]
    amount20 = panel["amount"].rolling(20, min_periods=20).mean()
    stacked_amount = amount20.stack().rename("amount20").reset_index()
    stacked_amount.columns = ["date", "code", "amount20"]
    flows = flows.merge(stacked_amount, on=["date", "code"], how="left")
    one, three, five = (pd.to_numeric(flows[c], errors="coerce") for c in ("fund_amount", "fund_amount_3", "fund_amount_5"))
    flows["flow_intensity"] = (one + (three - one) + (five - three)) / (5 * flows["amount20"])
    flows["flow_1d"] = one / flows["amount20"]

    def pivot(frame, column):
        table = frame.pivot_table(index="date", columns="code", values=column, aggfunc="last")
        return table.reindex(index=panel["close"].index, columns=panel["close"].columns)

    return {
        "TTM ROE(PB/PE)": pivot(selection, "roe_ttm"),
        "报告期ROE(原口径)": pivot(selection, "roe_weight"),
        "净利同比": pivot(selection, "netprofit_yoy_ratio"),
        "经营现金流为正(TTM)": pivot(selection, "cash_positive"),
        "资产负债率": pivot(selection, "debt_asset_ratio"),
        "低PE(盈利收益率)": 1.0 / pivot(selection, "pe9"),
        "5日主力资金强度": pivot(flows, "flow_intensity"),
        "当日主力资金强度": pivot(flows, "flow_1d"),
    }


def main(argv: list[str]) -> int:
    start = pd.Timestamp("2023-09-01")
    min_amount = 5e7
    output = None
    use_db = False
    args = list(argv)
    while args:
        item = args.pop(0)
        if item == "--start":
            start = pd.Timestamp(args.pop(0))
        elif item == "--min-amount":
            min_amount = float(args.pop(0))
        elif item == "--db":
            use_db = True
        else:
            output = Path(item)
    calendar = dh.trading_calendar()
    end = calendar.max()
    print(f"loading panel {start.date()}..{end.date()}", flush=True)
    panel = load_panel(start, end)
    print(f"panel: {panel['close'].shape[1]} codes x {panel['close'].shape[0]} sessions", flush=True)
    amount20 = panel["amount"].rolling(20, min_periods=20).mean()
    mask = (amount20 >= min_amount) & panel["close"].notna()
    factors = compute_factors(panel)
    snapshot_names: set[str] = set()
    if use_db:
        try:
            extra = snapshot_factors(panel)
            snapshot_names = set(extra)
            factors.update(extra)
        except Exception as exc:
            print(f"snapshot factors unavailable: {exc}", flush=True)
    results = []
    for horizon in HORIZONS:
        returns = forward_returns(panel, horizon)
        for name, factor in factors.items():
            ic = rank_ic(factor, returns, mask)
            spread, top_excess = quantile_spread(factor, returns, mask)
            yearly = {
                str(year): summarize(part, horizon).get("mean")
                for year, part in ic.groupby(ic.index.year)
            }
            results.append(
                {
                    "factor": name,
                    "short_sample": name in snapshot_names,
                    "horizon": horizon,
                    "ic": summarize(ic, horizon),
                    "ic_by_year": yearly,
                    "top_minus_bottom": summarize(spread, horizon),
                    "top_quintile_excess": summarize(top_excess, horizon),
                }
            )
        print(f"horizon {horizon} done", flush=True)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "min_average_amount_20": min_amount,
        "universe_codes": int(panel["close"].shape[1]),
        "notes": [
            "Entry next open (limit-up opens skipped), exit close h sessions later; no costs.",
            "Universe excludes delisted stocks (survivorship bias).",
            "t-statistics use non-overlapping periods.",
        ],
        "factors": results,
        "regimes": regime_study(panel, mask),
    }
    output = output or REPORT_DIR / f"factor-research-{end.date().isoformat()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}", flush=True)
    for horizon in (1, 5, 20):
        print(f"\n== horizon {horizon} sessions: rank IC mean (t) | top-bottom quintile per period ==")
        rows = [item for item in results if item["horizon"] == horizon]
        rows.sort(key=lambda item: -abs(item["ic"].get("mean") or 0))
        for item in rows:
            ic, spread = item["ic"], item["top_minus_bottom"]
            print(
                f"  {'*' if item['short_sample'] else ' '}{item['factor']:<26} IC {ic.get('mean', float('nan')):+.4f} (t {ic.get('t') or float('nan'):+.2f}, "
                f"正 {ic.get('positive_share', float('nan')):.0%})  多空 {spread.get('mean', float('nan')):+.2%}  "
                f"年度 {item['ic_by_year']}"
            )
    print("\n== market regime ==")
    for row in report["regimes"]:
        print(" ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
