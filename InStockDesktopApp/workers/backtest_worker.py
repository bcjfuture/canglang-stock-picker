from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from instock_app.settings import AppPaths
from workers.common import write_json_atomic


INDEX_SECIDS = {"000300": "1.000300", "000905": "1.000905"}
Emitter = Callable[[str, str, int, dict | None], None]


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: list[dict[str, object]]
    reasons: list[str]
    policy_checks: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "policy_checks": self.policy_checks,
            "reasons": self.reasons,
        }


def annualized_return(returns: pd.Series, hold: int) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    wealth = float(np.prod(1.0 + clean.to_numpy(dtype="float64")))
    if wealth <= 0:
        return -1.0
    return float(wealth ** (252.0 / (len(clean) * hold)) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if clean.empty:
        return 0.0
    wealth = (1.0 + clean).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def periodic_returns(signals: pd.DataFrame, index_bars: pd.DataFrame, hold: int) -> pd.DataFrame:
    bars = index_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    bars = bars.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    positions = {value: index for index, value in enumerate(bars["date"])}
    rows: list[dict[str, object]] = []
    last_exit_position = -1
    ordered = signals.copy()
    ordered["signal_date"] = pd.to_datetime(ordered["signal_date"]).dt.normalize()
    ordered = ordered.sort_values("signal_date")
    for signal in ordered.to_dict("records"):
        signal_pos = positions.get(signal["signal_date"])
        if signal_pos is None:
            continue
        entry_pos = signal_pos + 1
        exit_pos = entry_pos + hold
        if entry_pos <= last_exit_position or exit_pos >= len(bars):
            continue
        entry = float(bars.iloc[entry_pos]["open"])
        exit_price = float(bars.iloc[exit_pos]["close"])
        if not (math.isfinite(entry) and entry > 0 and math.isfinite(exit_price)):
            continue
        rows.append(
            {
                "signal_date": signal["signal_date"].date().isoformat(),
                "entry_date": bars.iloc[entry_pos]["date"].date().isoformat(),
                "exit_date": bars.iloc[exit_pos]["date"].date().isoformat(),
                "strategy_return": float(signal["strategy_return"]),
                "index_return": float(exit_price / entry - 1.0),
            }
        )
        last_exit_position = exit_pos
    return pd.DataFrame(rows)


def evaluate_gate(
    strategy: dict[str, float],
    baseline: dict[str, float],
    benchmark: dict[str, float],
    folds: list[dict[str, object]],
    policy_checks: list[dict[str, object]] | None = None,
) -> GateResult:
    excess = strategy["annualized_return"] - benchmark["annualized_return"]
    checks = [
        {
            "name": "样本外年化超额至少5个百分点",
            "passed": excess >= 0.05,
            "value": excess,
        },
        {
            "name": "10日平均净收益为正且高于原策略",
            "passed": strategy["mean_return_10"] > 0
            and strategy["mean_return_10"] > baseline["mean_return_10"],
            "value": strategy["mean_return_10"] - baseline["mean_return_10"],
        },
        {
            "name": "多数滚动窗口跑赢沪深300",
            "passed": len(folds) >= 5
            and sum(bool(item.get("beat_000300")) for item in folds) > len(folds) / 2,
            "value": sum(bool(item.get("beat_000300")) for item in folds),
        },
        {
            "name": "最大回撤限制",
            "passed": abs(strategy["max_drawdown"]) <= abs(benchmark["max_drawdown"]) + 0.10
            and abs(strategy["max_drawdown"]) <= abs(baseline["max_drawdown"]) * 1.15 + 1e-12,
            "value": strategy["max_drawdown"],
        },
        {
            "name": "5日与20日不同时恶化",
            "passed": not (
                strategy["mean_return_5"] < baseline["mean_return_5"]
                and strategy["mean_return_20"] < baseline["mean_return_20"]
            ),
            "value": {
                "mean_return_5": strategy["mean_return_5"],
                "mean_return_20": strategy["mean_return_20"],
            },
        },
        {
            "name": "收益贡献不过度集中",
            "passed": strategy["top_contributor_share"] <= 0.35
            and strategy["best_regime_share"] <= 0.60,
            "value": {
                "top_contributor_share": strategy["top_contributor_share"],
                "best_regime_share": strategy["best_regime_share"],
            },
        },
    ]
    reasons: list[str] = []
    labels = [
        "年化超额不足5个百分点",
        "10日净收益没有同时为正且优于原策略",
        "跑赢沪深300的滚动窗口未过半",
        "最大回撤超出限制",
        "5日和20日收益同时恶化",
        "收益贡献过度集中",
    ]
    for check, label in zip(checks, labels):
        if not check["passed"]:
            reasons.append(label)
    policy_checks = list(policy_checks or [])
    for check in policy_checks:
        if not bool(check.get("passed")):
            reasons.append(f"策略约束未通过：{check.get('name', '未命名约束')}")
    return GateResult(
        passed=not reasons,
        checks=checks,
        policy_checks=policy_checks,
        reasons=reasons,
    )


def _normalize_index_frame(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    data = frame.rename(columns=aliases).copy()
    required = ["date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"指数行情缺少字段：{','.join(missing)}")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in data:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "open", "close"])
    data = data[(data["open"] > 0) & (data["close"] > 0)]
    return (
        data[["date", "open", "high", "low", "close", "volume", "amount"]]
        .sort_values("date", kind="stable")
        .drop_duplicates("date", keep="first")
    )


def _fetch_index_eastmoney(code: str, start: str, end: str) -> pd.DataFrame:
    session = requests.Session()
    session.trust_env = False
    response = session.get(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        params={
            "secid": INDEX_SECIDS[code],
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "1",
            "beg": start.replace("-", ""),
            "end": end.replace("-", ""),
        },
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    klines = ((payload.get("data") or {}).get("klines") or [])
    rows = []
    for line in klines:
        values = line.split(",")
        rows.append(
            {
                "date": values[0],
                "open": values[1],
                "close": values[2],
                "high": values[3],
                "low": values[4],
                "volume": values[5],
                "amount": values[6],
            }
        )
    return _normalize_index_frame(pd.DataFrame(rows))


def _fetch_index_tencent(code: str, start: str, end: str) -> pd.DataFrame:
    symbol = f"sh{code}"
    session = requests.Session()
    session.trust_env = False
    response = session.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": f"{symbol},day,{start},{end},2000,qfq"},
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(
            f"Tencent index request failed: {payload.get('msg') or payload.get('code')}"
        )
    index_data = (payload.get("data") or {}).get(symbol) or {}
    daily_rows = index_data.get("qfqday") or index_data.get("day") or []
    rows = []
    for values in daily_rows:
        if len(values) < 6:
            continue
        rows.append(
            {
                "date": values[0],
                "open": values[1],
                "close": values[2],
                "high": values[3],
                "low": values[4],
                "volume": values[5],
                "amount": 0.0,
            }
        )
    if not rows:
        raise RuntimeError("Tencent index response has no daily bars")
    return _normalize_index_frame(pd.DataFrame(rows))


def _fetch_index_fallback(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    try:
        frame = ak.stock_zh_index_daily_em(
            symbol=f"csi{code}", start_date=start.replace("-", ""), end_date=end.replace("-", "")
        )
    except Exception:
        frame = ak.stock_zh_index_daily(symbol=f"sh{code}")
    return _normalize_index_frame(frame)


def fetch_index_bars(code: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    if code not in INDEX_SECIDS:
        raise ValueError(f"不支持的指数：{code}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{code}.csv"
    cached = _normalize_index_frame(pd.read_csv(cache_path)) if cache_path.exists() else pd.DataFrame()
    requested_end = pd.Timestamp(end)
    if not cached.empty and cached["date"].max() >= requested_end:
        return cached[(cached["date"] >= pd.Timestamp(start)) & (cached["date"] <= requested_end)].reset_index(drop=True)
    errors: list[str] = []
    best: pd.DataFrame | None = None
    best_key: tuple[pd.Timestamp, int] | None = None
    for fetcher in (
        _fetch_index_eastmoney,
        _fetch_index_tencent,
        _fetch_index_fallback,
    ):
        fetcher_name = getattr(fetcher, "__name__", type(fetcher).__name__)
        try:
            fresh = fetcher(code, start, end)
            merged = (
                pd.concat([fresh, cached], ignore_index=True)
                if not cached.empty
                else fresh
            )
            merged = _normalize_index_frame(merged)
            if len(merged) < 120:
                errors.append(
                    f"{fetcher_name}: only {len(merged)} valid rows after cache merge"
                )
                continue
            candidate_key = (merged["date"].max(), len(merged))
            if best_key is None or candidate_key > best_key:
                best = merged
                best_key = candidate_key
        except Exception as exc:
            errors.append(f"{fetcher_name}: {exc}")
    if best is None:
        if len(cached) >= 120:
            return cached[
                (cached["date"] >= pd.Timestamp(start))
                & (cached["date"] <= requested_end)
            ].reset_index(drop=True)
        raise RuntimeError("指数行情获取失败：" + " | ".join(errors))
    temporary = cache_path.with_suffix(".csv.tmp")
    best.assign(date=best["date"].dt.date.astype(str)).to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, cache_path)
    return best[
        (best["date"] >= pd.Timestamp(start)) & (best["date"] <= requested_end)
    ].reset_index(drop=True)


def _series_metrics(returns: pd.Series, hold: int, source: dict | None = None) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").dropna().reset_index(drop=True)
    positives = clean.clip(lower=0)
    positive_sum = float(positives.sum())
    top_share = float(positives.max() / positive_sum) if positive_sum > 0 else 1.0
    groups = np.array_split(positives.to_numpy(), min(4, max(1, len(positives))))
    regime_totals = [float(np.sum(group)) for group in groups]
    best_regime = max(regime_totals) / positive_sum if positive_sum > 0 else 1.0
    source = source or {}
    return {
        "annualized_return": annualized_return(clean, hold),
        "mean_return_10": float(clean.mean()) if not clean.empty else 0.0,
        "max_drawdown": max_drawdown(clean),
        "mean_return_5": float(source.get("mean_return_5", 0.0)),
        "mean_return_20": float(source.get("mean_return_20", 0.0)),
        "top_contributor_share": top_share,
        "best_regime_share": float(best_regime),
        "periods": int(len(clean)),
        "win_rate": float((clean > 0).mean()) if not clean.empty else 0.0,
    }


def _aligned_metrics(profile_metrics: dict, index_bars: pd.DataFrame, hold: int = 10) -> tuple[dict, pd.DataFrame]:
    signals = pd.DataFrame(
        {
            "signal_date": profile_metrics.get("selection_dates", []),
            "strategy_return": profile_metrics.get("period_returns_10", []),
        }
    )
    aligned = periodic_returns(signals, index_bars, hold)
    return _series_metrics(aligned.get("strategy_return", pd.Series(dtype=float)), hold, profile_metrics), aligned


def _benchmark_metrics(aligned: pd.DataFrame, hold: int = 10) -> dict:
    return _series_metrics(aligned.get("index_return", pd.Series(dtype=float)), hold)


def _fold_results(folds: list[dict], index_bars: pd.DataFrame) -> list[dict[str, object]]:
    output = []
    for fold in folds:
        metrics = fold.get("validation_metrics", {})
        if metrics.get("policy") and "trades" in metrics:
            strategy, aligned = _policy_aligned_metrics(metrics, index_bars)
        else:
            strategy, aligned = _aligned_metrics(metrics, index_bars)
        benchmark = _policy_benchmark_metrics(aligned) if metrics.get("policy") else _benchmark_metrics(aligned)
        output.append(
            {
                "train_end": fold.get("train_end"),
                "train_max_date": fold.get("train_max_date"),
                "validation_min_date": fold.get("validation_min_date"),
                "validation_max_date": fold.get("validation_max_date"),
                "strategy_annualized": strategy["annualized_return"],
                "index_annualized": benchmark["annualized_return"],
                "beat_000300": strategy["annualized_return"] > benchmark["annualized_return"],
            }
        )
    return output


def _policy_aligned_metrics(profile_metrics: dict, index_bars: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Align policy cohorts, including weak-regime cash, to identical index intervals."""
    bars = index_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    bars = bars.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    positions = {day: index for index, day in enumerate(bars["date"])}
    trades = pd.DataFrame(profile_metrics.get("trades", []))
    rows = []
    for signal_text, codes in profile_metrics.get("selections", {}).items():
        signal = pd.Timestamp(signal_text).normalize()
        signal_pos = positions.get(signal)
        if signal_pos is None:
            continue
        matching = trades[trades.get("signal_date", pd.Series(dtype=str)).astype(str) == signal_text] if not trades.empty else pd.DataFrame()
        if not matching.empty:
            entry_date = pd.Timestamp(matching.iloc[0]["entry_date"]).normalize()
            exit_date = pd.Timestamp(matching.iloc[0]["cohort_end_date"]).normalize()
            strategy_return = float(pd.to_numeric(matching["net_return"], errors="coerce").mean())
        else:
            entry_pos = signal_pos + 1
            exit_pos = min(signal_pos + 5, len(bars) - 1)
            if entry_pos >= len(bars) or exit_pos <= entry_pos:
                continue
            entry_date = bars.iloc[entry_pos]["date"]
            exit_date = bars.iloc[exit_pos]["date"]
            strategy_return = 0.0
        entry_pos = positions.get(entry_date)
        exit_pos = positions.get(exit_date)
        if entry_pos is None or exit_pos is None or exit_pos <= entry_pos:
            continue
        entry = float(bars.iloc[entry_pos]["open"])
        exit_price = float(bars.iloc[exit_pos]["close"])
        rows.append(
            {
                "signal_date": signal.date().isoformat(),
                "entry_date": entry_date.date().isoformat(),
                "exit_date": exit_date.date().isoformat(),
                "strategy_return": strategy_return,
                "index_return": exit_price / entry - 1.0,
                "trading_days": exit_pos - entry_pos,
            }
        )
    aligned = pd.DataFrame(rows)
    strategy = {
        key: profile_metrics[key]
        for key in (
            "annualized_return", "mean_return_10", "max_drawdown", "mean_return_5",
            "mean_return_20", "top_contributor_share", "best_regime_share", "periods",
            "win_rate",
        )
        if key in profile_metrics
    }
    return strategy, aligned


def _policy_benchmark_metrics(aligned: pd.DataFrame) -> dict:
    returns = pd.to_numeric(aligned.get("index_return", pd.Series(dtype=float)), errors="coerce").dropna()
    durations = pd.to_numeric(aligned.get("trading_days", pd.Series(dtype=float)), errors="coerce").fillna(0)
    total_days = max(1.0, float(durations.sum()))
    wealth = float(np.prod(1.0 + returns.to_numpy(dtype="float64"))) if not returns.empty else 1.0
    annualized = float(wealth ** (252.0 / total_days) - 1.0) if wealth > 0 else -1.0
    metrics = _series_metrics(returns, hold=10)
    metrics["annualized_return"] = annualized
    return metrics


def _candidate_policy_checks(evidence: dict) -> list[dict[str, object]]:
    participation = evidence.get("participation", {})
    checks = list(participation.get("checks", []))
    neighbor = evidence.get("neighbor_stability", {})
    checks.append(
        {
            "name": "相邻止盈止损参数保持稳定",
            "passed": bool(neighbor.get("passed")),
            "value": {
                "stable_neighbors": neighbor.get("stable_neighbors", 0),
                "required_neighbors": neighbor.get("required_neighbors", 1),
            },
        }
    )
    return checks


def _overall_benchmark(bars: pd.DataFrame) -> dict[str, float]:
    close = pd.to_numeric(bars["close"], errors="coerce").dropna()
    daily = close.pct_change(fill_method=None).dropna()
    years = max(len(daily) / 252.0, 1 / 252.0)
    annualized = float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) if len(close) > 1 else 0.0
    wealth = close / close.iloc[0] if len(close) else pd.Series(dtype=float)
    drawdown = float((wealth / wealth.cummax() - 1).min()) if not wealth.empty else 0.0
    return {"annualized_return": annualized, "max_drawdown": drawdown, "rows": int(len(bars))}


def run_backtest(
    paths: AppPaths,
    emit: Emitter,
    *,
    start: str = "2022-01-01",
    end: str | None = None,
    refresh: bool = False,
    source_path: Path | None = None,
) -> dict[str, object]:
    end = end or date.today().isoformat()
    cache_dir = Path(r"C:\InStockLocal\backtest_cache\indices")
    emit("progress", "正在加载沪深300指数", 10, {"benchmark": "000300"})
    hs300 = fetch_index_bars("000300", start, end, cache_dir)
    emit("progress", "正在加载中证500指数", 20, {"benchmark": "000905"})
    zz500 = fetch_index_bars("000905", start, end, cache_dir)

    source_path = source_path or paths.reports_dir / "picker_backtest_results.json"
    engine_dir = paths.engine_dir if paths.engine_dir.exists() else Path(__file__).resolve().parents[1] / "resources" / "engine"
    if refresh or not source_path.exists():
        command = [
            str(paths.python_exe),
            str(engine_dir / "picker_backtest.py"),
            "optimize",
            "--start",
            start,
            "--end",
            end,
            "--profiles",
            str(engine_dir / "picker_factor_profiles.json"),
            "--output",
            str(source_path),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        completed = subprocess.run(command, cwd=engine_dir, creationflags=creationflags, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"滚动回测引擎失败，退出码 {completed.returncode}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    profiles: dict[str, object] = {}
    for index, (key, evidence) in enumerate(source.get("profiles", {}).items(), start=1):
        emit("progress", f"正在评估 {key} 的指数门槛", 20 + index * 20, {"strategy": key})
        candidate_source = evidence["candidate"]
        baseline_source = evidence["baseline"]
        policy_aware = bool(candidate_source.get("policy") and "trades" in candidate_source)
        if policy_aware:
            candidate, aligned_300 = _policy_aligned_metrics(candidate_source, hs300)
            baseline, _ = _policy_aligned_metrics(baseline_source, hs300)
        else:
            candidate, aligned_300 = _aligned_metrics(candidate_source, hs300)
            baseline, _ = _aligned_metrics(baseline_source, hs300)
        benchmark_300 = _policy_benchmark_metrics(aligned_300) if policy_aware else _benchmark_metrics(aligned_300)
        if policy_aware:
            _, aligned_500 = _policy_aligned_metrics(candidate_source, zz500)
        else:
            _, aligned_500 = _aligned_metrics(candidate_source, zz500)
        benchmark_500 = _policy_benchmark_metrics(aligned_500) if policy_aware else _benchmark_metrics(aligned_500)
        fold_items = _fold_results(evidence.get("walk_forward", {}).get("folds", []), hs300)
        policy_checks = _candidate_policy_checks(evidence) if policy_aware else []
        gate = evaluate_gate(candidate, baseline, benchmark_300, fold_items, policy_checks=policy_checks)
        profiles[key] = {
            "strategy": candidate,
            "baseline": baseline,
            "benchmark_000300": benchmark_300,
            "benchmark_000905": benchmark_500,
            "folds": fold_items,
            "gate": gate.to_dict(),
            "policy": candidate_source.get("policy"),
            "market_policy": candidate_source.get("market_policy"),
            "chosen_weights": evidence.get("chosen_weights"),
            "participation": evidence.get("participation"),
            "neighbor_stability": evidence.get("neighbor_stability"),
            "trades": candidate_source.get("trades", []),
            "selections": candidate_source.get("selections", {}),
            "curve": {
                "dates": aligned_300.get("exit_date", pd.Series(dtype=str)).tolist(),
                "strategy": (1 + aligned_300.get("strategy_return", pd.Series(dtype=float))).cumprod().tolist(),
                "000300": (1 + aligned_300.get("index_return", pd.Series(dtype=float))).cumprod().tolist(),
                "000905": (1 + aligned_500.get("index_return", pd.Series(dtype=float))).cumprod().tolist(),
            },
            "intervals": aligned_300.to_dict("records"),
        }
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start": start,
        "end": end,
        "data_dates": {
            "000300": hs300["date"].max().date().isoformat(),
            "000905": zz500["date"].max().date().isoformat(),
        },
        "benchmarks": {"000300": _overall_benchmark(hs300), "000905": _overall_benchmark(zz500)},
        "profiles": profiles,
        "disclaimer": "历史样本外结果不代表未来收益，不构成投资建议。",
    }
    emit(
        "result",
        "指数基准回测完成",
        100,
        {
            "data_dates": result["data_dates"],
            "gates": {key: value["gate"] for key, value in profiles.items()},
        },
    )
    return result
