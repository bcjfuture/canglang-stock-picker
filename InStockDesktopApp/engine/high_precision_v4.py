from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from high_precision_picker import save_json_atomic
from high_precision_v3 import calculate_bounded_trade_returns
from picker_backtest import max_drawdown


def _passes_market(row: pd.Series, specification: dict) -> bool:
    return bool(
        float(row["breadth_above_ma20"]) >= float(specification["breadth_floor"])
        and float(row["market_median_mom20"]) >= float(specification["momentum_floor"])
        and float(row["market_median_drawdown60"]) >= float(specification["drawdown_floor"])
    )


def classify_market_regime(row: pd.Series, rule: dict) -> str:
    if _passes_market(row, rule["good"]):
        return "good"
    if _passes_market(row, rule["normal"]):
        return "normal"
    return "poor"


def build_regime_scores(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    rank_columns = (
        "mom_5", "mom_20", "mom_60", "trend_20_60",
        "overheat", "drawdown_60", "volatility_20",
    )
    for factor in rank_columns:
        data[f"{factor}_rank"] = data.groupby("date")[factor].rank(
            pct=True, method="average"
        ).fillna(0.5)
    data["pullback_score"] = (
        data["mom_60_rank"]
        + data["trend_20_60_rank"]
        - data["mom_5_rank"]
        - data["overheat_rank"]
    )
    data["low_vol_trend_score"] = (
        data["mom_20_rank"]
        + data["mom_60_rank"]
        + data["trend_20_60_rank"]
        - data["volatility_20_rank"]
    )
    data["rebound_score"] = (
        -data["mom_5_rank"]
        - data["drawdown_60_rank"]
        + data["trend_20_60_rank"]
        - data["overheat_rank"]
    )
    return data


def select_dynamic_candidates(panel: pd.DataFrame, rule: dict) -> pd.DataFrame:
    data = panel.copy()
    good = (
        (pd.to_numeric(data["breadth_above_ma20"], errors="coerce") >= float(rule["good"]["breadth_floor"]))
        & (pd.to_numeric(data["market_median_mom20"], errors="coerce") >= float(rule["good"]["momentum_floor"]))
        & (pd.to_numeric(data["market_median_drawdown60"], errors="coerce") >= float(rule["good"]["drawdown_floor"]))
    )
    normal = (
        ~good
        & (pd.to_numeric(data["breadth_above_ma20"], errors="coerce") >= float(rule["normal"]["breadth_floor"]))
        & (pd.to_numeric(data["market_median_mom20"], errors="coerce") >= float(rule["normal"]["momentum_floor"]))
        & (pd.to_numeric(data["market_median_drawdown60"], errors="coerce") >= float(rule["normal"]["drawdown_floor"]))
    )
    data["market_regime"] = np.where(good, "good", np.where(normal, "normal", "poor"))
    score_names = {
        regime: str(specification["score"])
        for regime, specification in rule.items()
    }
    target_counts = {
        regime: int(specification["target_count"])
        for regime, specification in rule.items()
    }
    data["selection_score_name"] = data["market_regime"].map(score_names)
    data["target_count"] = data["market_regime"].map(target_counts).astype(int)
    data["selection_score"] = np.select(
        [
            data["market_regime"].eq("good"),
            data["market_regime"].eq("normal"),
        ],
        [data[score_names["good"]], data[score_names["normal"]]],
        default=data[score_names["poor"]],
    )
    data["code"] = data["code"].astype(str).str.zfill(6)
    ranked = data.sort_values(
        ["date", "selection_score", "code"],
        ascending=[True, False, True],
    )
    position = ranked.groupby("date", sort=True).cumcount()
    return ranked[position < ranked["target_count"]].reset_index(drop=True)


def _group_metrics(group: pd.DataFrame) -> dict:
    returns = pd.to_numeric(group["bounded_return_5"], errors="coerce")
    return {
        "trades": int(len(returns)),
        "wins": int((returns > 0).sum()),
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "mean_net_return": float(returns.mean()) if len(returns) else 0.0,
    }


def evaluate_v4(
    selected: pd.DataFrame,
    scheduled_dates,
    position_fraction: float,
) -> dict:
    data = selected.dropna(subset=["bounded_return_5"]).copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    aggregate = _group_metrics(data)
    years = []
    for year, group in data.groupby(data["date"].dt.year):
        years.append({"year": int(year), **_group_metrics(group)})
    regimes = []
    for regime in ("good", "normal", "poor"):
        group = data[data["market_regime"].eq(regime)]
        regimes.append({"market_regime": regime, **_group_metrics(group)})
    contributions = (
        pd.to_numeric(data["bounded_return_5"], errors="coerce")
        * float(position_fraction)
    )
    signal_returns = data.assign(_contribution=contributions).groupby(
        "date", sort=True
    )["_contribution"].sum()
    scheduled_count = len(pd.DatetimeIndex(scheduled_dates).unique())
    return {
        **aggregate,
        "scheduled_dates": int(scheduled_count),
        "active_dates": int(data["date"].nunique()),
        "activity_rate": int(data["date"].nunique()) / max(scheduled_count, 1),
        "max_signal_basket_drawdown": max_drawdown(signal_returns),
        "year_metrics": years,
        "regime_metrics": regimes,
        "exit_reasons": {
            str(key): int(value)
            for key, value in data.get(
                "exit_reason", pd.Series(dtype="object")
            ).value_counts().items()
        },
    }


def v4_deployment_checks(metrics: dict, neighbors: list[dict], gates: dict) -> dict:
    years = metrics.get("year_metrics", [])
    regimes = metrics.get("regime_metrics", [])
    good_years = sum(
        float(item["win_rate"]) >= float(gates["minimum_year_win_rate"])
        for item in years
    )
    worst_year = min(
        (float(item["win_rate"]) for item in years), default=0.0
    )
    regime_means = {
        str(item["market_regime"]): float(item["mean_net_return"])
        for item in regimes
    }
    all_regimes_positive = (
        set(regime_means) == {"good", "normal", "poor"}
        and all(value > 0.0 for value in regime_means.values())
    )
    positive_neighbors = sum(
        float(item.get("win_rate", 0.0)) >= 0.65
        and float(item.get("mean_net_return", 0.0)) > 0.0
        for item in neighbors
    )
    checks = {
        "aggregate_win_rate": float(metrics["win_rate"]) >= float(gates["minimum_win_rate"]),
        "trade_count": int(metrics["trades"]) >= int(gates["minimum_trades"]),
        "mean_net_return": float(metrics["mean_net_return"]) > float(gates["minimum_mean_net_return"]),
        "maximum_signal_basket_drawdown": abs(float(metrics["max_signal_basket_drawdown"])) <= float(gates["maximum_signal_basket_drawdown"]),
        "all_regimes_positive": all_regimes_positive,
        "good_year_count": good_years >= int(gates["minimum_good_years"]),
        "worst_year_win_rate": worst_year >= float(gates["minimum_worst_year_win_rate"]),
        "positive_neighbors": positive_neighbors >= int(gates["minimum_positive_neighbors"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "good_years": int(good_years),
        "worst_year_win_rate": float(worst_year),
        "positive_neighbors": int(positive_neighbors),
        "regime_means": regime_means,
    }


def _evaluate_variant(scored: pd.DataFrame, config: dict, variant: dict) -> dict:
    data = scored.copy().reset_index(drop=True)
    take_profit = float(variant.get("take_profit_pct", config["take_profit_pct"]))
    stop_loss = float(variant.get("stop_loss_pct", config["stop_loss_pct"]))
    outcomes = calculate_bounded_trade_returns(
        data, take_profit, stop_loss, float(config["round_trip_cost"])
    )
    data[["bounded_return_5", "exit_reason", "exit_day"]] = outcomes
    rule = json.loads(json.dumps(config["rule"]))
    if "normal_breadth_floor" in variant:
        rule["normal"]["breadth_floor"] = float(variant["normal_breadth_floor"])
    if "normal_momentum_floor" in variant:
        rule["normal"]["momentum_floor"] = float(variant["normal_momentum_floor"])
    selected = select_dynamic_candidates(data, rule)
    metrics = evaluate_v4(
        selected,
        sorted(data["date"].dropna().unique()),
        float(config["position_fraction"]),
    )
    return {
        **variant,
        "take_profit_pct": take_profit,
        "stop_loss_pct": stop_loss,
        **metrics,
    }


def _records(frame: pd.DataFrame) -> list[dict]:
    keep = [
        "date", "code", "name", "industry", "market_regime",
        "target_count", "selection_score_name", "selection_score",
        "bounded_return_5", "exit_reason", "exit_day",
    ]
    available = [column for column in keep if column in frame]
    output = []
    for item in frame.loc[:, available].to_dict("records"):
        for key, value in list(item.items()):
            if isinstance(value, (pd.Timestamp, datetime)):
                item[key] = value.date().isoformat()
            elif pd.isna(value):
                item[key] = None
            elif isinstance(value, np.generic):
                item[key] = value.item()
        output.append(item)
    return output


def run_v4_backtest(panel: pd.DataFrame, config: dict) -> tuple[dict, dict]:
    raw = panel.copy().reset_index(drop=True)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    scored = build_regime_scores(raw)
    base_variant = {
        "kind": "base",
        "take_profit_pct": float(config["take_profit_pct"]),
        "stop_loss_pct": float(config["stop_loss_pct"]),
    }
    base = _evaluate_variant(scored, config, base_variant)
    neighbor_variants = [
        {"kind": "stop_loss", "stop_loss_pct": 0.08},
        {"kind": "stop_loss", "stop_loss_pct": 0.10},
        {"kind": "normal_breadth", "normal_breadth_floor": 0.43},
        {"kind": "normal_breadth", "normal_breadth_floor": 0.47},
        {"kind": "normal_momentum", "normal_momentum_floor": 0.0},
        {"kind": "normal_momentum", "normal_momentum_floor": 0.01},
    ]
    neighbors = [
        _evaluate_variant(scored, config, variant)
        for variant in neighbor_variants
    ]
    deployment = v4_deployment_checks(
        base, neighbors, config["deployment_gates"]
    )
    outcomes = calculate_bounded_trade_returns(
        scored,
        float(config["take_profit_pct"]),
        float(config["stop_loss_pct"]),
        float(config["round_trip_cost"]),
    )
    scored[["bounded_return_5", "exit_reason", "exit_day"]] = outcomes
    selected = select_dynamic_candidates(scored, config["rule"])
    validated = selected.dropna(subset=["bounded_return_5"]).copy()
    run_id = datetime.now().strftime("hpv4-%Y%m%d-%H%M%S")
    report = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": "high-precision-v4",
        "rule": config["rule"],
        "trade_policy": {
            "holding_days": int(config["holding_days"]),
            "take_profit_pct": float(config["take_profit_pct"]),
            "stop_loss_pct": float(config["stop_loss_pct"]),
            "position_fraction": float(config["position_fraction"]),
            "round_trip_cost": float(config["round_trip_cost"]),
            "same_day_priority": "stop_loss",
        },
        "coverage": {
            "actual_start": raw["date"].min().date().isoformat(),
            "actual_end": raw["date"].max().date().isoformat(),
            "usable_symbols": int(raw["code"].nunique()),
            "stock_dates": int(len(raw)),
        },
        "aggregate_metrics": base,
        "neighbor_metrics": neighbors,
        "deployment": deployment,
        "validation_selections": _records(validated),
        "caveats": [
            "The thresholds and ranking formulas were selected from the available historical sample and are exposed to data-mining bias.",
            "Daily bars cannot reveal intraday order, so same-day take-profit and stop-loss conflicts are conservatively counted as stop-loss.",
            "Signal-basket drawdown does not fully model broker cash constraints or overlapping live positions.",
        ],
    }
    model = {
        "schema_version": "high-precision-v4",
        "run_id": run_id,
        "rule": config["rule"],
        "historical_win_rate": float(base["win_rate"]),
        "trade_policy": report["trade_policy"],
        "deployment": deployment,
    }
    return report, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InStock dynamic high-win V4")
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize")
    optimize.add_argument("--config", type=Path, default=Path(__file__).with_name("high_precision_v4_config.json"))
    optimize.add_argument("--panel-cache", type=Path, default=Path(r"C:\InStockLocal\backtest_cache\policy_panel_600_20260714.pkl"))
    optimize.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "data" / "high_precision-v4-backtest-latest.json")
    optimize.add_argument("--candidate-model", type=Path, default=Path(__file__).parent.parent / "work" / "candidate-high_precision_v4_model.json")
    optimize.add_argument("--verification", type=Path, default=Path(__file__).parent.parent / "verification" / "high-precision-v4-2026-07-17.json")
    optimize.add_argument("--production-model", type=Path, default=Path(__file__).with_name("high_precision_model.json"))
    optimize.add_argument("--runtime-report", type=Path, default=Path(__file__).parent.parent / "data" / "high_precision-backtest-latest.json")
    return parser


def run_cli(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cached = pd.read_pickle(args.panel_cache)
    panel = cached["panel"] if isinstance(cached, dict) else cached
    report, model = run_v4_backtest(panel, config)
    verification = {
        "run_id": report["run_id"],
        "aggregate_metrics": report["aggregate_metrics"],
        "neighbor_metrics": report["neighbor_metrics"],
        "deployment": report["deployment"],
    }
    save_json_atomic(args.output, report)
    save_json_atomic(args.candidate_model, model)
    save_json_atomic(args.verification, verification)
    if report["deployment"]["passed"]:
        save_json_atomic(args.production_model, model)
        save_json_atomic(args.runtime_report, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_cli(args)
    print(json.dumps({
        "run_id": report["run_id"],
        "aggregate_metrics": report["aggregate_metrics"],
        "deployment": report["deployment"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
