from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from high_precision_picker import save_json_atomic
from picker_backtest import max_drawdown


def calculate_bounded_trade_returns(
    panel: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    cost: float,
) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["_position"] = np.arange(len(data))
    data = data.sort_values(["code", "date", "_position"])
    grouped = data.groupby("code", sort=False)
    entry = pd.to_numeric(grouped["open"].shift(-1), errors="coerce")
    result = pd.Series(np.nan, index=data.index, dtype="float64")
    reason = pd.Series("", index=data.index, dtype="object")
    exit_day = pd.Series(np.nan, index=data.index, dtype="float64")
    alive = entry.notna() & (entry > 0)
    for day_number in range(1, 6):
        high = pd.to_numeric(grouped["high"].shift(-day_number), errors="coerce")
        low = pd.to_numeric(grouped["low"].shift(-day_number), errors="coerce")
        stopped = alive & (low <= entry * (1.0 - float(stop_loss)))
        taken = (
            alive
            & ~stopped
            & (high >= entry * (1.0 + float(take_profit)))
        )
        result.loc[stopped] = -float(stop_loss) - float(cost)
        reason.loc[stopped] = "stop_loss"
        exit_day.loc[stopped] = day_number
        result.loc[taken] = float(take_profit) - float(cost)
        reason.loc[taken] = "take_profit"
        exit_day.loc[taken] = day_number
        alive &= ~(stopped | taken)
    close_5 = pd.to_numeric(grouped["close"].shift(-5), errors="coerce")
    fallback = alive & close_5.notna() & (close_5 > 0)
    result.loc[fallback] = close_5[fallback] / entry[fallback] - 1.0 - float(cost)
    reason.loc[fallback] = "day_5_close"
    exit_day.loc[fallback] = 5
    output = pd.DataFrame(
        {
            "_position": data["_position"].to_numpy(),
            "bounded_return_5": result.to_numpy(),
            "exit_reason": reason.to_numpy(),
            "exit_day": exit_day.to_numpy(),
        }
    ).sort_values("_position")
    return output.drop(columns="_position").reset_index(drop=True)


def build_pullback_scores(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for factor in ("mom_60", "trend_20_60", "mom_5", "overheat"):
        data[f"{factor}_rank"] = data.groupby("date")[factor].rank(
            pct=True, method="average"
        ).fillna(0.5)
    data["pullback_score"] = (
        data["mom_60_rank"]
        + data["trend_20_60_rank"]
        - data["mom_5_rank"]
        - data["overheat_rank"]
    )
    return data


def select_pullback_candidates(panel: pd.DataFrame, rule: dict) -> pd.DataFrame:
    data = panel.copy()
    market = (
        (pd.to_numeric(data["breadth_above_ma20"], errors="coerce") >= float(rule["breadth_floor"]))
        & (pd.to_numeric(data["market_median_mom20"], errors="coerce") >= float(rule["momentum_floor"]))
        & (pd.to_numeric(data["market_median_drawdown60"], errors="coerce") >= float(rule["drawdown_floor"]))
    )
    selected = data[market].sort_values(
        ["date", "pullback_score", "code"],
        ascending=[True, False, True],
    )
    return selected.groupby("date", sort=True).head(
        int(rule.get("max_picks", 1))
    ).reset_index(drop=True)


def evaluate_v3(
    selected: pd.DataFrame,
    scheduled_dates,
    position_fraction: float,
) -> dict:
    data = selected.dropna(subset=["bounded_return_5"]).copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    returns = pd.to_numeric(data["bounded_return_5"], errors="coerce")
    years = []
    for year, group in data.groupby(data["date"].dt.year):
        values = pd.to_numeric(group["bounded_return_5"], errors="coerce")
        years.append(
            {
                "year": int(year),
                "trades": int(len(values)),
                "wins": int((values > 0).sum()),
                "win_rate": float((values > 0).mean()),
                "mean_net_return": float(values.mean()),
            }
        )
    scheduled_count = len(pd.DatetimeIndex(scheduled_dates).unique())
    active_dates = int(data["date"].nunique())
    return {
        "scheduled_dates": int(scheduled_count),
        "active_dates": active_dates,
        "activity_rate": active_dates / max(scheduled_count, 1),
        "trades": int(len(returns)),
        "wins": int((returns > 0).sum()),
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "mean_net_return": float(returns.mean()) if len(returns) else 0.0,
        "max_position_drawdown": max_drawdown(
            returns * float(position_fraction)
        ),
        "year_metrics": years,
        "exit_reasons": {
            str(key): int(value)
            for key, value in data.get(
                "exit_reason", pd.Series(dtype="object")
            ).value_counts().items()
        },
    }


def v3_deployment_checks(metrics: dict, neighbors: list[dict], gates: dict) -> dict:
    years = metrics.get("year_metrics", [])
    good_years = sum(
        float(item["win_rate"]) >= float(gates["minimum_year_win_rate"])
        for item in years
    )
    worst_year = min(
        (float(item["win_rate"]) for item in years), default=0.0
    )
    positive_neighbors = sum(
        float(item.get("win_rate", 0.0)) >= 0.60
        and float(item.get("mean_net_return", 0.0)) > 0.0
        for item in neighbors
    )
    checks = {
        "aggregate_win_rate": float(metrics["win_rate"]) >= float(gates["minimum_win_rate"]),
        "trade_count": int(metrics["trades"]) >= int(gates["minimum_trades"]),
        "activity_rate": float(metrics["activity_rate"]) >= float(gates["minimum_activity_rate"]),
        "mean_net_return": float(metrics["mean_net_return"]) > float(gates["minimum_mean_net_return"]),
        "maximum_position_drawdown": abs(float(metrics["max_position_drawdown"])) <= float(gates["maximum_position_drawdown"]),
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
    }


def _records(frame: pd.DataFrame) -> list[dict]:
    keep = [
        "date", "code", "name", "industry", "pullback_score",
        "bounded_return_5", "exit_reason", "exit_day",
    ]
    output = []
    for row in frame[[column for column in keep if column in frame]].to_dict("records"):
        item = {}
        for key, value in row.items():
            if isinstance(value, (pd.Timestamp, datetime, date)):
                item[key] = pd.Timestamp(value).date().isoformat()
            elif isinstance(value, np.generic):
                item[key] = value.item()
            elif pd.isna(value):
                item[key] = None
            else:
                item[key] = value
        output.append(item)
    return output


def _evaluate_variant(scored: pd.DataFrame, config: dict, variant: dict) -> dict:
    data = scored.copy()
    outcomes = calculate_bounded_trade_returns(
        data,
        float(variant["take_profit_pct"]),
        float(variant["stop_loss_pct"]),
        float(config["round_trip_cost"]),
    )
    data[["bounded_return_5", "exit_reason", "exit_day"]] = outcomes
    rule = {**config["rule"], **variant.get("rule", {})}
    selected = select_pullback_candidates(data, rule)
    dates = sorted(data["date"].dropna().unique())
    metrics = evaluate_v3(selected, dates, float(config["position_fraction"]))
    return {**variant, **metrics}


def run_v3_backtest(panel: pd.DataFrame, config: dict) -> tuple[dict, dict]:
    raw = panel.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    scored = build_pullback_scores(raw)
    base_variant = {
        "take_profit_pct": float(config["take_profit_pct"]),
        "stop_loss_pct": float(config["stop_loss_pct"]),
        "rule": {},
    }
    base = _evaluate_variant(scored, config, base_variant)
    variants = [
        {"kind": "take_profit", "take_profit_pct": 0.025, "stop_loss_pct": 0.06, "rule": {}},
        {"kind": "take_profit", "take_profit_pct": 0.035, "stop_loss_pct": 0.06, "rule": {}},
        {"kind": "stop_loss", "take_profit_pct": 0.03, "stop_loss_pct": 0.05, "rule": {}},
        {"kind": "stop_loss", "take_profit_pct": 0.03, "stop_loss_pct": 0.07, "rule": {}},
        {"kind": "breadth", "take_profit_pct": 0.03, "stop_loss_pct": 0.06, "rule": {"breadth_floor": 0.45}},
        {"kind": "breadth", "take_profit_pct": 0.03, "stop_loss_pct": 0.06, "rule": {"breadth_floor": 0.55}},
    ]
    neighbors = [_evaluate_variant(scored, config, variant) for variant in variants]
    deployment = v3_deployment_checks(
        base, neighbors, config["deployment_gates"]
    )
    run_id = datetime.now().strftime("hpv3-%Y%m%d-%H%M%S")
    outcomes = calculate_bounded_trade_returns(
        scored,
        float(config["take_profit_pct"]),
        float(config["stop_loss_pct"]),
        float(config["round_trip_cost"]),
    )
    scored[["bounded_return_5", "exit_reason", "exit_day"]] = outcomes
    selected = select_pullback_candidates(scored, config["rule"])
    report = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": "high-precision-v3",
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
        "validation_selections": _records(selected),
        "caveats": [
            "The fixed rule was selected from the available historical sample and is exposed to data-mining bias.",
            "Daily bars cannot reveal intraday order, so same-day take-profit and stop-loss conflicts are conservatively counted as stop-loss.",
        ],
    }
    model = {
        "schema_version": "high-precision-v3",
        "run_id": run_id,
        "rule": config["rule"],
        "historical_win_rate": float(base["win_rate"]),
        "trade_policy": report["trade_policy"],
        "deployment": deployment,
    }
    return report, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InStock practical high-win V3")
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize")
    optimize.add_argument("--config", type=Path, default=Path(__file__).with_name("high_precision_v3_config.json"))
    optimize.add_argument("--panel-cache", type=Path, default=Path(r"C:\InStockLocal\backtest_cache\policy_panel_600_20260714.pkl"))
    optimize.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "data" / "high_precision-v3-backtest-latest.json")
    optimize.add_argument("--candidate-model", type=Path, default=Path(__file__).parent.parent / "work" / "candidate-high_precision_v3_model.json")
    optimize.add_argument("--verification", type=Path, default=Path(__file__).parent.parent / "verification" / "high-precision-v3-2026-07-17.json")
    optimize.add_argument("--production-model", type=Path, default=Path(__file__).with_name("high_precision_model.json"))
    optimize.add_argument("--runtime-report", type=Path, default=Path(__file__).parent.parent / "data" / "high_precision-backtest-latest.json")
    return parser


def run_cli(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cached = pd.read_pickle(args.panel_cache)
    panel = cached["panel"] if isinstance(cached, dict) else cached
    report, model = run_v3_backtest(panel, config)
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
    print(json.dumps({"run_id": report["run_id"], "aggregate_metrics": report["aggregate_metrics"], "deployment": report["deployment"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
