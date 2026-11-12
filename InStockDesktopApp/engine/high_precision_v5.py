from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from high_precision_picker import (
    apply_probability_calibrator,
    fit_logistic,
    fit_probability_calibrator,
    predict_raw_probability,
    save_json_atomic,
)
from picker_backtest import max_drawdown


HORIZONS = (1, 3, 5)
MODEL_FEATURES = (
    "mom_5_rank",
    "mom_20_rank",
    "mom_60_rank",
    "trend_20_60_rank",
    "drawdown_60_rank",
    "volatility_20_rank",
    "liquidity_20_rank",
    "overheat_rank",
)


def calculate_horizon_returns(
    bars: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    cost: float = 0.003,
) -> pd.DataFrame:
    """Calculate signal-close to next-open entry and horizon close net returns."""
    entry = pd.to_numeric(bars["open"], errors="coerce").shift(-1)
    output = pd.DataFrame(index=bars.index)
    for horizon in horizons:
        if int(horizon) < 1:
            raise ValueError("horizons must be positive")
        exit_price = pd.to_numeric(bars["close"], errors="coerce").shift(-int(horizon))
        result = exit_price / entry - 1.0 - float(cost)
        invalid = (
            ~np.isfinite(entry)
            | (entry <= 0)
            | ~np.isfinite(exit_price)
            | (exit_price <= 0)
        )
        output[f"return_{int(horizon)}"] = result.mask(invalid).astype("float64")
    return output


def attach_benchmark_returns(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    data = panel.copy()
    market = benchmark.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    columns = ["date"] + [f"benchmark_return_{int(h)}" for h in horizons]
    data = data.merge(market.loc[:, columns].drop_duplicates("date"), on="date", how="left")
    for horizon in horizons:
        data[f"excess_{int(horizon)}"] = (
            pd.to_numeric(data[f"return_{int(horizon)}"], errors="coerce")
            - pd.to_numeric(data[f"benchmark_return_{int(horizon)}"], errors="coerce")
        )
    return data


def _rank(data: pd.DataFrame, factor: str) -> pd.Series:
    if factor not in data:
        return pd.Series(0.5, index=data.index, dtype="float64")
    values = pd.to_numeric(data[factor], errors="coerce")
    return values.groupby(data["date"]).rank(pct=True, method="average").fillna(0.5)


def build_v5_scores(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    factors = (
        "mom_5",
        "mom_20",
        "mom_60",
        "trend_20_60",
        "drawdown_60",
        "volatility_20",
        "liquidity_20",
        "overheat",
    )
    for factor in factors:
        data[f"{factor}_rank"] = _rank(data, factor)
    data["growth_score"] = (
        0.2 * data["mom_5_rank"]
        + 1.2 * data["mom_20_rank"]
        + 1.0 * data["mom_60_rank"]
        + 1.1 * data["trend_20_60_rank"]
        + 0.6 * data["drawdown_60_rank"]
        + 0.4 * data["liquidity_20_rank"]
        - 0.8 * data["volatility_20_rank"]
        - 0.8 * data["overheat_rank"]
    )
    data["balanced_score"] = (
        0.1 * data["mom_5_rank"]
        + 1.0 * data["mom_20_rank"]
        + 0.7 * data["mom_60_rank"]
        + 1.0 * data["trend_20_60_rank"]
        + 1.0 * data["drawdown_60_rank"]
        + 0.5 * data["liquidity_20_rank"]
        - 1.3 * data["volatility_20_rank"]
        - 0.7 * data["overheat_rank"]
    )
    data["defensive_score"] = (
        0.2 * data["mom_5_rank"]
        + 0.8 * data["mom_20_rank"]
        + 0.4 * data["mom_60_rank"]
        + 0.8 * data["trend_20_60_rank"]
        + 1.6 * data["drawdown_60_rank"]
        + 0.4 * data["liquidity_20_rank"]
        - 2.0 * data["volatility_20_rank"]
        - 0.4 * data["overheat_rank"]
    )
    return data


def _assign_regimes(data: pd.DataFrame, rule: dict) -> pd.Series:
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
    return pd.Series(np.where(good, "good", np.where(normal, "normal", "poor")), index=data.index)


def fit_v5_regime_models(
    scored: pd.DataFrame,
    training_end: str,
    ridge: float = 10.0,
    feature_names: tuple[str, ...] = MODEL_FEATURES,
) -> dict:
    data = scored.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    train = data[data["date"] <= pd.Timestamp(training_end).normalize()].copy()
    output = {"feature_names": list(feature_names), "regime_models": {}}
    for regime in ("good", "normal", "poor"):
        target = "target_beat_rate" if regime == "poor" else "target_mean_excess"
        part = train[train["market_regime"].eq(regime)].dropna(
            subset=[*feature_names, target]
        )
        if part.empty:
            coefficients = np.zeros(len(feature_names), dtype="float64")
        else:
            matrix = part.loc[:, feature_names].to_numpy(dtype="float64") - 0.5
            labels = pd.to_numeric(part[target], errors="coerce").to_numpy(dtype="float64")
            coefficients = np.linalg.solve(
                matrix.T @ matrix + float(ridge) * np.eye(len(feature_names)),
                matrix.T @ labels,
            )
        output["regime_models"][regime] = {
            "target": target,
            "coefficients": [float(value) for value in coefficients],
            "training_rows": int(len(part)),
        }
    return output


def fit_v5_probability_models(
    scored: pd.DataFrame,
    *,
    fit_end: str,
    calibration_start: str,
    calibration_end: str,
    l2: float = 1.0,
    feature_names: tuple[str, ...] = MODEL_FEATURES,
    bins: int = 8,
) -> dict:
    data = scored.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["_positive_5d"] = (
        pd.to_numeric(data["return_5"], errors="coerce") > 0.0
    ).astype("float64")
    output = {
        "target": "return_5_positive_after_cost",
        "feature_names": list(feature_names),
        "fit_end": str(fit_end),
        "calibration_start": str(calibration_start),
        "calibration_end": str(calibration_end),
        "regimes": {},
    }
    fit_cutoff = pd.Timestamp(fit_end).normalize()
    calibration_floor = pd.Timestamp(calibration_start).normalize()
    calibration_cutoff = pd.Timestamp(calibration_end).normalize()
    for regime in ("good", "normal", "poor"):
        regime_rows = data[data["market_regime"].eq(regime)]
        fit_rows = regime_rows[regime_rows["date"] <= fit_cutoff].dropna(
            subset=[*feature_names, "return_5"]
        )
        calibration_rows = regime_rows[
            regime_rows["date"].between(
                calibration_floor, calibration_cutoff, inclusive="both"
            )
        ].dropna(subset=[*feature_names, "return_5"])
        if fit_rows.empty or calibration_rows.empty:
            raise ValueError(f"{regime} probability fit/calibration rows are empty")
        matrix = fit_rows.loc[:, feature_names].to_numpy(dtype="float64")
        labels = fit_rows["_positive_5d"].to_numpy(dtype="float64")
        logistic = fit_logistic(matrix, labels, l2=float(l2))
        calibration_matrix = calibration_rows.loc[:, feature_names].to_numpy(
            dtype="float64"
        )
        calibration_labels = calibration_rows["_positive_5d"].to_numpy(
            dtype="float64"
        )
        calibration_raw = predict_raw_probability(logistic, calibration_matrix)
        calibration = {
            "type": "binned_residual_blend",
            "bins": fit_probability_calibrator(
                calibration_raw,
                calibration_labels,
                bins=int(bins),
            ),
            # Preserve a small, continuous part of the stock-level signal so
            # equal reliability bins never turn every stock into one number.
            "residual_weight": 0.05,
        }
        output["regimes"][regime] = {
            "model": logistic,
            "calibration": calibration,
            "fit_rows": int(len(fit_rows)),
            "calibration_rows": int(len(calibration_rows)),
        }
    return output


def apply_v5_probability_models(
    scored: pd.DataFrame, probability_model: dict
) -> pd.DataFrame:
    data = scored.copy()
    feature_names = list(probability_model.get("feature_names") or [])
    regimes = probability_model.get("regimes")
    if not feature_names or not isinstance(regimes, dict):
        raise ValueError("V5 probability model is missing")
    data["raw_probability"] = np.nan
    data["calibrated_probability"] = np.nan
    for regime in ("good", "normal", "poor"):
        mask = data["market_regime"].eq(regime)
        if not mask.any():
            continue
        specification = regimes.get(regime)
        if not isinstance(specification, dict):
            raise ValueError(f"V5 probability model is missing regime {regime}")
        matrix = data.loc[mask, feature_names].to_numpy(dtype="float64")
        raw = predict_raw_probability(specification["model"], matrix)
        calibration = specification["calibration"]
        if (
            isinstance(calibration, dict)
            and calibration.get("type") == "binned_residual_blend"
        ):
            binned = apply_probability_calibrator(raw, calibration["bins"])
            residual_weight = float(calibration.get("residual_weight", 0.05))
            calibrated = (
                (1.0 - residual_weight) * binned + residual_weight * raw
            )
        else:
            calibrated = apply_probability_calibrator(raw, calibration)
        data.loc[mask, "raw_probability"] = raw
        data.loc[mask, "calibrated_probability"] = calibrated
    return data


def score_v5_probabilities(panel: pd.DataFrame, model: dict) -> pd.DataFrame:
    if "probability_model" not in model:
        raise ValueError("V5 production model has no per-stock probability model")
    scored = score_v5_with_model(panel, model)
    return apply_v5_probability_models(scored, model["probability_model"])


def evaluate_v5_probability_calibration(
    scored: pd.DataFrame, validation_start: str
) -> dict:
    data = scored.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data[data["date"] >= pd.Timestamp(validation_start).normalize()].dropna(
        subset=["return_5", "calibrated_probability"]
    )
    if data.empty:
        raise ValueError("V5 probability validation rows are empty")
    labels = (
        pd.to_numeric(data["return_5"], errors="coerce").to_numpy(dtype="float64")
        > 0.0
    ).astype("float64")
    probabilities = np.clip(
        pd.to_numeric(data["calibrated_probability"], errors="coerce").to_numpy(
            dtype="float64"
        ),
        1e-9,
        1.0 - 1e-9,
    )
    lower = float(np.quantile(probabilities, 0.25))
    upper = float(np.quantile(probabilities, 0.75))
    low_mask = probabilities <= lower
    high_mask = probabilities >= upper
    return {
        "rows": int(len(data)),
        "unique_probabilities": int(len(np.unique(probabilities))),
        "mean_probability": float(probabilities.mean()),
        "actual_win_rate": float(labels.mean()),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(
            -np.mean(
                labels * np.log(probabilities)
                + (1.0 - labels) * np.log(1.0 - probabilities)
            )
        ),
        "low_probability_win_rate": float(labels[low_mask].mean()),
        "high_probability_win_rate": float(labels[high_mask].mean()),
        "probability_separation": float(
            labels[high_mask].mean() - labels[low_mask].mean()
        ),
    }


def score_v5_with_model(panel: pd.DataFrame, model: dict) -> pd.DataFrame:
    data = build_v5_scores(panel)
    data["market_regime"] = _assign_regimes(data, model["rule"])
    feature_names = list(model.get("feature_names") or MODEL_FEATURES)
    matrix = data.loc[:, feature_names].to_numpy(dtype="float64") - 0.5
    data["v5_score"] = 0.0
    for regime in ("good", "normal", "poor"):
        mask = data["market_regime"].eq(regime)
        coefficients = np.asarray(
            model["regime_models"][regime]["coefficients"], dtype="float64"
        )
        data.loc[mask, "v5_score"] = matrix[mask.to_numpy()] @ coefficients
    return data


def select_v5_candidates(panel: pd.DataFrame, rule: dict) -> pd.DataFrame:
    data = panel.copy()
    data["market_regime"] = _assign_regimes(data, rule)
    score_names = {name: str(spec["score"]) for name, spec in rule.items()}
    target_counts = {name: int(spec["target_count"]) for name, spec in rule.items()}
    data["selection_score_name"] = data["market_regime"].map(score_names)
    data["target_count"] = data["market_regime"].map(target_counts).astype(int)
    data["selection_score"] = np.select(
        [data["market_regime"].eq("good"), data["market_regime"].eq("normal")],
        [data[score_names["good"]], data[score_names["normal"]]],
        default=data[score_names["poor"]],
    )
    data["code"] = data["code"].astype(str).str.zfill(6)
    ranked = data.sort_values(
        ["date", "selection_score", "code"], ascending=[True, False, True]
    )
    position = ranked.groupby("date", sort=True).cumcount()
    return ranked[position < ranked["target_count"]].reset_index(drop=True)


def evaluate_v5(selected: pd.DataFrame, position_fraction: float) -> dict:
    data = selected.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    horizon_metrics: dict[str, dict] = {}
    for horizon in HORIZONS:
        stock_column = f"return_{horizon}"
        benchmark_column = f"benchmark_return_{horizon}"
        part = data.dropna(subset=[stock_column, benchmark_column]).copy()
        daily = part.groupby("date", sort=True)[[stock_column, benchmark_column]].mean()
        excess = daily[stock_column] - daily[benchmark_column]
        horizon_metrics[str(horizon)] = {
            "dates": int(len(daily)),
            "mean_net_return": float(daily[stock_column].mean()) if len(daily) else 0.0,
            "win_rate": float((daily[stock_column] > 0).mean()) if len(daily) else 0.0,
            "mean_benchmark_return": float(daily[benchmark_column].mean()) if len(daily) else 0.0,
            "mean_excess_return": float(excess.mean()) if len(excess) else 0.0,
            "benchmark_beat_rate": float((excess > 0).mean()) if len(excess) else 0.0,
        }
    five = data.dropna(subset=["return_5"]).copy()
    contributions = (
        pd.to_numeric(five["return_5"], errors="coerce") * float(position_fraction)
    )
    signal_returns = five.assign(_contribution=contributions).groupby(
        "date", sort=True
    )["_contribution"].sum()
    return {
        "trades": int(len(five)),
        "active_dates": int(five["date"].nunique()),
        "horizon_metrics": horizon_metrics,
        "max_signal_basket_drawdown": float(max_drawdown(signal_returns)),
    }


def v5_deployment_checks(metrics: dict, gates: dict) -> dict:
    checks = {
        "trade_count": int(metrics.get("trades", 0)) >= int(gates["minimum_trades"]),
        "maximum_signal_basket_drawdown": abs(float(metrics.get("max_signal_basket_drawdown", -1.0))) <= float(gates["maximum_signal_basket_drawdown"]),
    }
    for horizon in HORIZONS:
        item = metrics.get("horizon_metrics", {}).get(str(horizon), {})
        checks[f"horizon_{horizon}_mean_excess_return"] = float(item.get("mean_excess_return", -1.0)) > float(gates["minimum_mean_excess_return"])
        checks[f"horizon_{horizon}_benchmark_beat_rate"] = float(item.get("benchmark_beat_rate", 0.0)) >= float(gates["minimum_benchmark_beat_rate"])
        checks[f"horizon_{horizon}_mean_net_return"] = float(item.get("mean_net_return", -1.0)) > float(gates["minimum_mean_net_return"])
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def prepare_v5_panel(
    panel: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    cost: float,
) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data.sort_values(["code", "date"]).reset_index(drop=True)
    grouped = data.groupby("code", sort=False)
    entry = pd.to_numeric(grouped["open"].shift(-1), errors="coerce")
    for horizon in HORIZONS:
        exit_price = pd.to_numeric(grouped["close"].shift(-horizon), errors="coerce")
        data[f"return_{horizon}"] = exit_price / entry - 1.0 - float(cost)

    benchmark = benchmark_bars.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce").dt.normalize()
    benchmark_returns = calculate_horizon_returns(benchmark, HORIZONS, cost)
    labels = pd.DataFrame({"date": benchmark["date"]})
    for horizon in HORIZONS:
        labels[f"benchmark_return_{horizon}"] = benchmark_returns[f"return_{horizon}"]
    data = attach_benchmark_returns(data, labels, HORIZONS)
    excess_columns = [f"excess_{horizon}" for horizon in HORIZONS]
    clipped = data.loc[:, excess_columns].clip(-0.20, 0.20)
    data["target_mean_excess"] = clipped.mean(axis=1)
    data["target_beat_rate"] = (clipped > 0).sum(axis=1) / len(HORIZONS) - 0.5
    return data


def _serialize_records(frame: pd.DataFrame) -> list[dict]:
    keep = [
        "date",
        "code",
        "name",
        "industry",
        "market_regime",
        "target_count",
        "selection_score_name",
        "selection_score",
        "raw_probability",
        "calibrated_probability",
        *[f"return_{h}" for h in HORIZONS],
        *[f"benchmark_return_{h}" for h in HORIZONS],
        *[f"excess_{h}" for h in HORIZONS],
    ]
    records = []
    for item in frame.loc[:, [column for column in keep if column in frame]].to_dict("records"):
        for key, value in list(item.items()):
            if isinstance(value, (pd.Timestamp, datetime)):
                item[key] = value.date().isoformat()
            elif pd.isna(value):
                item[key] = None
            elif isinstance(value, np.generic):
                item[key] = value.item()
        records.append(item)
    return records


def _recent_validation(
    selected: pd.DataFrame,
    benchmark_dates,
    horizons=HORIZONS,
) -> dict:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(benchmark_dates).dropna().unique()))
    selected = selected.copy()
    selected["date"] = pd.to_datetime(selected["date"])
    output = {}
    for offset in horizons:
        if len(dates) <= offset:
            continue
        return_column = f"return_{offset}"
        benchmark_column = f"benchmark_return_{offset}"
        signal_index = len(dates) - offset - 1
        rows = selected.iloc[0:0].copy()

        # The newest benchmark window can be observable before every selected
        # stock has enough future bars (for example when the panel ends one day
        # earlier). Use the newest signal whose complete basket is observable.
        for candidate_index in range(signal_index, -1, -1):
            candidate_date = dates[candidate_index]
            candidate_rows = selected[selected["date"].eq(candidate_date)].copy()
            if not len(candidate_rows):
                continue
            required = candidate_rows[[return_column, benchmark_column]]
            if required.notna().all().all():
                signal_index = candidate_index
                rows = candidate_rows
                break

        signal_date = dates[signal_index]
        evaluation_date = dates[signal_index + offset]
        if len(rows):
            stock_return = float(rows[return_column].mean())
            benchmark_return = float(rows[benchmark_column].iloc[0])
        else:
            stock_return = None
            benchmark_return = None
        output[str(offset)] = {
            "signal_date": signal_date.date().isoformat(),
            "evaluation_date": evaluation_date.date().isoformat(),
            "selected_count": int(len(rows)),
            "mean_net_return": stock_return,
            "benchmark_return": benchmark_return,
            "excess_return": (
                stock_return - benchmark_return
                if stock_return is not None and benchmark_return is not None
                else None
            ),
            "beat_benchmark": bool(stock_return > benchmark_return)
            if stock_return is not None and benchmark_return is not None
            else False,
            "selections": _serialize_records(rows),
        }
    return output


def run_v5_backtest(
    panel: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    config: dict,
) -> tuple[dict, dict]:
    labeled = prepare_v5_panel(panel, benchmark_bars, float(config["round_trip_cost"]))
    scored = build_v5_scores(labeled)
    scored["market_regime"] = _assign_regimes(scored, config["rule"])
    fitted = fit_v5_regime_models(
        scored,
        config["training_end"],
        ridge=float(config.get("ridge", 10.0)),
    )
    probability_model = fit_v5_probability_models(
        scored,
        fit_end=config["probability_fit_end"],
        calibration_start=config["probability_calibration_start"],
        calibration_end=config["training_end"],
        l2=float(config.get("probability_l2", 1.0)),
        bins=int(config.get("probability_bins", 8)),
    )
    model_for_scoring = {
        "rule": config["rule"],
        **fitted,
        "probability_model": probability_model,
    }
    scored = score_v5_probabilities(labeled, model_for_scoring)
    selected = select_v5_candidates(scored, config["rule"])
    validation = selected[
        selected["date"] >= pd.Timestamp(config["validation_start"])
    ].copy()
    metrics = evaluate_v5(validation, float(config["position_fraction"]))
    probability_validation = evaluate_v5_probability_calibration(
        scored, config["validation_start"]
    )
    deployment = v5_deployment_checks(metrics, config["deployment_gates"])
    probability_checks = {
        "probability_varies_by_stock": probability_validation["unique_probabilities"] > 1,
        "probability_separates_high_and_low_groups": probability_validation[
            "probability_separation"
        ] > 0.0,
        "probability_brier_beats_constant_baseline": probability_validation[
            "brier_score"
        ]
        < probability_validation["actual_win_rate"]
        * (1.0 - probability_validation["actual_win_rate"]),
    }
    deployment["checks"].update(probability_checks)
    for name, passed in probability_checks.items():
        if not passed:
            deployment["failed"].append(name)
    deployment["passed"] = not deployment["failed"]
    run_id = datetime.now().strftime("hpv5-%Y%m%d-%H%M%S")
    recent = _recent_validation(selected, benchmark_bars["date"])
    recent_passed = bool(recent) and all(
        item["beat_benchmark"] for item in recent.values()
    )
    deployment["recent_windows_passed"] = recent_passed
    deployment["checks"]["recent_windows_beat_benchmark"] = recent_passed
    if not recent_passed:
        deployment["passed"] = False
        deployment["failed"].append("recent_windows_beat_benchmark")

    report = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": "high-precision-v5",
        "objective": "robust_1_3_5_day_excess_return_over_csi300",
        "rule": config["rule"],
        "training": {
            "training_end": config["training_end"],
            "validation_start": config["validation_start"],
            "embargo_note": "Training labels end before the validation year; validation dates are not used to fit coefficients.",
            **fitted,
            "probability_model": probability_model,
        },
        "trade_policy": {
            "holding_days": 5,
            "round_trip_cost": float(config["round_trip_cost"]),
            "position_fraction": float(config["position_fraction"]),
            "take_profit_pct": float(config["take_profit_pct"]),
            "stop_loss_pct": float(config["stop_loss_pct"]),
        },
        "coverage": {
            "actual_start": labeled["date"].min().date().isoformat(),
            "actual_end": labeled["date"].max().date().isoformat(),
            "usable_symbols": int(labeled["code"].nunique()),
            "stock_dates": int(len(labeled)),
        },
        "aggregate_metrics": metrics,
        "probability_validation": probability_validation,
        "recent_validation": recent,
        "deployment": deployment,
        "validation_selections": _serialize_records(validation),
        "caveats": [
            "Historical outperformance does not guarantee future outperformance.",
            "The recent three windows are a smoke gate, not a training set.",
            "Daily bars cannot reproduce intraday execution or slippage beyond the fixed cost assumption.",
        ],
    }
    model = {
        "schema_version": "high-precision-v5",
        "run_id": run_id,
        "objective": report["objective"],
        "rule": config["rule"],
        **fitted,
        "probability_model": probability_model,
        "historical_win_rate": float(metrics["horizon_metrics"]["5"]["win_rate"]),
        "historical_benchmark_beat_rate": float(metrics["horizon_metrics"]["5"]["benchmark_beat_rate"]),
        "trade_policy": report["trade_policy"],
        "universe": (
            panel.sort_values("date")
            .groupby(panel["code"].astype(str).str.zfill(6), as_index=False)
            .tail(1)
            .loc[:, [column for column in ("code", "name", "industry", "concept") if column in panel]]
            .assign(code=lambda frame: frame["code"].astype(str).str.zfill(6))
            .to_dict("records")
        ),
        "deployment": deployment,
    }
    return report, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InStock relative benchmark V5")
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize")
    optimize.add_argument("--config", type=Path, default=Path(__file__).with_name("high_precision_v5_config.json"))
    optimize.add_argument("--panel-cache", type=Path, default=Path(__file__).parent.parent / "work" / "policy_panel_593_20260716.pkl")
    optimize.add_argument("--benchmark-cache", type=Path, default=Path(r"C:\InStockLocal\backtest_cache\indices\000300.csv"))
    optimize.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "data" / "high_precision-v5-backtest-latest.json")
    optimize.add_argument("--candidate-model", type=Path, default=Path(__file__).parent.parent / "work" / "candidate-high_precision_v5_model.json")
    optimize.add_argument("--verification", type=Path, default=Path(__file__).parent.parent / "verification" / "high-precision-v5-2026-07-17.json")
    optimize.add_argument("--production-model", type=Path, default=Path(__file__).with_name("high_precision_model.json"))
    optimize.add_argument("--runtime-report", type=Path, default=Path(__file__).parent.parent / "data" / "high_precision-backtest-latest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    panel = pd.read_pickle(args.panel_cache)
    if isinstance(panel, dict):
        panel = panel["panel"]
    benchmark = pd.read_csv(args.benchmark_cache)
    report, model = run_v5_backtest(panel, benchmark, config)
    verification = {
        "run_id": report["run_id"],
        "aggregate_metrics": report["aggregate_metrics"],
        "recent_validation": report["recent_validation"],
        "deployment": report["deployment"],
    }
    save_json_atomic(args.output, report)
    save_json_atomic(args.candidate_model, model)
    save_json_atomic(args.verification, verification)
    if report["deployment"]["passed"]:
        save_json_atomic(args.production_model, model)
        save_json_atomic(args.runtime_report, report)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
