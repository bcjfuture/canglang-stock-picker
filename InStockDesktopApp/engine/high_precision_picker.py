from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from picker_backtest import (
    build_market_panel,
    default_folds,
    direct_network_environment,
    load_universe_from_db,
    max_drawdown,
    score_factors,
)


EPSILON = 1e-12
RAW_STOCK_FEATURES = (
    "mom_5",
    "mom_20",
    "mom_60",
    "trend_20_60",
    "volatility_20",
    "drawdown_60",
    "distance_high_60",
    "liquidity_20",
    "overheat",
    "rebound_setup",
    "atr_pct_14",
)
MARKET_FEATURES = (
    "breadth_above_ma20",
    "market_median_mom20",
    "market_median_drawdown60",
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype="float64"), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> dict:
    matrix = np.asarray(X, dtype="float64")
    labels = np.asarray(y, dtype="float64").reshape(-1)
    if matrix.ndim != 2 or len(matrix) != len(labels) or len(labels) == 0:
        raise ValueError("X and y must contain aligned non-empty rows")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y must be binary")
    if float(l2) < 0:
        raise ValueError("l2 must be non-negative")
    design = np.column_stack([np.ones(len(matrix)), matrix])
    weights = np.zeros(design.shape[1], dtype="float64")
    penalty = np.eye(design.shape[1], dtype="float64") * float(l2)
    penalty[0, 0] = 0.0
    iteration = 0
    for iteration in range(int(max_iter)):
        probabilities = _sigmoid(design @ weights)
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
        gradient = design.T @ (probabilities - labels) + penalty @ weights
        hessian = design.T @ (design * variance[:, None]) + penalty
        step = np.linalg.solve(
            hessian + np.eye(len(weights), dtype="float64") * 1e-10,
            gradient,
        )
        weights -= step
        if float(np.max(np.abs(step))) <= float(tol):
            break
    return {
        "intercept": float(weights[0]),
        "coef": [float(value) for value in weights[1:]],
        "l2": float(l2),
        "iterations": int(iteration + 1),
    }


def predict_raw_probability(model: dict, X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype="float64")
    coefficient = np.asarray(model["coef"], dtype="float64")
    if matrix.ndim != 2 or matrix.shape[1] != len(coefficient):
        raise ValueError("feature count does not match model")
    return _sigmoid(matrix @ coefficient + float(model["intercept"]))


def fit_probability_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 8,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> list[dict]:
    raw = np.asarray(probabilities, dtype="float64").reshape(-1)
    y = np.asarray(labels, dtype="float64").reshape(-1)
    if len(raw) != len(y) or len(raw) == 0:
        raise ValueError("calibration rows must be aligned and non-empty")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    output: list[dict] = []
    global_rate = float((y.sum() + alpha) / (len(y) + alpha + beta))
    for index in range(int(bins)):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        mask = (raw >= lower) & (
            raw <= upper if index == int(bins) - 1 else raw < upper
        )
        count = int(mask.sum())
        wins = float(y[mask].sum())
        probability = (
            float((wins + alpha) / (count + alpha + beta))
            if count
            else global_rate
        )
        output.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "wins": wins,
                "probability": probability,
            }
        )
    running = 0.0
    for item in output:
        running = max(running, float(item["probability"]))
        item["probability"] = min(running, 1.0 - EPSILON)
    return output


def apply_probability_calibrator(
    probabilities: np.ndarray, calibration: list[dict]
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype="float64").reshape(-1)
    if not calibration:
        raise ValueError("calibration is empty")
    result = np.empty(len(raw), dtype="float64")
    for position, value in enumerate(raw):
        chosen = calibration[-1]
        for item in calibration:
            upper_inclusive = item is calibration[-1]
            if float(item["lower"]) <= value and (
                value < float(item["upper"])
                or (upper_inclusive and value <= float(item["upper"]))
            ):
                chosen = item
                break
        result[position] = float(chosen["probability"])
    return np.clip(result, EPSILON, 1.0 - EPSILON)


def save_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def calculate_high_precision_forward_5(
    panel: pd.DataFrame, cost: float = 0.003
) -> pd.Series:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["_position"] = np.arange(len(data))
    data = data.sort_values(["code", "date", "_position"])
    entry = data.groupby("code", sort=False)["open"].shift(-1)
    exit_price = data.groupby("code", sort=False)["close"].shift(-5)
    result = exit_price / entry - 1.0 - float(cost)
    result = result.mask(
        ~np.isfinite(entry)
        | (entry <= 0)
        | ~np.isfinite(exit_price)
        | (exit_price <= 0)
    )
    ordered = pd.Series(result.to_numpy(), index=data["_position"].to_numpy())
    return ordered.sort_index().astype("float64")


def _profile_by_name(profiles: dict, group: str, name: str) -> dict:
    for candidate in profiles[group]["candidates"]:
        if candidate["name"] == name:
            return candidate
    raise KeyError(f"profile not found: {group}/{name}")


def _tech_mask(panel: pd.DataFrame) -> pd.Series:
    industry = (
        panel.get("industry", pd.Series("", index=panel.index))
        .fillna("")
        .astype(str)
    )
    concept = (
        panel.get("concept", pd.Series("", index=panel.index))
        .fillna("")
        .astype(str)
    )
    text = (industry + " " + concept).str.upper()
    keywords = (
        "半导体",
        "芯片",
        "人工智能",
        "AI",
        "算力",
        "服务器",
        "软件",
        "通信",
        "5G",
        "6G",
        "光模块",
        "CPO",
        "PCB",
        "电子",
        "机器人",
        "智能制造",
        "智能驾驶",
        "物联网",
        "鸿蒙",
        "低空经济",
        "商业航天",
    )
    mask = pd.Series(False, index=panel.index)
    for keyword in keywords:
        mask |= text.str.contains(keyword.upper(), regex=False)
    return mask.astype(float)


def _ensure_market_context(data: pd.DataFrame) -> pd.DataFrame:
    if all(column in data for column in MARKET_FEATURES):
        return data
    output = data.copy()
    output["above_ma20"] = (
        pd.to_numeric(output["close"], errors="coerce")
        > pd.to_numeric(output["ma_20"], errors="coerce")
    ).astype(float)
    grouped = output.groupby("date")
    output["breadth_above_ma20"] = grouped["above_ma20"].transform("mean")
    output["market_median_mom20"] = grouped["mom_20"].transform("median")
    output["market_median_drawdown60"] = grouped["drawdown_60"].transform(
        "median"
    )
    return output


def _market_statistics(frame: pd.DataFrame) -> dict:
    output = {}
    for column in MARKET_FEATURES:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        median = float(numeric.median()) if numeric.notna().any() else 0.0
        scale = (
            float((numeric - median).abs().median())
            if numeric.notna().any()
            else 1.0
        )
        output[column] = {
            "median": median,
            "scale": scale if scale > 1e-12 else 1.0,
        }
    return output


def build_high_precision_features(
    panel: pd.DataFrame,
    profiles: dict,
    config: dict,
    train_statistics: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data.dropna(subset=["date", "code"]).sort_values(
        ["date", "code"]
    )
    data = _ensure_market_context(data)
    all_profile = _profile_by_name(
        profiles, "all_market", "all_trend_quality"
    )
    tech_profile = _profile_by_name(
        profiles, "tech_attack", "tech_momentum_quality"
    )
    rebound_profile = _profile_by_name(
        profiles, "rebound_trend", "rebound_confirmation"
    )
    data["all_trend_quality_score"] = score_factors(data, all_profile)
    data["tech_momentum_quality_score"] = score_factors(data, tech_profile)
    data["rebound_score"] = score_factors(data, rebound_profile)
    for column in config["stock_rank_features"]:
        numeric = pd.to_numeric(data[column], errors="coerce")
        data[column] = numeric.groupby(data["date"]).rank(
            pct=True, method="average"
        ).fillna(0.5)
    data["is_tech"] = _tech_mask(data)
    statistics = train_statistics or _market_statistics(data)
    for column in config["market_features"]:
        item = statistics[column]
        data[f"{column}_raw"] = pd.to_numeric(
            data[column], errors="coerce"
        ).fillna(float(item["median"]))
        data[column] = (
            data[f"{column}_raw"] - float(item["median"])
        ) / float(item["scale"])
    data["forward_5"] = calculate_high_precision_forward_5(
        data, cost=float(config["round_trip_cost"])
    ).to_numpy()
    data["label_5"] = (
        pd.to_numeric(data["forward_5"], errors="coerce") > 0
    ).astype(int)
    keep = [
        "date",
        "code",
        "industry",
        "name",
        "concept",
        "forward_5",
        "label_5",
        "breadth_above_ma20_raw",
        "market_median_mom20_raw",
        "market_median_drawdown60_raw",
        *config["features"],
    ]
    keep = list(dict.fromkeys(column for column in keep if column in data))
    return data.loc[:, keep].reset_index(drop=True), statistics


def make_training_rows(feature_frame: pd.DataFrame) -> pd.DataFrame:
    data = feature_frame.copy()
    data["forward_5"] = pd.to_numeric(data["forward_5"], errors="coerce")
    data = data.dropna(subset=["forward_5"])
    data["label_5"] = (data["forward_5"] > 0).astype(int)
    return data.reset_index(drop=True)


def split_fit_and_calibration(
    train: pd.DataFrame, calibration_trading_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = train.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    dates = sorted(data["date"].dropna().unique())
    days = int(calibration_trading_days)
    if len(dates) <= days:
        raise ValueError("training history is shorter than calibration tail")
    calibration_dates = set(dates[-days:])
    fit = data[~data["date"].isin(calibration_dates)].copy()
    calibration = data[data["date"].isin(calibration_dates)].copy()
    if (
        fit.empty
        or calibration.empty
        or fit["date"].max() >= calibration["date"].min()
    ):
        raise ValueError("fit and calibration periods must be chronological")
    return fit, calibration


@dataclass(frozen=True)
class HighPrecisionPolicy:
    breadth_floor: float
    momentum_floor: float
    drawdown_floor: float
    strong_threshold: float
    normal_threshold_delta: float
    max_picks: int

    def __post_init__(self) -> None:
        if self.max_picks not in {1, 2, 3}:
            raise ValueError("max_picks must be 1, 2 or 3")
        if not 0.5 <= self.strong_threshold < 1.0:
            raise ValueError("strong_threshold must be in [0.5, 1.0)")
        if self.normal_threshold_delta < 0:
            raise ValueError("normal_threshold_delta must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "HighPrecisionPolicy":
        return cls(
            **{name: value[name] for name in cls.__dataclass_fields__}
        )


def classify_high_precision_regime(
    row: pd.Series, policy: HighPrecisionPolicy
) -> str:
    breadth = float(row["breadth_above_ma20_raw"])
    momentum = float(row["market_median_mom20_raw"])
    drawdown = float(row["market_median_drawdown60_raw"])
    if (
        breadth < policy.breadth_floor
        or momentum < policy.momentum_floor
        or drawdown < policy.drawdown_floor
    ):
        return "weak"
    if (
        breadth >= policy.breadth_floor + 0.10
        and momentum >= policy.momentum_floor + 0.10
        and drawdown >= policy.drawdown_floor + 0.05
    ):
        return "strong"
    return "normal"


def select_daily_candidates(
    predictions: pd.DataFrame, policy: HighPrecisionPolicy
) -> pd.DataFrame:
    data = predictions.copy()
    if data.empty:
        return data.assign(
            market_regime=pd.Series(dtype="object"),
            probability_threshold=pd.Series(dtype="float64"),
        )
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    output = []
    for _signal_date, cross in data.groupby("date", sort=True):
        regime = classify_high_precision_regime(cross.iloc[0], policy)
        if regime == "weak":
            continue
        threshold = float(policy.strong_threshold)
        if regime == "normal":
            threshold += float(policy.normal_threshold_delta)
        eligible = cross[
            pd.to_numeric(
                cross["calibrated_probability"], errors="coerce"
            )
            >= threshold
        ].copy()
        eligible = eligible.sort_values(
            ["calibrated_probability", "code"],
            ascending=[False, True],
        )
        if "industry" in eligible:
            eligible["_industry_key"] = eligible["industry"].fillna("").astype(str)
            eligible.loc[
                eligible["_industry_key"].eq(""), "_industry_key"
            ] = eligible["code"].astype(str)
            eligible = eligible.drop_duplicates("_industry_key", keep="first")
            eligible = eligible.drop(columns="_industry_key")
        eligible = eligible.head(policy.max_picks)
        if eligible.empty:
            continue
        eligible["market_regime"] = regime
        eligible["probability_threshold"] = threshold
        output.append(eligible)
    if not output:
        return data.iloc[0:0].assign(
            market_regime=pd.Series(dtype="object"),
            probability_threshold=pd.Series(dtype="float64"),
        )
    return pd.concat(output, ignore_index=True)


def evaluate_selected_trades(
    selected: pd.DataFrame,
    scheduled_dates: list[pd.Timestamp],
    folds: list[dict],
) -> dict:
    data = selected.copy()
    scheduled = [
        pd.Timestamp(value).normalize() for value in scheduled_dates
    ]
    if data.empty:
        return {
            "scheduled_dates": len(scheduled),
            "active_dates": 0,
            "activity_rate": 0.0,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "mean_net_return": 0.0,
            "mean_excess_return": 0.0,
            "max_drawdown": 0.0,
            "fold_active_dates": [0 for _ in folds],
            "fold_win_rates": [0.0 for _ in folds],
        }
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["forward_5"] = pd.to_numeric(data["forward_5"], errors="coerce")
    data["benchmark_forward_5"] = pd.to_numeric(
        data["benchmark_forward_5"], errors="coerce"
    )
    data = data.dropna(subset=["forward_5", "benchmark_forward_5"])
    active_returns = data.groupby("date", sort=True)["forward_5"].mean()
    fold_active_dates = []
    fold_win_rates = []
    for fold in folds:
        mask = data["date"].between(
            pd.Timestamp(fold["valid_start"]),
            pd.Timestamp(fold["valid_end"]),
        )
        part = data[mask]
        fold_active_dates.append(int(part["date"].nunique()))
        fold_win_rates.append(
            float((part["forward_5"] > 0).mean()) if not part.empty else 0.0
        )
    active_count = int(data["date"].nunique())
    trades = int(len(data))
    return {
        "scheduled_dates": len(scheduled),
        "active_dates": active_count,
        "activity_rate": active_count / len(scheduled) if scheduled else 0.0,
        "trades": trades,
        "wins": int((data["forward_5"] > 0).sum()),
        "win_rate": float((data["forward_5"] > 0).mean()) if trades else 0.0,
        "mean_net_return": float(data["forward_5"].mean()) if trades else 0.0,
        "mean_excess_return": float(
            (data["forward_5"] - data["benchmark_forward_5"]).mean()
        )
        if trades
        else 0.0,
        "max_drawdown": max_drawdown(active_returns),
        "fold_active_dates": fold_active_dates,
        "fold_win_rates": fold_win_rates,
    }


def deployment_checks(
    metrics: dict, neighbor_metrics: list[dict], config: dict
) -> dict:
    folds = list(metrics["fold_win_rates"])
    checks = {
        "aggregate_win_rate": metrics["win_rate"]
        >= config["minimum_win_rate"],
        "activity_rate": metrics["activity_rate"]
        >= config["minimum_activity_rate"],
        "trade_count": metrics["trades"] >= config["minimum_trades"],
        "active_dates_per_fold": all(
            value >= config["minimum_active_dates_per_fold"]
            for value in metrics["fold_active_dates"]
        ),
        "worst_fold_win_rate": bool(folds)
        and min(folds) >= config["minimum_fold_win_rate"],
        "majority_fold_win_rate": sum(
            value >= config["minimum_majority_fold_win_rate"]
            for value in folds
        )
        >= config["minimum_majority_fold_count"],
        "mean_net_return": metrics["mean_net_return"]
        > config["minimum_mean_net_return"],
        "mean_excess_return": metrics["mean_excess_return"]
        > config["minimum_mean_excess_return"],
        "maximum_drawdown": abs(metrics["max_drawdown"])
        <= config["maximum_drawdown"],
        "positive_neighbors": sum(
            float(item["mean_net_return"]) > 0 for item in neighbor_metrics
        )
        >= config["minimum_positive_neighbors"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def parameter_grid(config: dict) -> list[dict]:
    output = []
    for l2, threshold, delta, max_picks, market in product(
        config["l2_values"],
        config["strong_thresholds"],
        config["normal_threshold_deltas"],
        config["max_picks_values"],
        config["market_policies"],
    ):
        output.append(
            {
                "l2": float(l2),
                "policy": HighPrecisionPolicy(
                    breadth_floor=float(market["breadth_floor"]),
                    momentum_floor=float(market["momentum_floor"]),
                    drawdown_floor=float(market["drawdown_floor"]),
                    strong_threshold=float(threshold),
                    normal_threshold_delta=float(delta),
                    max_picks=int(max_picks),
                ).to_dict(),
            }
        )
    return output


def _feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return (
        frame.loc[:, feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )


def _benchmark_returns(benchmark: pd.DataFrame, cost: float) -> pd.DataFrame:
    data = benchmark.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if "code" not in data:
        data["code"] = "000300"
    data["benchmark_forward_5"] = calculate_high_precision_forward_5(
        data, cost=cost
    ).to_numpy()
    return data.loc[:, ["date", "benchmark_forward_5"]].drop_duplicates(
        "date", keep="last"
    )


def _attach_benchmark(
    frame: pd.DataFrame, benchmark_returns: pd.DataFrame
) -> pd.DataFrame:
    return frame.merge(benchmark_returns, on="date", how="left")


def _predict_frame(
    frame: pd.DataFrame,
    model: dict,
    calibration: list[dict],
    feature_names: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    raw = predict_raw_probability(
        model, _feature_matrix(output, feature_names)
    )
    output["raw_probability"] = raw
    output["calibrated_probability"] = apply_probability_calibrator(
        raw, calibration
    )
    return output


def _calibration_candidate_key(item: dict) -> tuple:
    metrics = item["metrics"]
    return (
        bool(item["eligible"]),
        float(metrics["win_rate"]),
        float(metrics["mean_net_return"]),
        float(metrics["mean_excess_return"]),
        -abs(float(metrics["max_drawdown"])),
        -int(item["grid_index"]),
    )


def _json_records(frame: pd.DataFrame) -> list[dict]:
    output = []
    for record in frame.to_dict("records"):
        normalized = {}
        for key, value in record.items():
            if isinstance(value, (pd.Timestamp, datetime, date)):
                normalized[key] = pd.Timestamp(value).date().isoformat()
            elif isinstance(value, np.generic):
                normalized[key] = value.item()
            elif pd.isna(value):
                normalized[key] = None
            else:
                normalized[key] = value
        output.append(normalized)
    return output


def _neighbor_values(value, configured: list) -> list:
    values = list(configured)
    position = values.index(value)
    output = []
    for index in (position - 1, position + 1):
        if 0 <= index < len(values):
            output.append(values[index])
    return output


def run_walk_forward_search(
    panel: pd.DataFrame,
    profiles: dict,
    config: dict,
    benchmark: pd.DataFrame,
    folds: list[dict],
) -> dict:
    raw = panel.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    benchmark_returns = _benchmark_returns(
        benchmark, float(config["round_trip_cost"])
    )
    feature_names = list(config["features"])
    grid = parameter_grid(config)
    attempts = []
    fold_outputs = []
    fold_runtime = []
    selected_parts = []
    scheduled_dates: list[pd.Timestamp] = []
    minimum_months = int(config.get("minimum_training_months", 18))

    for fold_index, fold in enumerate(folds):
        train_end = pd.Timestamp(fold["train_end"]).normalize()
        valid_start = pd.Timestamp(fold["valid_start"]).normalize()
        valid_end = pd.Timestamp(fold["valid_end"]).normalize()
        if not train_end < valid_start <= valid_end:
            raise ValueError("fold dates must be chronological")
        train_raw = raw[raw["date"] <= train_end].copy()
        valid_raw = raw[raw["date"].between(valid_start, valid_end)].copy()
        if train_raw.empty or valid_raw.empty:
            raise ValueError("fold contains no training or validation rows")
        if minimum_months:
            required_start = train_end - pd.DateOffset(months=minimum_months)
            if train_raw["date"].min() > required_start:
                raise ValueError("training window is shorter than required months")

        train_features, statistics = build_high_precision_features(
            train_raw, profiles, config
        )
        train_features = _attach_benchmark(
            make_training_rows(train_features), benchmark_returns
        ).dropna(subset=["benchmark_forward_5"])
        fit_rows, calibration_rows = split_fit_and_calibration(
            train_features, int(config["calibration_trading_days"])
        )
        valid_features, _ = build_high_precision_features(
            valid_raw,
            profiles,
            config,
            train_statistics=statistics,
        )
        valid_features = _attach_benchmark(
            make_training_rows(valid_features), benchmark_returns
        ).dropna(subset=["benchmark_forward_5"])
        if valid_features.empty:
            raise ValueError("validation fold has no fully labeled rows")

        calibration_dates = sorted(calibration_rows["date"].unique())
        valid_dates = sorted(valid_features["date"].unique())
        scheduled_dates.extend(pd.Timestamp(value) for value in valid_dates)
        l2_runtime: dict[float, dict] = {}
        for l2 in config["l2_values"]:
            model = fit_logistic(
                _feature_matrix(fit_rows, feature_names),
                fit_rows["label_5"].to_numpy(dtype="float64"),
                l2=float(l2),
            )
            calibration_raw = predict_raw_probability(
                model, _feature_matrix(calibration_rows, feature_names)
            )
            calibrator = fit_probability_calibrator(
                calibration_raw,
                calibration_rows["label_5"].to_numpy(dtype="float64"),
                bins=int(config["calibration_bins"]),
                alpha=float(config["beta_alpha"]),
                beta=float(config["beta_beta"]),
            )
            calibration_predictions = calibration_rows.copy()
            calibration_predictions["raw_probability"] = calibration_raw
            calibration_predictions[
                "calibrated_probability"
            ] = apply_probability_calibrator(calibration_raw, calibrator)
            validation_predictions = _predict_frame(
                valid_features, model, calibrator, feature_names
            )
            l2_runtime[float(l2)] = {
                "model": model,
                "calibration": calibrator,
                "validation_predictions": validation_predictions,
            }

            for grid_index, candidate in enumerate(grid):
                if float(candidate["l2"]) != float(l2):
                    continue
                policy = HighPrecisionPolicy.from_dict(candidate["policy"])
                selected = select_daily_candidates(
                    calibration_predictions, policy
                )
                calibration_fold = [
                    {
                        "valid_start": pd.Timestamp(
                            calibration_dates[0]
                        ).date().isoformat(),
                        "valid_end": pd.Timestamp(
                            calibration_dates[-1]
                        ).date().isoformat(),
                    }
                ]
                metrics = evaluate_selected_trades(
                    selected, calibration_dates, calibration_fold
                )
                eligible = (
                    metrics["activity_rate"]
                    >= float(config["calibration_minimum_activity_rate"])
                    and metrics["trades"]
                    >= int(config["calibration_minimum_trades"])
                )
                attempt = {
                    "fold_index": fold_index,
                    "grid_index": grid_index,
                    "parameters": candidate,
                    "eligible": bool(eligible),
                    "metrics": metrics,
                    "rejection_reasons": [
                        name
                        for name, passed in (
                            (
                                "calibration_activity_rate",
                                metrics["activity_rate"]
                                >= float(
                                    config[
                                        "calibration_minimum_activity_rate"
                                    ]
                                ),
                            ),
                            (
                                "calibration_trade_count",
                                metrics["trades"]
                                >= int(config["calibration_minimum_trades"]),
                            ),
                        )
                        if not passed
                    ],
                }
                attempts.append(attempt)

        fold_attempts = [
            item for item in attempts if item["fold_index"] == fold_index
        ]
        chosen_attempt = max(fold_attempts, key=_calibration_candidate_key)
        chosen = chosen_attempt["parameters"]
        chosen_runtime = l2_runtime[float(chosen["l2"])]
        chosen_policy = HighPrecisionPolicy.from_dict(chosen["policy"])
        validation_predictions = chosen_runtime["validation_predictions"]
        validation_selected = select_daily_candidates(
            validation_predictions, chosen_policy
        )
        validation_metrics = evaluate_selected_trades(
            validation_selected, valid_dates, [fold]
        )
        selected_parts.append(validation_selected)
        fold_runtime.append(
            {
                "fold": fold,
                "valid_dates": valid_dates,
                "l2_runtime": l2_runtime,
                "chosen": chosen,
            }
        )
        fold_outputs.append(
            {
                "train_end": train_end.date().isoformat(),
                "fit_max_date": fit_rows["date"].max().date().isoformat(),
                "calibration_min_date": calibration_rows[
                    "date"
                ].min().date().isoformat(),
                "calibration_max_date": calibration_rows[
                    "date"
                ].max().date().isoformat(),
                "validation_min_date": valid_features[
                    "date"
                ].min().date().isoformat(),
                "validation_max_date": valid_features[
                    "date"
                ].max().date().isoformat(),
                "chosen": chosen,
                "calibration_metrics": chosen_attempt["metrics"],
                "validation_metrics": validation_metrics,
                "validation_prediction_rows": int(len(validation_predictions)),
                "validation_selections": _json_records(validation_selected),
            }
        )

    all_selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else pd.DataFrame()
    )
    aggregate_metrics = evaluate_selected_trades(
        all_selected, sorted(set(scheduled_dates)), folds
    )
    chosen_keys = [
        json.dumps(item["chosen"], sort_keys=True) for item in fold_runtime
    ]
    modal_key = Counter(chosen_keys).most_common(1)[0][0]
    modal_chosen = json.loads(modal_key)

    neighbor_specs = []
    for value in _neighbor_values(
        float(modal_chosen["policy"]["strong_threshold"]),
        config["strong_thresholds"],
    ):
        neighbor_specs.append(
            {"kind": "strong_threshold", "value": float(value)}
        )
    for value in _neighbor_values(
        float(modal_chosen["l2"]), config["l2_values"]
    ):
        neighbor_specs.append({"kind": "l2", "value": float(value)})
    neighbor_metrics = []
    for neighbor in neighbor_specs:
        parts = []
        dates_for_neighbor = []
        for runtime in fold_runtime:
            chosen = runtime["chosen"]
            policy_data = dict(chosen["policy"])
            l2_value = float(chosen["l2"])
            if neighbor["kind"] == "strong_threshold":
                policy_data["strong_threshold"] = neighbor["value"]
            else:
                l2_value = neighbor["value"]
                if l2_value not in runtime["l2_runtime"]:
                    continue
            predictions = runtime["l2_runtime"][l2_value][
                "validation_predictions"
            ]
            parts.append(
                select_daily_candidates(
                    predictions,
                    HighPrecisionPolicy.from_dict(policy_data),
                )
            )
            dates_for_neighbor.extend(runtime["valid_dates"])
        selected_neighbor = (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        )
        metrics = evaluate_selected_trades(
            selected_neighbor, sorted(set(dates_for_neighbor)), folds
        )
        neighbor_metrics.append({**neighbor, **metrics})

    gates = deployment_checks(
        aggregate_metrics,
        neighbor_metrics,
        config["deployment_gates"],
    )
    run_id = datetime.now().strftime("hp-%Y%m%d-%H%M%S")
    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "walk_forward": {"folds": fold_outputs},
        "aggregate_metrics": aggregate_metrics,
        "neighbor_metrics": neighbor_metrics,
        "chosen": modal_chosen,
        "deployment": gates,
    }


def fit_final_model(
    panel: pd.DataFrame,
    profiles: dict,
    config: dict,
    chosen: dict,
    run_id: str,
    deployment: dict,
) -> dict:
    features, statistics = build_high_precision_features(
        panel, profiles, config
    )
    rows = make_training_rows(features)
    fit_rows, calibration_rows = split_fit_and_calibration(
        rows, int(config["calibration_trading_days"])
    )
    feature_names = list(config["features"])
    model = fit_logistic(
        _feature_matrix(fit_rows, feature_names),
        fit_rows["label_5"].to_numpy(dtype="float64"),
        l2=float(chosen["l2"]),
    )
    raw = predict_raw_probability(
        model, _feature_matrix(calibration_rows, feature_names)
    )
    calibration = fit_probability_calibrator(
        raw,
        calibration_rows["label_5"].to_numpy(dtype="float64"),
        bins=int(config["calibration_bins"]),
        alpha=float(config["beta_alpha"]),
        beta=float(config["beta_beta"]),
    )
    return {
        "schema_version": "high-precision-model-v1",
        "run_id": str(run_id),
        "feature_names": feature_names,
        "model": model,
        "market_statistics": statistics,
        "calibration": calibration,
        "policy": chosen["policy"],
        "deployment": deployment,
    }


def recalculate_sample_returns(
    report: dict,
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    sample_size: int = 10,
    seed: int = 20260717,
) -> dict:
    selections = []
    for fold in report.get("walk_forward", {}).get("folds", []):
        selections.extend(fold.get("validation_selections", []))
    if not selections:
        return {"passed": True, "sample_count": 0, "samples": []}
    randomizer = random.Random(int(seed))
    sample = randomizer.sample(
        selections, min(int(sample_size), len(selections))
    )
    stock_data = panel.copy()
    stock_data["date"] = pd.to_datetime(
        stock_data["date"], errors="coerce"
    ).dt.normalize()
    stock_data["code"] = stock_data["code"].astype(str).str.zfill(6)
    stock_data["recalculated_forward_5"] = calculate_high_precision_forward_5(
        stock_data
    ).to_numpy()
    stock_map = {
        (str(row.code).zfill(6), pd.Timestamp(row.date).normalize()): float(
            row.recalculated_forward_5
        )
        for row in stock_data.itertuples()
        if pd.notna(row.recalculated_forward_5)
    }
    benchmark_data = _benchmark_returns(benchmark, 0.003)
    benchmark_map = {
        pd.Timestamp(row.date).normalize(): float(row.benchmark_forward_5)
        for row in benchmark_data.itertuples()
        if pd.notna(row.benchmark_forward_5)
    }
    output = []
    for item in sample:
        signal_date = pd.Timestamp(item["date"]).normalize()
        code = str(item["code"]).zfill(6)
        actual_return = stock_map[(code, signal_date)]
        actual_benchmark = benchmark_map[signal_date]
        return_diff = abs(actual_return - float(item["forward_5"]))
        benchmark_diff = abs(
            actual_benchmark - float(item["benchmark_forward_5"])
        )
        output.append(
            {
                "date": signal_date.date().isoformat(),
                "code": code,
                "stored_return": float(item["forward_5"]),
                "recalculated_return": actual_return,
                "return_abs_diff": return_diff,
                "stored_benchmark_return": float(
                    item["benchmark_forward_5"]
                ),
                "recalculated_benchmark_return": actual_benchmark,
                "benchmark_abs_diff": benchmark_diff,
            }
        )
    return {
        "passed": all(
            item["return_abs_diff"] < 1e-12
            and item["benchmark_abs_diff"] < 1e-12
            for item in output
        ),
        "sample_count": len(output),
        "samples": output,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_or_build_panel(
    *,
    start: str,
    end: str,
    universe_limit: int,
    workers: int,
    cache_dir: Path,
    panel_cache: Path,
    cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    universe = load_universe_from_db(limit=universe_limit)
    if panel_cache.exists():
        cached = pd.read_pickle(panel_cache)
        if (
            isinstance(cached, dict)
            and cached.get("start") == start
            and int(cached.get("universe_limit", 0)) == int(universe_limit)
            and pd.Timestamp(cached["panel"]["date"].max())
            >= pd.Timestamp(end)
        ):
            return cached["panel"], universe, cached.get("fetch_errors", [])
    with direct_network_environment():
        panel, errors = build_market_panel(
            universe,
            start,
            end,
            cache_dir,
            workers=workers,
            cost=cost,
        )
    panel_cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {
            "start": start,
            "end": end,
            "universe_limit": universe_limit,
            "panel": panel,
            "fetch_errors": errors,
        },
        panel_cache,
    )
    return panel, universe, errors


def run_optimizer_cli(args: argparse.Namespace) -> dict:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    cost = float(config["round_trip_cost"])
    panel, universe, errors = _load_or_build_panel(
        start=args.start,
        end=args.end,
        universe_limit=args.universe_limit,
        workers=args.workers,
        cache_dir=args.cache_dir,
        panel_cache=args.panel_cache,
        cost=cost,
    )
    benchmark = pd.read_csv(args.benchmark_cache)
    search = run_walk_forward_search(
        panel,
        profiles,
        config,
        benchmark,
        default_folds(),
    )
    model = fit_final_model(
        panel,
        profiles,
        config,
        search["chosen"],
        search["run_id"],
        search["deployment"],
    )
    dates = pd.to_datetime(panel["date"], errors="coerce")
    report = {
        **search,
        "method": {
            "signal": "close",
            "entry": "next_open",
            "exit": "signal_plus_5_close",
            "round_trip_cost": cost,
            "calibration_trading_days": int(
                config["calibration_trading_days"]
            ),
            "walk_forward_folds": default_folds(),
        },
        "coverage": {
            "requested_start": args.start,
            "requested_end": args.end,
            "actual_start": dates.min().date().isoformat(),
            "actual_end": dates.max().date().isoformat(),
            "requested_symbols": int(len(universe)),
            "usable_symbols": int(panel["code"].nunique()),
            "stock_dates": int(len(panel)),
            "fetch_failures": errors,
        },
        "caveats": [
            "Current-universe construction introduces survivorship bias.",
            "Rolling out-of-sample performance does not guarantee future returns.",
        ],
    }
    verification = {
        "run_id": report["run_id"],
        "generated_at": report["generated_at"],
        "input_hashes": {
            "config": _file_sha256(args.config),
            "profiles": _file_sha256(args.profiles),
            "benchmark": _file_sha256(args.benchmark_cache),
        },
        "attempt_count": report["attempt_count"],
        "fold_metrics": [
            item["validation_metrics"]
            for item in report["walk_forward"]["folds"]
        ],
        "aggregate_metrics": report["aggregate_metrics"],
        "neighbor_metrics": report["neighbor_metrics"],
        "deployment": report["deployment"],
    }
    save_json_atomic(args.output, report)
    save_json_atomic(args.candidate_model, model)
    save_json_atomic(args.verification, verification)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InStock high precision picker"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize")
    optimize.add_argument("--start", default="2022-01-01")
    optimize.add_argument("--end", default=date.today().isoformat())
    optimize.add_argument("--universe-limit", type=int, default=600)
    optimize.add_argument("--workers", type=int, default=16)
    optimize.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(r"C:\InStockLocal\backtest_cache\daily_bars"),
    )
    optimize.add_argument(
        "--panel-cache",
        type=Path,
        default=Path(
            r"C:\InStockLocal\backtest_cache\high_precision_panel.pkl"
        ),
    )
    optimize.add_argument(
        "--benchmark-cache",
        type=Path,
        default=Path(r"C:\InStockLocal\backtest_cache\indices\000300.csv"),
    )
    optimize.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).with_name("picker_factor_profiles.json"),
    )
    optimize.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("high_precision_config.json"),
    )
    optimize.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"C:\InStockLocal\InStockDesktopApp\data"
            r"\high_precision-backtest-latest.json"
        ),
    )
    optimize.add_argument(
        "--candidate-model",
        type=Path,
        default=Path(
            r"C:\InStockLocal\InStockDesktopApp\work"
            r"\candidate-high_precision_model.json"
        ),
    )
    optimize.add_argument(
        "--verification",
        type=Path,
        default=Path(
            r"C:\InStockLocal\InStockDesktopApp\verification"
            r"\high-precision-picker-2026-07-17.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "optimize":
        report = run_optimizer_cli(args)
        print(
            json.dumps(
                {
                    "run_id": report["run_id"],
                    "attempt_count": report["attempt_count"],
                    "deployment": report["deployment"],
                    "aggregate_metrics": report["aggregate_metrics"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
