from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import date, datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from high_precision_picker import (
    HighPrecisionPolicy,
    apply_probability_calibrator,
    calculate_high_precision_forward_5,
    deployment_checks,
    evaluate_selected_trades,
    fit_logistic,
    fit_probability_calibrator,
    predict_raw_probability,
    save_json_atomic,
    select_daily_candidates,
    split_fit_and_calibration,
)
from picker_backtest import default_folds

QUARTER_DATES = (
    "20211231",
    "20220331",
    "20220630",
    "20220930",
    "20221231",
    "20230331",
    "20230630",
    "20230930",
    "20231231",
    "20240331",
    "20240630",
    "20240930",
    "20241231",
    "20250331",
    "20250630",
    "20250930",
    "20251231",
    "20260331",
)

FUNDAMENTAL_VALUE_COLUMNS = (
    "net_profit_yoy",
    "revenue_yoy",
    "net_margin",
    "cfo_to_profit",
    "debt_ratio",
    "cash_to_assets",
    "receivables_to_assets",
    "inventory_to_assets",
    "total_assets_yoy",
)

TREND_FEATURES = (
    "mom_3",
    "mom_10",
    "mom_20",
    "mom_60",
    "excess_mom5",
    "excess_mom20",
    "excess_mom60",
    "trend_slope_20",
    "trend_r2_20",
    "up_day_ratio_20",
    "industry_median_mom5",
    "industry_median_mom20",
    "industry_median_mom60",
    "industry_breadth_ma20",
    "industry_strength_rank",
    "stock_minus_industry_mom20",
)

REVERSAL_FEATURES = (
    "volume_accel_5_20",
    "amount_accel_5_20",
    "rsi_14",
    "bollinger_position_20",
    "opening_gap",
    "intraday_close_strength",
    "volume_price_corr_20",
    "max_return_5",
    "min_return_5",
    "atr_contraction",
    "industry_mom20_dispersion",
)

QUALITY_FEATURES = (
    "net_profit_yoy",
    "revenue_yoy",
    "net_margin",
    "cfo_to_profit",
    "debt_ratio",
    "cash_to_assets",
    "receivables_to_assets",
    "inventory_to_assets",
    "total_assets_yoy",
    "fundamental_age_rank",
    "fundamental_missing",
    "downside_volatility_20",
    "liquidity_20",
)


def _clean_code(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.extract(r"(\d{1,6})", expand=False)
        .fillna("")
        .str.zfill(6)
    )


def _statement_part(
    frame: pd.DataFrame,
    mapping: dict[str, str],
    prefix: str,
) -> pd.DataFrame:
    selected = [column for column in mapping if column in frame.columns]
    if "股票代码" not in selected:
        return pd.DataFrame(columns=["code", f"{prefix}_announcement_date"])
    result = frame.loc[:, selected].rename(columns=mapping).copy()
    result["code"] = _clean_code(result["code"])
    if "announcement_date" in result:
        result["announcement_date"] = pd.to_datetime(
            result["announcement_date"], errors="coerce"
        ).dt.normalize()
        result = result.sort_values(["code", "announcement_date"])
        result = result.drop_duplicates("code", keep="last")
        result = result.rename(
            columns={"announcement_date": f"{prefix}_announcement_date"}
        )
    else:
        result[f"{prefix}_announcement_date"] = pd.NaT
    for column in result.columns:
        if column not in {"code", f"{prefix}_announcement_date"}:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    return (numerator / denominator.where(denominator.abs() > 1e-12)).replace(
        [np.inf, -np.inf], np.nan
    )


def normalize_quarterly_statements(
    report_date: str,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> pd.DataFrame:
    income_part = _statement_part(
        income,
        {
            "股票代码": "code",
            "公告日期": "announcement_date",
            "净利润": "net_profit",
            "净利润同比": "net_profit_yoy",
            "营业总收入": "revenue",
            "营业总收入同比": "revenue_yoy",
        },
        "income",
    )
    balance_part = _statement_part(
        balance,
        {
            "股票代码": "code",
            "公告日期": "announcement_date",
            "货币资金": "cash",
            "资产-货币资金": "cash",
            "应收账款": "receivables",
            "资产-应收账款": "receivables",
            "存货": "inventory",
            "资产-存货": "inventory",
            "总资产": "total_assets",
            "资产-总资产": "total_assets",
            "总资产同比": "total_assets_yoy",
            "资产-总资产同比": "total_assets_yoy",
            "总负债": "total_liabilities",
            "负债-总负债": "total_liabilities",
            "资产负债率": "debt_ratio",
        },
        "balance",
    )
    cashflow_part = _statement_part(
        cashflow,
        {
            "股票代码": "code",
            "公告日期": "announcement_date",
            "经营性现金流净额": "operating_cashflow",
            "经营性现金流-现金流量净额": "operating_cashflow",
        },
        "cashflow",
    )
    result = income_part.merge(balance_part, on="code", how="outer")
    result = result.merge(cashflow_part, on="code", how="outer")
    announcement_columns = [
        column for column in result if column.endswith("_announcement_date")
    ]
    result["announcement_date"] = result[announcement_columns].max(axis=1)
    result["report_date"] = pd.to_datetime(report_date, format="%Y%m%d")
    result["net_margin"] = _safe_ratio(result["net_profit"], result["revenue"])
    result["cfo_to_profit"] = _safe_ratio(
        result["operating_cashflow"], result["net_profit"]
    )
    result["cash_to_assets"] = _safe_ratio(result["cash"], result["total_assets"])
    result["receivables_to_assets"] = _safe_ratio(
        result["receivables"], result["total_assets"]
    )
    result["inventory_to_assets"] = _safe_ratio(
        result["inventory"], result["total_assets"]
    )
    result = result.drop(columns=announcement_columns)
    return result.sort_values(["code", "report_date", "announcement_date"]).reset_index(
        drop=True
    )


def save_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temporary)
    os.replace(temporary, path)


def _fetch_with_retries(function, report_date: str, attempts: int = 3):
    last_error: Exception | None = None
    for attempt in range(int(attempts)):
        try:
            return function(date=report_date)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    assert last_error is not None
    raise last_error


def load_or_fetch_fundamentals(
    config: dict,
    cache_path: Path,
    ak_module,
) -> tuple[pd.DataFrame, dict]:
    cache_path = Path(cache_path)
    existing = (
        pd.read_pickle(cache_path)
        if cache_path.exists()
        else pd.DataFrame()
    )
    loaded = set()
    if not existing.empty and "report_date" in existing:
        loaded = set(
            pd.to_datetime(existing["report_date"])
            .dt.strftime("%Y%m%d")
            .dropna()
            .tolist()
        )
    frames = [existing] if not existing.empty else []
    failures: list[dict] = []
    requested = list(config.get("quarter_dates", QUARTER_DATES))
    for report_date in requested:
        if report_date in loaded:
            continue
        print(f"fundamentals: fetching {report_date}", flush=True)
        try:
            income = _fetch_with_retries(ak_module.stock_lrb_em, report_date)
            balance = _fetch_with_retries(ak_module.stock_zcfz_em, report_date)
            cashflow = _fetch_with_retries(ak_module.stock_xjll_em, report_date)
            frames.append(
                normalize_quarterly_statements(
                    report_date, income, balance, cashflow
                )
            )
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.drop_duplicates(
                ["code", "report_date", "announcement_date"], keep="last"
            )
            save_frame_atomic(cache_path, combined)
            print(
                f"fundamentals: cached {report_date} rows={len(combined)}",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {"report_date": report_date, "error": str(exc)}
            )
            print(
                f"fundamentals: failed {report_date}: {exc}",
                flush=True,
            )
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    if not result.empty:
        result = result.drop_duplicates(
            ["code", "report_date", "announcement_date"], keep="last"
        ).sort_values(["code", "announcement_date", "report_date"])
        save_frame_atomic(cache_path, result)
    loaded_dates = (
        sorted(pd.to_datetime(result["report_date"]).dt.strftime("%Y%m%d").unique())
        if not result.empty
        else []
    )
    quality = {
        "requested_quarters": requested,
        "loaded_quarters": loaded_dates,
        "row_count": int(len(result)),
        "minimum_announcement_date": (
            pd.to_datetime(result["announcement_date"]).min().date().isoformat()
            if not result.empty and result["announcement_date"].notna().any()
            else None
        ),
        "maximum_announcement_date": (
            pd.to_datetime(result["announcement_date"]).max().date().isoformat()
            if not result.empty and result["announcement_date"].notna().any()
            else None
        ),
        "failures": failures,
    }
    return result.reset_index(drop=True), quality


def _next_trading_date(
    announcement_dates: pd.Series, trading_dates: pd.DatetimeIndex
) -> pd.Series:
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize().unique()
    values = pd.to_datetime(announcement_dates, errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    positions = calendar.searchsorted(values, side="right")
    output = np.full(len(values), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = positions < len(calendar)
    output[valid] = calendar.to_numpy()[positions[valid]]
    return pd.Series(output, index=announcement_dates.index)


def align_fundamentals_point_in_time(
    signal_rows: pd.DataFrame,
    fundamentals: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    signals = signal_rows.copy()
    signals["date"] = (
        pd.to_datetime(signals["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    signals["code"] = _clean_code(signals["code"])
    signals["_row_order"] = np.arange(len(signals))
    statements = fundamentals.copy()
    if statements.empty:
        result = signals.copy()
        for column in FUNDAMENTAL_VALUE_COLUMNS:
            result[f"{column}_raw"] = np.nan
            result[column] = 0.5
        result["fundamental_age_days"] = np.nan
        result["fundamental_missing"] = 1.0
        return result.sort_values("_row_order").drop(columns="_row_order")
    statements["code"] = _clean_code(statements["code"])
    statements["announcement_date"] = pd.to_datetime(
        statements["announcement_date"], errors="coerce"
    ).dt.normalize()
    statements["available_date"] = _next_trading_date(
        statements["announcement_date"], trading_dates
    ).astype("datetime64[ns]")
    statements = statements.dropna(subset=["available_date"])
    merged_parts = []
    for code, group in signals.groupby("code", sort=False):
        right = statements[statements["code"] == code].sort_values(
            ["available_date", "report_date"]
        )
        left = group.sort_values("date")
        if right.empty:
            joined = left.copy()
            for column in statements.columns:
                if column not in joined and column != "code":
                    joined[column] = np.nan
        else:
            joined = pd.merge_asof(
                left,
                right.drop(columns="code"),
                left_on="date",
                right_on="available_date",
                direction="backward",
            )
        merged_parts.append(joined)
    result = pd.concat(merged_parts, ignore_index=True)
    result["fundamental_missing"] = result["report_date"].isna().astype(float)
    result["fundamental_age_days"] = (
        result["date"] - pd.to_datetime(result["announcement_date"])
    ).dt.days.astype("float64")
    for column in FUNDAMENTAL_VALUE_COLUMNS:
        raw_column = f"{column}_raw"
        result[raw_column] = pd.to_numeric(
            result.get(column), errors="coerce"
        )
        rank = result.groupby("date")[raw_column].rank(
            pct=True, method="average"
        )
        result[column] = rank.fillna(0.5)
    return result.sort_values("_row_order").drop(columns="_row_order").reset_index(
        drop=True
    )


def _rolling_slope_and_r2(values: pd.Series, window: int = 20) -> pd.DataFrame:
    array = np.log(
        pd.to_numeric(values, errors="coerce").where(
            pd.to_numeric(values, errors="coerce") > 0
        )
    )
    x = np.arange(window, dtype="float64")
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())

    def slope(sample: np.ndarray) -> float:
        if len(sample) != window or not np.isfinite(sample).all():
            return np.nan
        centered = sample - sample.mean()
        return float(np.dot(x_centered, centered) / denominator)

    def r_squared(sample: np.ndarray) -> float:
        if len(sample) != window or not np.isfinite(sample).all():
            return np.nan
        centered = sample - sample.mean()
        variance = float(np.square(centered).sum())
        if variance <= 1e-18:
            return 0.0
        covariance = float(np.dot(x_centered, centered))
        return float((covariance * covariance) / (denominator * variance))

    return pd.DataFrame(
        {
            "trend_slope_20": array.rolling(window, min_periods=window).apply(
                slope, raw=True
            ),
            "trend_r2_20": array.rolling(window, min_periods=window).apply(
                r_squared, raw=True
            ),
        },
        index=values.index,
    )


def _build_price_group(group: pd.DataFrame) -> pd.DataFrame:
    data = group.sort_values("date").copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    open_price = pd.to_numeric(data["open"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    amount = pd.to_numeric(data["amount"], errors="coerce")
    returns = close.pct_change()
    data["mom_3"] = close.pct_change(3)
    data["mom_10"] = close.pct_change(10)
    if "mom_5" not in data:
        data["mom_5"] = close.pct_change(5)
    if "mom_20" not in data:
        data["mom_20"] = close.pct_change(20)
    if "mom_60" not in data:
        data["mom_60"] = close.pct_change(60)
    if "ma_20" not in data:
        data["ma_20"] = close.rolling(20, min_periods=20).mean()
    if "drawdown_60" not in data:
        data["drawdown_60"] = (
            close / close.rolling(60, min_periods=60).max() - 1.0
        )
    volume_5 = volume.rolling(5, min_periods=5).mean()
    volume_20 = volume.rolling(20, min_periods=20).mean()
    amount_5 = amount.rolling(5, min_periods=5).mean()
    amount_20 = amount.rolling(20, min_periods=20).mean()
    data["volume_accel_5_20"] = _safe_ratio(volume_5, volume_20) - 1.0
    data["amount_accel_5_20"] = _safe_ratio(amount_5, amount_20) - 1.0
    data["up_day_ratio_20"] = (returns > 0).astype(float).rolling(
        20, min_periods=20
    ).mean()
    data["downside_volatility_20"] = returns.where(returns < 0, 0.0).rolling(
        20, min_periods=20
    ).std(ddof=0)
    trend = _rolling_slope_and_r2(close)
    data[["trend_slope_20", "trend_r2_20"]] = trend
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    relative_strength = _safe_ratio(gains, losses)
    data["rsi_14"] = 100.0 - 100.0 / (1.0 + relative_strength)
    mean_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std(ddof=0)
    data["bollinger_position_20"] = _safe_ratio(
        close - mean_20, 2.0 * std_20
    )
    data["opening_gap"] = _safe_ratio(open_price, close.shift(1)) - 1.0
    data["intraday_close_strength"] = _safe_ratio(
        close - low, high - low
    ).clip(0.0, 1.0)
    volume_change = volume.pct_change()
    data["volume_price_corr_20"] = returns.rolling(
        20, min_periods=20
    ).corr(volume_change)
    data["max_return_5"] = returns.rolling(5, min_periods=5).max()
    data["min_return_5"] = returns.rolling(5, min_periods=5).min()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.rolling(14, min_periods=14).mean()
    atr_60 = true_range.rolling(60, min_periods=60).mean()
    if "atr_14" not in data:
        data["atr_14"] = atr_14
    data["atr_contraction"] = _safe_ratio(atr_14, atr_60)
    if "liquidity_20" not in data:
        data["liquidity_20"] = np.log1p(
            amount.rolling(20, min_periods=20).mean()
        )
    return data


def build_price_features(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    output = [
        _build_price_group(group)
        for _code, group in data.groupby("code", sort=False)
    ]
    return pd.concat(output, ignore_index=True).sort_values(
        ["date", "code"]
    ).reset_index(drop=True)


def attach_benchmark_features(
    panel: pd.DataFrame, benchmark: pd.DataFrame
) -> pd.DataFrame:
    data = panel.copy()
    index = benchmark.sort_values("date").copy()
    index["date"] = pd.to_datetime(index["date"], errors="coerce").dt.normalize()
    index_close = pd.to_numeric(index["close"], errors="coerce")
    for days in (5, 20, 60):
        index[f"benchmark_mom{days}"] = index_close.pct_change(days)
    columns = ["date"] + [f"benchmark_mom{days}" for days in (5, 20, 60)]
    data = data.merge(index[columns].drop_duplicates("date"), on="date", how="left")
    for days in (5, 20, 60):
        data[f"excess_mom{days}"] = (
            pd.to_numeric(data[f"mom_{days}"], errors="coerce")
            - pd.to_numeric(data[f"benchmark_mom{days}"], errors="coerce")
        )
    return data


def attach_industry_features(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["industry"] = data.get(
        "industry", pd.Series("", index=data.index)
    ).fillna("").astype(str)
    industry_group = data.groupby(["date", "industry"], dropna=False)
    for days in (5, 20, 60):
        data[f"industry_median_mom{days}"] = industry_group[
            f"mom_{days}"
        ].transform("median")
    if "ma_20" in data:
        data["_above_ma20"] = (
            pd.to_numeric(data["close"], errors="coerce")
            > pd.to_numeric(data["ma_20"], errors="coerce")
        ).astype(float)
    else:
        data["_above_ma20"] = np.nan
    data["industry_breadth_ma20"] = industry_group["_above_ma20"].transform(
        "mean"
    )
    data["industry_mom20_dispersion"] = industry_group["mom_20"].transform(
        "std"
    )
    data["stock_minus_industry_mom20"] = (
        pd.to_numeric(data["mom_20"], errors="coerce")
        - data["industry_median_mom20"]
    )
    industry_daily = (
        data[["date", "industry", "industry_median_mom20"]]
        .drop_duplicates(["date", "industry"])
        .copy()
    )
    industry_daily["industry_strength_rank"] = industry_daily.groupby("date")[
        "industry_median_mom20"
    ].rank(pct=True, method="average")
    data = data.merge(
        industry_daily[["date", "industry", "industry_strength_rank"]],
        on=["date", "industry"],
        how="left",
    )
    return data.drop(columns="_above_ma20")


def _feature_matrix(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    return (
        frame.loc[:, list(feature_names)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype="float64")
    )


def fit_submodels(fit_rows: pd.DataFrame, l2: float) -> dict:
    labels = pd.to_numeric(fit_rows["label_5"], errors="coerce").to_numpy(
        dtype="float64"
    )
    return {
        "trend": {
            "feature_names": list(TREND_FEATURES),
            "model": fit_logistic(
                _feature_matrix(fit_rows, TREND_FEATURES), labels, l2
            ),
        },
        "reversal": {
            "feature_names": list(REVERSAL_FEATURES),
            "model": fit_logistic(
                _feature_matrix(fit_rows, REVERSAL_FEATURES), labels, l2
            ),
        },
        "quality": {
            "feature_names": list(QUALITY_FEATURES),
            "model": fit_logistic(
                _feature_matrix(fit_rows, QUALITY_FEATURES), labels, l2
            ),
        },
    }


def predict_submodels(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=rows.index)
    for name in ("trend", "reversal", "quality"):
        specification = models[name]
        feature_names = tuple(specification["feature_names"])
        output[name] = predict_raw_probability(
            specification["model"],
            _feature_matrix(rows, feature_names),
        )
    return output


def combine_probabilities(
    predictions: pd.DataFrame, weights: dict
) -> np.ndarray:
    combined = np.zeros(len(predictions), dtype="float64")
    for name in ("trend", "reversal", "quality"):
        combined += (
            pd.to_numeric(predictions[name], errors="coerce")
            .fillna(0.5)
            .to_numpy(dtype="float64")
            * float(weights.get(name, 0.0))
        )
    return np.clip(combined, 0.0, 1.0)


def choose_ensemble(
    calibration_rows: pd.DataFrame,
    models: dict,
    config: dict,
) -> dict:
    predictions = predict_submodels(models, calibration_rows)
    labels = pd.to_numeric(
        calibration_rows["label_5"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype="float64")
    returns = pd.to_numeric(
        calibration_rows.get(
            "forward_5", pd.Series(np.nan, index=calibration_rows.index)
        ),
        errors="coerce",
    ).to_numpy(dtype="float64")
    candidates = []
    for index, weights in enumerate(config["ensemble_weights"]):
        raw = combine_probabilities(predictions, weights)
        calibration = fit_probability_calibrator(
            raw,
            labels,
            bins=int(config.get("calibration_bins", 8)),
            alpha=float(config.get("beta_alpha", 1.0)),
            beta=float(config.get("beta_beta", 1.0)),
        )
        calibrated = apply_probability_calibrator(raw, calibration)
        selected = calibrated >= 0.5
        selected_count = int(selected.sum())
        win_rate = (
            float(labels[selected].mean()) if selected_count else 0.0
        )
        mean_return = (
            float(np.nanmean(returns[selected])) if selected_count else 0.0
        )
        activity_rate = selected_count / max(len(calibration_rows), 1)
        metrics = {
            "rows": int(len(calibration_rows)),
            "selected": selected_count,
            "win_rate": win_rate,
            "mean_net_return": mean_return,
            "activity_rate": float(activity_rate),
        }
        key = (
            win_rate >= 0.85,
            activity_rate >= 0.20,
            win_rate,
            mean_return,
            activity_rate,
            -index,
        )
        candidates.append(
            {
                "name": str(weights["name"]),
                "weights": {
                    name: float(weights[name])
                    for name in ("trend", "reversal", "quality")
                },
                "calibration": calibration,
                "metrics": metrics,
                "_key": key,
            }
        )
    chosen = max(candidates, key=lambda item: item["_key"])
    return {key: value for key, value in chosen.items() if key != "_key"}


def build_v2_features(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    fundamentals: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    data = attach_industry_features(
        attach_benchmark_features(build_price_features(panel), benchmark)
    )
    if "breadth_above_ma20" not in data:
        data["_above_ma20_market"] = (
            pd.to_numeric(data["close"], errors="coerce")
            > pd.to_numeric(data["ma_20"], errors="coerce")
        ).astype(float)
        grouped = data.groupby("date")
        data["breadth_above_ma20"] = grouped[
            "_above_ma20_market"
        ].transform("mean")
        data["market_median_mom20"] = grouped["mom_20"].transform("median")
        data["market_median_drawdown60"] = grouped[
            "drawdown_60"
        ].transform("median")
        data = data.drop(columns="_above_ma20_market")
    for column in (
        "breadth_above_ma20",
        "market_median_mom20",
        "market_median_drawdown60",
    ):
        data[f"{column}_raw"] = pd.to_numeric(data[column], errors="coerce")
    data = align_fundamentals_point_in_time(
        data, fundamentals, trading_dates
    )
    data["fundamental_age_rank"] = data.groupby("date")[
        "fundamental_age_days"
    ].rank(pct=True, method="average").fillna(0.5)
    rank_columns = [
        feature
        for feature in TREND_FEATURES + REVERSAL_FEATURES + QUALITY_FEATURES
        if feature not in FUNDAMENTAL_VALUE_COLUMNS
        and feature not in {"fundamental_missing", "fundamental_age_rank"}
    ]
    for column in rank_columns:
        if column not in data:
            data[column] = np.nan
        data[f"{column}_raw"] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data.groupby("date")[f"{column}_raw"].rank(
            pct=True, method="average"
        ).fillna(0.5)
    data["forward_5"] = calculate_high_precision_forward_5(data)
    data["label_5"] = np.where(
        data["forward_5"].notna(),
        (data["forward_5"] > 0).astype(int),
        np.nan,
    )
    return data.replace([np.inf, -np.inf], np.nan)


def _benchmark_forward_returns(
    benchmark: pd.DataFrame, cost: float
) -> pd.DataFrame:
    data = benchmark.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if "code" not in data:
        data["code"] = "000300"
    data["benchmark_forward_5"] = calculate_high_precision_forward_5(
        data, cost=cost
    )
    return data[["date", "benchmark_forward_5"]].drop_duplicates("date")


def v2_parameter_grid(config: dict) -> list[dict]:
    output = []
    for l2, market, threshold, delta, max_picks in product(
        config["l2_values"],
        config["market_policies"],
        config["strong_thresholds"],
        config["normal_threshold_deltas"],
        config["max_picks_values"],
    ):
        output.append(
            {
                "l2": float(l2),
                "policy": {
                    **{key: float(value) for key, value in market.items()},
                    "strong_threshold": float(threshold),
                    "normal_threshold_delta": float(delta),
                    "max_picks": int(max_picks),
                },
            }
        )
    return output


def _json_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for item in frame.to_dict("records"):
        normalized = {}
        for key, value in item.items():
            if isinstance(value, (pd.Timestamp, datetime, date)):
                normalized[key] = pd.Timestamp(value).date().isoformat()
            elif isinstance(value, np.generic):
                normalized[key] = value.item()
            elif pd.isna(value):
                normalized[key] = None
            else:
                normalized[key] = value
        records.append(normalized)
    return records


def _predict_v2_rows(
    rows: pd.DataFrame,
    models: dict,
    ensemble: dict,
) -> pd.DataFrame:
    output = rows.copy()
    submodel_predictions = predict_submodels(models, output)
    for name in ("trend", "reversal", "quality"):
        output[f"{name}_probability"] = submodel_predictions[name].to_numpy()
    raw = combine_probabilities(
        submodel_predictions, ensemble["weights"]
    )
    output["raw_probability"] = raw
    output["calibrated_probability"] = apply_probability_calibrator(
        raw, ensemble["calibration"]
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


def _neighbor_values(value, configured: list) -> list:
    values = list(configured)
    position = values.index(value)
    return [
        values[index]
        for index in (position - 1, position + 1)
        if 0 <= index < len(values)
    ]


def run_v2_walk_forward_search(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: dict,
    folds: list[dict],
) -> dict:
    raw = panel.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    trading_dates = pd.DatetimeIndex(sorted(raw["date"].dropna().unique()))
    features = build_v2_features(
        raw, benchmark, fundamentals, trading_dates
    )
    features = features.merge(
        _benchmark_forward_returns(
            benchmark, float(config["round_trip_cost"])
        ),
        on="date",
        how="left",
    )
    features = features.dropna(
        subset=["forward_5", "benchmark_forward_5", "label_5"]
    ).copy()
    grid = v2_parameter_grid(config)
    attempts: list[dict] = []
    fold_outputs: list[dict] = []
    fold_runtime: list[dict] = []
    selected_parts: list[pd.DataFrame] = []
    scheduled_dates: list[pd.Timestamp] = []
    minimum_months = int(config.get("minimum_training_months", 18))

    for fold_index, fold in enumerate(folds):
        train_end = pd.Timestamp(fold["train_end"]).normalize()
        valid_start = pd.Timestamp(fold["valid_start"]).normalize()
        valid_end = pd.Timestamp(fold["valid_end"]).normalize()
        if not train_end < valid_start <= valid_end:
            raise ValueError("fold dates must be chronological")
        train_rows = features[features["date"] <= train_end].copy()
        valid_rows = features[
            features["date"].between(valid_start, valid_end)
        ].copy()
        if train_rows.empty or valid_rows.empty:
            raise ValueError("fold contains no training or validation rows")
        if minimum_months:
            required_start = train_end - pd.DateOffset(months=minimum_months)
            if train_rows["date"].min() > required_start:
                raise ValueError("training window is shorter than required months")
        fit_rows, calibration_rows = split_fit_and_calibration(
            train_rows, int(config["calibration_trading_days"])
        )
        calibration_dates = sorted(calibration_rows["date"].unique())
        valid_dates = sorted(valid_rows["date"].unique())
        scheduled_dates.extend(pd.Timestamp(value) for value in valid_dates)
        l2_runtime: dict[float, dict] = {}
        for l2 in config["l2_values"]:
            models = fit_submodels(fit_rows, float(l2))
            ensemble = choose_ensemble(
                calibration_rows, models, config
            )
            calibration_predictions = _predict_v2_rows(
                calibration_rows, models, ensemble
            )
            validation_predictions = _predict_v2_rows(
                valid_rows, models, ensemble
            )
            l2_runtime[float(l2)] = {
                "models": models,
                "ensemble": ensemble,
                "validation_predictions": validation_predictions,
            }
            for grid_index, candidate in enumerate(grid):
                if float(candidate["l2"]) != float(l2):
                    continue
                selected = select_daily_candidates(
                    calibration_predictions,
                    HighPrecisionPolicy.from_dict(candidate["policy"]),
                )
                metrics = evaluate_selected_trades(
                    selected,
                    calibration_dates,
                    [
                        {
                            "valid_start": pd.Timestamp(
                                calibration_dates[0]
                            ).date().isoformat(),
                            "valid_end": pd.Timestamp(
                                calibration_dates[-1]
                            ).date().isoformat(),
                        }
                    ],
                )
                eligible = (
                    metrics["activity_rate"]
                    >= float(
                        config.get(
                            "calibration_minimum_activity_rate", 0.2
                        )
                    )
                    and metrics["trades"]
                    >= int(config.get("calibration_minimum_trades", 20))
                )
                attempts.append(
                    {
                        "fold_index": fold_index,
                        "grid_index": grid_index,
                        "parameters": {
                            **candidate,
                            "ensemble": {
                                "name": ensemble["name"],
                                "weights": ensemble["weights"],
                            },
                        },
                        "eligible": bool(eligible),
                        "metrics": metrics,
                    }
                )
        fold_attempts = [
            item for item in attempts if item["fold_index"] == fold_index
        ]
        chosen_attempt = max(fold_attempts, key=_calibration_candidate_key)
        chosen = chosen_attempt["parameters"]
        runtime = l2_runtime[float(chosen["l2"])]
        selected = select_daily_candidates(
            runtime["validation_predictions"],
            HighPrecisionPolicy.from_dict(chosen["policy"]),
        )
        validation_metrics = evaluate_selected_trades(
            selected, valid_dates, [fold]
        )
        selected_parts.append(selected)
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
                "validation_min_date": valid_rows[
                    "date"
                ].min().date().isoformat(),
                "validation_max_date": valid_rows[
                    "date"
                ].max().date().isoformat(),
                "chosen": chosen,
                "calibration_metrics": chosen_attempt["metrics"],
                "validation_metrics": validation_metrics,
                "validation_prediction_rows": int(
                    len(runtime["validation_predictions"])
                ),
                "validation_selections": _json_records(selected),
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
    modal_key = Counter(
        json.dumps(item["chosen"], sort_keys=True)
        for item in fold_runtime
    ).most_common(1)[0][0]
    modal_chosen = json.loads(modal_key)
    neighbor_specs = [
        {"kind": "strong_threshold", "value": float(value)}
        for value in _neighbor_values(
            float(modal_chosen["policy"]["strong_threshold"]),
            config["strong_thresholds"],
        )
    ]
    neighbor_specs.extend(
        {"kind": "l2", "value": float(value)}
        for value in _neighbor_values(
            float(modal_chosen["l2"]), config["l2_values"]
        )
    )
    neighbor_metrics = []
    for neighbor in neighbor_specs:
        parts = []
        dates = []
        for runtime in fold_runtime:
            chosen = runtime["chosen"]
            policy_data = dict(chosen["policy"])
            l2_value = float(chosen["l2"])
            if neighbor["kind"] == "strong_threshold":
                policy_data["strong_threshold"] = neighbor["value"]
            else:
                l2_value = neighbor["value"]
            predictions = runtime["l2_runtime"][l2_value][
                "validation_predictions"
            ]
            parts.append(
                select_daily_candidates(
                    predictions,
                    HighPrecisionPolicy.from_dict(policy_data),
                )
            )
            dates.extend(runtime["valid_dates"])
        selected_neighbor = (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        )
        metrics = evaluate_selected_trades(
            selected_neighbor, sorted(set(dates)), folds
        )
        neighbor_metrics.append({**neighbor, **metrics})
    deployment = deployment_checks(
        aggregate_metrics,
        neighbor_metrics,
        config["deployment_gates"],
    )
    run_id = datetime.now().strftime("hpv2-%Y%m%d-%H%M%S")
    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "walk_forward": {"folds": fold_outputs},
        "aggregate_metrics": aggregate_metrics,
        "neighbor_metrics": neighbor_metrics,
        "chosen": modal_chosen,
        "deployment": deployment,
    }


def fit_final_v2_model(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: dict,
    chosen: dict,
    run_id: str,
    deployment: dict,
) -> dict:
    raw = panel.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    features = build_v2_features(
        raw,
        benchmark,
        fundamentals,
        pd.DatetimeIndex(sorted(raw["date"].dropna().unique())),
    ).dropna(subset=["forward_5", "label_5"])
    fit_rows, calibration_rows = split_fit_and_calibration(
        features, int(config["calibration_trading_days"])
    )
    models = fit_submodels(fit_rows, float(chosen["l2"]))
    predictions = predict_submodels(models, calibration_rows)
    weights = chosen["ensemble"]["weights"]
    raw_probability = combine_probabilities(predictions, weights)
    calibration = fit_probability_calibrator(
        raw_probability,
        calibration_rows["label_5"].to_numpy(dtype="float64"),
        bins=int(config["calibration_bins"]),
        alpha=float(config["beta_alpha"]),
        beta=float(config["beta_beta"]),
    )
    return {
        "schema_version": "high-precision-v2",
        "run_id": run_id,
        "submodels": models,
        "ensemble": {
            "name": chosen["ensemble"]["name"],
            "weights": weights,
            "calibration": calibration,
        },
        "policy": chosen["policy"],
        "deployment": deployment,
        "fundamental_cache": "data/fundamentals/quarterly-point-in-time.pkl",
    }


def run_optimizer_cli(args: argparse.Namespace) -> dict:
    import akshare as ak

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cached = pd.read_pickle(args.panel_cache)
    panel = cached["panel"] if isinstance(cached, dict) else cached
    benchmark = pd.read_csv(args.benchmark_cache)
    fundamentals, data_quality = load_or_fetch_fundamentals(
        config, args.fundamental_cache, ak
    )
    loaded = set(data_quality["loaded_quarters"])
    requested = set(data_quality["requested_quarters"])
    data_quality["passed"] = (
        len(fundamentals) > 0
        and len(loaded & requested) >= max(12, int(len(requested) * 0.8))
    )
    if not data_quality["passed"]:
        raise RuntimeError(
            "financial data quality gate failed: "
            + json.dumps(data_quality, ensure_ascii=False)
        )
    search = run_v2_walk_forward_search(
        panel, benchmark, fundamentals, config, default_folds()
    )
    model = fit_final_v2_model(
        panel,
        benchmark,
        fundamentals,
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
            "round_trip_cost": float(config["round_trip_cost"]),
            "calibration_trading_days": int(
                config["calibration_trading_days"]
            ),
            "walk_forward_folds": default_folds(),
        },
        "coverage": {
            "actual_start": dates.min().date().isoformat(),
            "actual_end": dates.max().date().isoformat(),
            "usable_symbols": int(panel["code"].nunique()),
            "stock_dates": int(len(panel)),
        },
        "fundamental_data_quality": data_quality,
        "caveats": [
            "Current-industry metadata introduces survivorship bias.",
            "Rolling out-of-sample performance does not guarantee future returns.",
        ],
    }
    verification = {
        "run_id": report["run_id"],
        "generated_at": report["generated_at"],
        "fold_metrics": [
            fold["validation_metrics"]
            for fold in report["walk_forward"]["folds"]
        ],
        "aggregate_metrics": report["aggregate_metrics"],
        "neighbor_metrics": report["neighbor_metrics"],
        "fundamental_data_quality": data_quality,
        "deployment": report["deployment"],
    }
    save_json_atomic(args.output, report)
    save_json_atomic(args.candidate_model, model)
    save_json_atomic(args.verification, verification)
    if report["deployment"]["passed"]:
        save_json_atomic(args.production_model, model)
        save_json_atomic(args.runtime_report, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InStock high precision V2 picker"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize")
    optimize.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("high_precision_v2_config.json"),
    )
    optimize.add_argument(
        "--panel-cache",
        type=Path,
        default=Path(
            r"C:\InStockLocal\backtest_cache\policy_panel_600_20260714.pkl"
        ),
    )
    optimize.add_argument(
        "--benchmark-cache",
        type=Path,
        default=Path(r"C:\InStockLocal\backtest_cache\indices\000300.csv"),
    )
    optimize.add_argument(
        "--fundamental-cache",
        type=Path,
        default=Path(
            r"C:\InStockLocal\InStockDesktopApp\data\fundamentals"
            r"\quarterly-point-in-time.pkl"
        ),
    )
    optimize.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"C:\InStockLocal\InStockDesktopApp\data"
            r"\high_precision-v2-backtest-latest.json"
        ),
    )
    optimize.add_argument(
        "--candidate-model",
        type=Path,
        default=Path(
            r"C:\InStockLocal\InStockDesktopApp\work"
            r"\candidate-high_precision_v2_model.json"
        ),
    )
    optimize.add_argument(
        "--verification",
        type=Path,
        default=Path(
            r"C:\InStockLocal\InStockDesktopApp\verification"
            r"\high-precision-v2-2026-07-17.json"
        ),
    )
    optimize.add_argument(
        "--production-model",
        type=Path,
        default=Path(__file__).with_name("high_precision_model.json"),
    )
    optimize.add_argument(
        "--runtime-report",
        type=Path,
        default=Path(__file__).parent.parent
        / "data"
        / "high_precision-backtest-latest.json",
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
                    "aggregate_metrics": report["aggregate_metrics"],
                    "deployment": report["deployment"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
