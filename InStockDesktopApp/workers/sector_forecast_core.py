from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "ret_3",
    "ret_5",
    "ret_10",
    "amount_ratio_5_20",
    "turnover_delta_5_20",
    "volatility_20",
    "drawdown_60",
    "amplitude_10",
)


@dataclass(frozen=True)
class RidgeModel:
    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = _numeric_matrix(frame, self.columns)
        return ((matrix - self.mean) / self.scale) @ self.coef + self.intercept


def _numeric_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    data = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    return data.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def _group_features(group: pd.DataFrame) -> pd.DataFrame:
    data = group.sort_values("date").copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    benchmark = pd.to_numeric(data["benchmark_close"], errors="coerce")
    for days in (3, 5, 10, 20, 60):
        data[f"ret_{days}"] = close.pct_change(days, fill_method=None) - benchmark.pct_change(
            days, fill_method=None
        )
    log_close = np.log(close.where(close > 0))
    data["trend_20"] = log_close.diff(19) / 19.0
    amount = pd.to_numeric(data["amount"], errors="coerce")
    turnover = pd.to_numeric(data["turnover"], errors="coerce")
    data["amount_ratio_5_20"] = amount.rolling(5).mean() / amount.rolling(20).mean() - 1.0
    data["turnover_delta_5_20"] = turnover.rolling(5).mean() - turnover.rolling(20).mean()
    data["volatility_20"] = close.pct_change(fill_method=None).rolling(20).std()
    data["drawdown_60"] = close / close.rolling(60).max() - 1.0
    data["amplitude_10"] = (
        (pd.to_numeric(data["high"], errors="coerce") - pd.to_numeric(data["low"], errors="coerce"))
        / close
    ).rolling(10).mean()
    data["target_excess_5"] = close.shift(-5) / close - benchmark.shift(-5) / benchmark
    return data


def build_feature_panel(history: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sector_type",
        "sector_id",
        "sector_name",
        "date",
        "high",
        "low",
        "close",
        "amount",
        "turnover",
        "benchmark_close",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError("板块历史缺少字段：" + ",".join(missing))
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date", "sector_id", "close", "benchmark_close"])
    parts = [
        _group_features(group)
        for _key, group in data.groupby(["sector_type", "sector_id"], sort=False)
    ]
    if not parts:
        return data.assign(target_excess_5=np.nan)
    return pd.concat(parts, ignore_index=True)


def build_live_rows(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(as_of).normalize()
    return panel.loc[pd.to_datetime(panel["date"]).dt.normalize() == day].copy()


def fit_ridge(train: pd.DataFrame, alpha: float = 8.0) -> RidgeModel:
    clean = train.dropna(subset=["target_excess_5"]).copy()
    if len(clean) < len(FEATURE_COLUMNS) + 2:
        raise ValueError("可训练样本不足")
    matrix = _numeric_matrix(clean, FEATURE_COLUMNS)
    target = pd.to_numeric(clean["target_excess_5"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(target)
    matrix = matrix[finite]
    target = target[finite]
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (matrix - mean) / scale
    intercept = float(target.mean())
    centered_target = target - intercept
    gram = standardized.T @ standardized + float(alpha) * np.eye(standardized.shape[1])
    coef = np.linalg.solve(gram, standardized.T @ centered_target)
    return RidgeModel(FEATURE_COLUMNS, mean, scale, coef, intercept)


def _rank_metrics(frame: pd.DataFrame, score_column: str, top_k: int = 10) -> dict[str, float]:
    daily_excess: list[float] = []
    daily_win_rate: list[float] = []
    daily_overlap: list[float] = []
    daily_ic: list[float] = []
    for _day, group in frame.groupby("date"):
        data = group.dropna(subset=[score_column, "target_excess_5"])
        if len(data) < 3:
            continue
        count = min(top_k, len(data))
        predicted = data.nlargest(count, score_column)
        actual = data.nlargest(count, "target_excess_5")
        daily_excess.append(float(predicted["target_excess_5"].mean()))
        daily_win_rate.append(float((predicted["target_excess_5"] > 0).mean()))
        daily_overlap.append(len(set(predicted["sector_id"]) & set(actual["sector_id"])) / count)
        correlation = data[score_column].rank().corr(data["target_excess_5"].rank())
        if pd.notna(correlation):
            daily_ic.append(float(correlation))
    prefix = f"top{top_k}"
    return {
        f"{prefix}_mean_excess": float(np.mean(daily_excess)) if daily_excess else 0.0,
        f"{prefix}_win_rate": float(np.mean(daily_win_rate)) if daily_win_rate else 0.0,
        f"{prefix}_overlap": float(np.mean(daily_overlap)) if daily_overlap else 0.0,
        "rank_ic": float(np.mean(daily_ic)) if daily_ic else 0.0,
        "periods": float(len(daily_excess)),
    }


def _short_momentum_score(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=frame.index, dtype=float)
    for column, weight in (("ret_3", 0.45), ("ret_5", 0.35), ("ret_10", 0.20)):
        numeric = pd.to_numeric(frame.get(column), errors="coerce")
        if "date" in frame:
            ranked = numeric.groupby(frame["date"]).rank(pct=True, method="average")
        else:
            ranked = numeric.rank(pct=True, method="average")
        score += float(weight) * ranked.fillna(0.5)
    return score


def stable_outperformance(
    *,
    candidate_mean: float,
    baseline_mean: float,
    candidate_folds: list[float],
    baseline_folds: list[float],
) -> tuple[bool, int]:
    superior_folds = sum(
        float(candidate) > float(baseline)
        for candidate, baseline in zip(candidate_folds, baseline_folds)
    )
    comparable = len(candidate_folds) == len(baseline_folds) and len(candidate_folds) >= 4
    passed = (
        comparable
        and float(candidate_mean) > float(baseline_mean)
        and superior_folds >= 3
    )
    return passed, superior_folds


def walk_forward_validate(panel: pd.DataFrame) -> dict[str, Any]:
    labeled = panel.dropna(subset=["target_excess_5"]).copy()
    dates = sorted(pd.Timestamp(value) for value in labeled["date"].unique())
    if len(dates) < 300:
        return {
            "passed": False,
            "quality": 0.0,
            "reasons": ["有效历史少于300个交易日，无法完成滚动验证"],
            "folds": [],
            "model_metrics": _rank_metrics(pd.DataFrame(), "model_score"),
            "baseline_metrics": _rank_metrics(pd.DataFrame(), "ret_5"),
            "legacy_metrics": _rank_metrics(pd.DataFrame(), "ret_20"),
        }

    validation_days = 60
    starts = [len(dates) - validation_days * index for index in (4, 3, 2, 1)]
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for start_position in starts:
        if start_position < 126:
            continue
        validation_dates = dates[start_position : start_position + validation_days]
        train_end_position = start_position - 6
        train_dates = dates[: train_end_position + 1]
        train = labeled[labeled["date"].isin(train_dates)]
        validation = labeled[labeled["date"].isin(validation_dates)].copy()
        if train.empty or validation.empty:
            continue
        model = fit_ridge(train)
        validation["model_score"] = model.predict(validation)
        validation["factor_score"] = _short_momentum_score(validation)
        validation["legacy_score"] = validation.groupby("date")["ret_20"].rank(
            pct=True, method="average"
        )
        validation["model_percentile"] = validation.groupby("date")["model_score"].rank(
            pct=True, method="average"
        )
        validation["hybrid_score"] = (
            0.80 * validation["factor_score"]
            + 0.20 * validation["model_percentile"]
        )
        model_fold = _rank_metrics(validation, "model_score")
        factor_fold = _rank_metrics(validation, "factor_score")
        hybrid_fold = _rank_metrics(validation, "hybrid_score")
        baseline_fold = _rank_metrics(validation, "ret_5")
        legacy_fold = _rank_metrics(validation, "legacy_score")
        model_head_fold = _rank_metrics(validation, "model_score", top_k=5)
        factor_head_fold = _rank_metrics(validation, "factor_score", top_k=5)
        hybrid_head_fold = _rank_metrics(validation, "hybrid_score", top_k=5)
        baseline_head_fold = _rank_metrics(validation, "ret_5", top_k=5)
        legacy_head_fold = _rank_metrics(validation, "legacy_score", top_k=5)
        folds.append(
            {
                "train_end": dates[train_end_position].date().isoformat(),
                "validation_start": validation_dates[0].date().isoformat(),
                "validation_end": validation_dates[-1].date().isoformat(),
                "model_top10_mean_excess": model_fold["top10_mean_excess"],
                "factor_top10_mean_excess": factor_fold["top10_mean_excess"],
                "hybrid_top10_mean_excess": hybrid_fold["top10_mean_excess"],
                "baseline_top10_mean_excess": baseline_fold["top10_mean_excess"],
                "legacy_top10_mean_excess": legacy_fold["top10_mean_excess"],
                "model_rank_ic": model_fold["rank_ic"],
                "factor_rank_ic": factor_fold["rank_ic"],
                "hybrid_rank_ic": hybrid_fold["rank_ic"],
                "model_top5_mean_excess": model_head_fold["top5_mean_excess"],
                "factor_top5_mean_excess": factor_head_fold["top5_mean_excess"],
                "hybrid_top5_mean_excess": hybrid_head_fold["top5_mean_excess"],
                "baseline_top5_mean_excess": baseline_head_fold["top5_mean_excess"],
                "legacy_top5_mean_excess": legacy_head_fold["top5_mean_excess"],
            }
        )
        predictions.append(validation)

    combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    model_metrics = _rank_metrics(combined, "model_score")
    factor_metrics = _rank_metrics(combined, "factor_score")
    hybrid_metrics = _rank_metrics(combined, "hybrid_score")
    baseline_metrics = _rank_metrics(combined, "ret_5")
    legacy_metrics = _rank_metrics(combined, "legacy_score")
    model_head_metrics = _rank_metrics(combined, "model_score", top_k=5)
    factor_head_metrics = _rank_metrics(combined, "factor_score", top_k=5)
    hybrid_head_metrics = _rank_metrics(combined, "hybrid_score", top_k=5)
    baseline_head_metrics = _rank_metrics(combined, "ret_5", top_k=5)
    legacy_head_metrics = _rank_metrics(combined, "legacy_score", top_k=5)
    candidates = {
        "factor": factor_metrics,
        "model": model_metrics,
        "hybrid": hybrid_metrics,
    }
    selected_mode, selected_metrics = max(
        candidates.items(), key=lambda item: item[1]["top10_mean_excess"]
    )
    head_candidates = {
        "factor": factor_head_metrics,
        "model": model_head_metrics,
        "hybrid": hybrid_head_metrics,
    }
    head_selected_mode, head_selected_metrics = max(
        head_candidates.items(), key=lambda item: item[1]["top5_mean_excess"]
    )
    positive_folds = sum(
        float(fold[f"{selected_mode}_top10_mean_excess"]) > 0 for fold in folds
    )
    selected_metrics["positive_folds"] = float(positive_folds)
    baseline_outperformance, baseline_superior_folds = stable_outperformance(
        candidate_mean=selected_metrics["top10_mean_excess"],
        baseline_mean=baseline_metrics["top10_mean_excess"],
        candidate_folds=[float(fold[f"{selected_mode}_top10_mean_excess"]) for fold in folds],
        baseline_folds=[float(fold["baseline_top10_mean_excess"]) for fold in folds],
    )
    selected_metrics["baseline_superior_folds"] = float(baseline_superior_folds)
    legacy_outperformance, legacy_superior_folds = stable_outperformance(
        candidate_mean=selected_metrics["top10_mean_excess"],
        baseline_mean=legacy_metrics["top10_mean_excess"],
        candidate_folds=[float(fold[f"{selected_mode}_top10_mean_excess"]) for fold in folds],
        baseline_folds=[float(fold["legacy_top10_mean_excess"]) for fold in folds],
    )
    selected_metrics["legacy_superior_folds"] = float(legacy_superior_folds)
    model_metrics["positive_folds"] = float(
        sum(float(fold["model_top10_mean_excess"]) > 0 for fold in folds)
    )
    head_positive_folds = sum(
        float(fold[f"{head_selected_mode}_top5_mean_excess"]) > 0 for fold in folds
    )
    head_selected_metrics["positive_folds"] = float(head_positive_folds)
    head_baseline_outperformance, head_baseline_superior_folds = stable_outperformance(
        candidate_mean=head_selected_metrics["top5_mean_excess"],
        baseline_mean=baseline_head_metrics["top5_mean_excess"],
        candidate_folds=[float(fold[f"{head_selected_mode}_top5_mean_excess"]) for fold in folds],
        baseline_folds=[float(fold["baseline_top5_mean_excess"]) for fold in folds],
    )
    head_selected_metrics["baseline_superior_folds"] = float(head_baseline_superior_folds)
    head_legacy_outperformance, head_legacy_superior_folds = stable_outperformance(
        candidate_mean=head_selected_metrics["top5_mean_excess"],
        baseline_mean=legacy_head_metrics["top5_mean_excess"],
        candidate_folds=[float(fold[f"{head_selected_mode}_top5_mean_excess"]) for fold in folds],
        baseline_folds=[float(fold["legacy_top5_mean_excess"]) for fold in folds],
    )
    head_selected_metrics["legacy_superior_folds"] = float(head_legacy_superior_folds)
    checks = {
        "top10_mean_excess": selected_metrics["top10_mean_excess"] > 0
        and selected_metrics["top10_mean_excess"] > baseline_metrics["top10_mean_excess"],
        "rank_ic": selected_metrics["rank_ic"] > baseline_metrics["rank_ic"],
        "fold_stability": len(folds) >= 4 and positive_folds >= 3,
        "baseline_outperformance": baseline_outperformance,
        "legacy_outperformance": legacy_outperformance,
    }
    passed = all(checks.values())
    head_checks = {
        "top5_mean_excess": head_selected_metrics["top5_mean_excess"] > 0
        and head_selected_metrics["top5_mean_excess"]
        > baseline_head_metrics["top5_mean_excess"],
        "top5_win_rate": head_selected_metrics["top5_win_rate"]
        > baseline_head_metrics["top5_win_rate"],
        "rank_ic": head_selected_metrics["rank_ic"] > 0,
        "fold_stability": len(folds) >= 4 and head_positive_folds >= 3,
        "baseline_outperformance": head_baseline_outperformance,
        "legacy_outperformance": head_legacy_outperformance,
    }
    head_passed = all(head_checks.values())
    relative_edge = selected_metrics["top10_mean_excess"] - max(
        baseline_metrics["top10_mean_excess"], legacy_metrics["top10_mean_excess"]
    )
    quality = float(
        np.clip(
            0.35
            + 8.0 * max(0.0, relative_edge)
            + 0.8 * max(0.0, selected_metrics["rank_ic"])
            + 0.05 * positive_folds,
            0.0,
            1.0,
        )
    )
    return {
        "passed": passed,
        "head_passed": head_passed,
        "quality": quality,
        "reasons": [name for name, result in checks.items() if not result],
        "head_reasons": [name for name, result in head_checks.items() if not result],
        "head_checks": head_checks,
        "checks": checks,
        "folds": folds,
        "selected_mode": selected_mode,
        "head_selected_mode": head_selected_mode,
        "model_weight": 1.0 if selected_mode == "model" else 0.20 if selected_mode == "hybrid" else 0.0,
        "selected_metrics": selected_metrics,
        "head_selected_metrics": head_selected_metrics,
        "model_metrics": model_metrics,
        "factor_metrics": factor_metrics,
        "hybrid_metrics": hybrid_metrics,
        "baseline_metrics": baseline_metrics,
        "legacy_metrics": legacy_metrics,
        "model_head_metrics": model_head_metrics,
        "factor_head_metrics": factor_head_metrics,
        "hybrid_head_metrics": hybrid_head_metrics,
        "baseline_head_metrics": baseline_head_metrics,
        "legacy_head_metrics": legacy_head_metrics,
    }


def _percentile(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranked = numeric.rank(pct=True, method="average")
    ranked = ranked.fillna(0.5)
    return ranked if higher_is_better else 1.0 - ranked


def explain_row(row: pd.Series) -> tuple[list[str], list[str]]:
    positive: list[tuple[float, str]] = []
    negative: list[tuple[float, str]] = []
    ret_3 = float(pd.to_numeric(row.get("ret_3"), errors="coerce") or 0.0)
    ret_5 = float(pd.to_numeric(row.get("ret_5"), errors="coerce") or 0.0)
    ret_10 = float(pd.to_numeric(row.get("ret_10"), errors="coerce") or 0.0)
    ret_20 = float(pd.to_numeric(row.get("ret_20"), errors="coerce") or 0.0)
    amount = float(pd.to_numeric(row.get("amount_ratio_5_20"), errors="coerce") or 0.0)
    fund = float(pd.to_numeric(row.get("fund_score"), errors="coerce") or 0.0)
    breadth = float(pd.to_numeric(row.get("breadth_score"), errors="coerce") or 0.0)
    volatility = float(pd.to_numeric(row.get("volatility_20"), errors="coerce") or 0.0)
    drawdown = float(pd.to_numeric(row.get("drawdown_60"), errors="coerce") or 0.0)
    if ret_3 > 0 and ret_5 > 0 and ret_10 > 0:
        positive.append((max(ret_3, ret_5, ret_10), "3/5/10日短期动量共振"))
    else:
        if ret_3 > 0:
            positive.append((ret_3 * 0.45, "3日相对动量转强"))
        if ret_5 > 0:
            positive.append((ret_5 * 0.35, "5日相对趋势领先"))
        if ret_10 > 0:
            positive.append((ret_10 * 0.20, "10日相对趋势领先"))
    if amount > 0.05:
        positive.append((abs(amount) * 0.1, "近期成交活跃度提升"))
    if fund > 0.2:
        positive.append((abs(fund) * 0.02, "主力资金信号偏强"))
    if breadth > 0.15:
        positive.append((abs(breadth) * 0.02, "板块内上涨覆盖较广"))
    if volatility > 0.035:
        negative.append((volatility, "短期波动偏高"))
    if drawdown < -0.10:
        negative.append((abs(drawdown), "距离阶段高点仍有明显回撤"))
    if ret_20 > 0.15:
        negative.append((ret_20 * 0.2, "阶段涨幅较大，存在拥挤回撤风险"))
    reasons = [label for _score, label in sorted(positive, reverse=True)[:3]]
    risks = [label for _score, label in sorted(negative, reverse=True)[:2]]
    if not reasons:
        reasons = ["多项趋势与活跃度信号综合占优"]
    if not risks:
        risks = ["预测可能受市场风格快速切换影响"]
    return reasons, risks


def rank_latest(panel: pd.DataFrame, validation: dict[str, Any], top_k: int = 10) -> list[dict[str, Any]]:
    if panel.empty:
        return []
    latest_day = pd.to_datetime(panel["date"]).max()
    latest = build_live_rows(panel, latest_day)
    if latest.empty:
        return []
    for column in ("fund_score", "breadth_score", "heat_score"):
        if column not in latest:
            latest[column] = 0.0
    if "current_data_completeness" not in latest:
        latest["current_data_completeness"] = 1.0
    if "history_days" not in latest:
        counts = panel.groupby("sector_id")["date"].nunique()
        latest["history_days"] = latest["sector_id"].map(counts).fillna(0)

    trend_score = _short_momentum_score(latest)
    activity_score = (
        _percentile(latest["amount_ratio_5_20"])
        + _percentile(latest["turnover_delta_5_20"])
    ) / 2.0
    quality_score = (
        _percentile(latest["volatility_20"], higher_is_better=False)
        + _percentile(latest["drawdown_60"])
    ) / 2.0
    fund_score = _percentile(latest["fund_score"])
    breadth_score = _percentile(latest["breadth_score"])
    heat_score = _percentile(latest["heat_score"])
    context_score = (
        0.35 * fund_score
        + 0.30 * breadth_score
        + 0.15 * heat_score
        + 0.10 * activity_score
        + 0.10 * quality_score
    )
    latest["factor_score"] = trend_score

    train = panel.dropna(subset=["target_excess_5"])
    model = fit_ridge(train)
    latest["model_prediction"] = model.predict(latest)
    latest["model_score"] = _percentile(latest["model_prediction"])
    validated = bool(validation.get("passed")) or bool(validation.get("head_passed"))
    selected_mode = str(
        validation.get("selected_mode")
        if bool(validation.get("passed"))
        else validation.get("head_selected_mode", "factor")
    )
    if not validated:
        selected_mode = "factor"
    if selected_mode == "model":
        latest["core_score"] = latest["model_score"]
    elif selected_mode == "hybrid":
        latest["core_score"] = 0.80 * latest["factor_score"] + 0.20 * latest["model_score"]
    else:
        latest["core_score"] = latest["factor_score"]
    context_weight = 0.15 if validated else 0.05
    latest["hybrid_score"] = (
        (1.0 - context_weight) * latest["core_score"]
        + context_weight * context_score
    )

    data_quality = pd.to_numeric(latest["current_data_completeness"], errors="coerce").fillna(0.0).clip(0, 1)
    validation_quality = float(np.clip(validation.get("quality", 0.0), 0.0, 1.0))
    agreement = (1.0 - (latest["factor_score"] - latest["model_score"]).abs()).clip(0, 1)
    history_quality = (pd.to_numeric(latest["history_days"], errors="coerce") / 500.0).clip(0, 1)
    confidence = 100.0 * (
        0.35 * data_quality
        + 0.30 * validation_quality
        + 0.20 * agreement
        + 0.15 * history_quality
    )
    confidence = confidence.where(data_quality >= 0.5, np.minimum(confidence, 44.0))
    latest["confidence"] = confidence.round().astype(int)
    latest = latest[latest["confidence"] >= 45].copy()
    latest = latest.sort_values(["hybrid_score", "confidence"], ascending=False).head(top_k)

    rows: list[dict[str, Any]] = []
    full_validated = bool(validation.get("passed"))
    head_validated = bool(validation.get("head_passed"))
    for rank, (_index, row) in enumerate(latest.iterrows(), start=1):
        reasons, risks = explain_row(row)
        percentile = float(row["hybrid_score"])
        if not full_validated and not (head_validated and rank <= 5):
            strength = "观察"
        else:
            strength = "强" if percentile >= 0.85 else "中" if percentile >= 0.70 else "弱"
        up_ratio = pd.to_numeric(row.get("up_ratio"), errors="coerce")
        rows.append(
            {
                "rank": rank,
                "sector_id": str(row["sector_id"]),
                "sector_name": str(row["sector_name"]),
                "strength": strength,
                "confidence": int(row["confidence"]),
                "trend": "3/5/10日短期强势" if float(row.get("ret_5", 0.0)) > 0 else "短期趋势仍待确认",
                "fund_state": "主力资金偏强" if float(row.get("fund_score", 0.0)) > 0.2 else "资金信号中性",
                "breadth": f"上涨家数占比{float(up_ratio):.0%}" if pd.notna(up_ratio) else "板块广度数据不足",
                "reasons": reasons,
                "risks": risks,
                "previous_realized_excess": None,
                "score": round(percentile * 100.0, 2),
                "core_mode": selected_mode,
                "context_weight": context_weight,
            }
        )
    return rows
