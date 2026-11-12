# -*- coding: utf-8 -*-
"""Pure point-in-time scoring primitives for the Zhengxi-inspired strategy."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import pandas as pd


def point_in_time_fundamentals(
    statements: pd.DataFrame,
    signal_date: str | date | pd.Timestamp,
) -> pd.DataFrame:
    """Return the latest public statement and the same period one year earlier.

    The current statement is the most recent report period public by
    ``signal_date`` (not the most recently revised one), and the comparison is
    year over year: a Q1 report compared with the prior annual report would
    mistake seasonality for improvement.
    """
    frame = statements.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["announcement_date"] = pd.to_datetime(
        frame["announcement_date"], errors="coerce"
    ).dt.normalize()
    frame["report_date"] = pd.to_datetime(
        frame["report_date"], errors="coerce"
    ).dt.normalize()
    cutoff = pd.Timestamp(signal_date).normalize()
    frame = frame[
        frame["announcement_date"].notna()
        & (frame["announcement_date"] <= cutoff)
    ].copy()
    if frame.empty:
        return frame

    # One row per report period: the latest public revision of it.
    frame = frame.sort_values(["code", "report_date", "announcement_date"])
    frame = frame.drop_duplicates(["code", "report_date"], keep="last")
    current = frame.groupby("code", sort=False, group_keys=False).tail(1).copy()
    value_columns = [
        column
        for column in frame.columns
        if column not in {"code", "announcement_date", "report_date"}
    ]
    previous = frame[["code", "report_date", *value_columns]].copy()
    previous["report_date"] = previous["report_date"] + pd.DateOffset(years=1)
    previous = previous.rename(
        columns={column: f"{column}_prev" for column in value_columns}
    )
    if not previous.empty:
        current = current.merge(previous, on=["code", "report_date"], how="left")
    else:
        for column in value_columns:
            current[f"{column}_prev"] = pd.NA
    current["fundamental_missing"] = 0.0
    current["fundamental_age_days"] = (
        cutoff - current["announcement_date"]
    ).dt.days.astype(float)
    return current.reset_index(drop=True)


NUMERIC_COLUMNS = (
    "new_price",
    "change_rate",
    "turnoverrate",
    "deal_amount",
    "average_amount_20",
    "history_days",
    "pe9",
    "pbnewmrq",
    "roe_weight",
    "revenue_yoy",
    "net_profit_yoy",
    "net_margin",
    "net_margin_prev",
    "cfo_to_profit",
    "cfo_to_profit_prev",
    "debt_ratio",
    "debt_ratio_prev",
    "receivables_to_assets",
    "receivables_to_assets_prev",
    "inventory_to_assets",
    "inventory_to_assets_prev",
    "mom_20",
    "mom_60",
    "above_ma20",
    "fund_amount_5",
)


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column not in result:
            result[column] = float("nan")
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def hard_filter_candidates(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply safety and tradability filters that fallback selection cannot relax."""
    result = _numeric_frame(frame)
    result["code"] = result.get("code", "").astype(str).str.zfill(6)
    result["name"] = result.get("name", "").fillna("").astype(str)
    result["industry"] = result.get("industry", "").fillna("").astype(str).str.strip()
    rejected = tuple(str(value).upper() for value in config.get("reject_name_keywords", []))
    name_is_safe = ~result["name"].str.upper().apply(
        lambda value: any(keyword in value for keyword in rejected)
    )
    valid_code = result["code"].str.fullmatch(r"\d{6}", na=False)
    eligible = (
        valid_code
        & name_is_safe
        & result["industry"].ne("")
        & result["new_price"].gt(0)
        & result["deal_amount"].ge(float(config.get("minimum_deal_amount", 100_000_000)))
        & result["average_amount_20"].ge(
            float(config.get("minimum_average_amount_20", 100_000_000))
        )
        & result["turnoverrate"].ge(float(config.get("minimum_turnover", 0.3)))
        & result["history_days"].ge(float(config.get("minimum_history_days", 40)))
    )
    return result.loc[eligible].copy().reset_index(drop=True)


def _rank(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.notna().any():
        return pd.Series(0.5, index=series.index, dtype=float)
    lower = numeric.quantile(0.05)
    upper = numeric.quantile(0.95)
    clipped = numeric.clip(lower=lower, upper=upper)
    ranked = clipped.rank(pct=True, method="average", ascending=higher_is_better)
    return ranked.fillna(0.5).astype(float)


def _within_industry_rank(
    frame: pd.DataFrame,
    column: str,
    *,
    higher_is_better: bool = True,
) -> pd.Series:
    return frame.groupby("industry", group_keys=False)[column].transform(
        lambda values: _rank(values, higher_is_better=higher_is_better)
    )


def build_industry_scores(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["flow_intensity"] = (
        work["fund_amount_5"] / (work["average_amount_20"].abs() * 5.0)
    ).clip(-1.0, 1.0)
    industry = (
        work.groupby("industry", as_index=False)
        .agg(
            mom_20=("mom_20", "median"),
            mom_60=("mom_60", "median"),
            breadth=("above_ma20", "mean"),
            revenue_yoy=("revenue_yoy", "median"),
            net_profit_yoy=("net_profit_yoy", "median"),
            flow_intensity=("flow_intensity", "median"),
        )
    )
    industry["mom_20_preference"] = -(industry["mom_20"] - 0.08).abs()
    industry["mom_60_preference"] = -(industry["mom_60"] - 0.20).abs()
    metric_ranks = pd.DataFrame(
        {
            column: _rank(industry[column])
            for column in (
                "mom_20_preference",
                "mom_60_preference",
                "breadth",
                "revenue_yoy",
                "net_profit_yoy",
                "flow_intensity",
            )
        }
    )
    industry["industry_prosperity_score"] = (
        metric_ranks.mean(axis=1) * 40.0
    ).clip(0.0, 40.0).round(2)
    return industry[["industry", "industry_prosperity_score"]]


def _improved(current: object, previous: object, margin: float = 0.0) -> bool:
    current_value = pd.to_numeric(pd.Series([current]), errors="coerce").iloc[0]
    previous_value = pd.to_numeric(pd.Series([previous]), errors="coerce").iloc[0]
    return bool(
        pd.notna(current_value)
        and pd.notna(previous_value)
        and float(current_value) > float(previous_value) + margin
    )


def _score_roe_repair(row: pd.Series) -> tuple[float, str, str, str]:
    signals = []
    if row["revenue_yoy"] > 0:
        signals.append("营收增长")
    if row["net_profit_yoy"] > 0:
        signals.append("利润增长")
    if _improved(row["net_margin"], row["net_margin_prev"], 0.005):
        signals.append("净利率改善")
    if (
        row["cfo_to_profit"] >= 0.8
        and _improved(row["cfo_to_profit"], row["cfo_to_profit_prev"])
    ):
        signals.append("现金流改善")
    if pd.notna(row["debt_ratio_prev"]) and row["debt_ratio"] <= row["debt_ratio_prev"]:
        signals.append("负债未恶化")
    if (
        pd.notna(row["receivables_to_assets_prev"])
        and row["receivables_to_assets"] <= row["receivables_to_assets_prev"]
    ):
        signals.append("应收受控")
    if (
        pd.notna(row["inventory_to_assets_prev"])
        and row["inventory_to_assets"] <= row["inventory_to_assets_prev"]
    ):
        signals.append("存货受控")

    structural_risk = bool(
        (row["revenue_yoy"] < 0 and row["net_profit_yoy"] < 0)
        or (
            row["cfo_to_profit"] < 0.5
            and pd.notna(row["cfo_to_profit_prev"])
            and row["cfo_to_profit"] < row["cfo_to_profit_prev"]
        )
        or (
            pd.notna(row["debt_ratio_prev"])
            and row["debt_ratio"] > row["debt_ratio_prev"] + 0.05
        )
    )
    improvement_count = len(signals)
    if structural_risk:
        score = 0.0
    elif improvement_count >= 2 and row.get("roe_ttm", row["roe_weight"]) > 0:
        score = min(
            25.0,
            3.0 + float(row["roe_repair_room"]) * 7.0 + improvement_count * 2.5,
        )
    else:
        score = min(5.0, improvement_count * 2.0)

    evidence = (
        "修复证据充分"
        if improvement_count >= 3 and not structural_risk
        else "修复证据有限"
        if improvement_count >= 2 and not structural_risk
        else "修复证据不足"
    )
    reason = "ROE修复代理:" + "、".join(signals[:4]) if signals else ""
    risk = "结构性低ROE风险" if structural_risk else ""
    return round(score, 2), evidence, reason, risk


def _score_company_quality(frame: pd.DataFrame) -> pd.Series:
    ranks = pd.DataFrame(index=frame.index)
    for column in ("revenue_yoy", "net_profit_yoy", "net_margin", "cfo_to_profit"):
        ranks[column] = _within_industry_rank(frame, column)
    ranks["debt_ratio"] = _within_industry_rank(
        frame, "debt_ratio", higher_is_better=False
    )
    return (ranks.mean(axis=1) * 15.0).clip(0.0, 15.0).round(2)


def _score_liquidity(frame: pd.DataFrame) -> pd.Series:
    turnover_preference = -((frame["turnoverrate"] - 4.0).abs())
    ranks = pd.DataFrame(
        {
            "deal": _rank(frame["deal_amount"]),
            "average": _rank(frame["average_amount_20"]),
            "turnover": _rank(turnover_preference),
        }
    )
    return (ranks.mean(axis=1) * 10.0).clip(0.0, 10.0).round(2)


def _score_valuation_timing(row: pd.Series) -> float:
    score = 0.0
    if 0 < row["pe9"] <= 80:
        score += 2.0
    if 0 < row["pbnewmrq"] <= 10:
        score += 1.0
    if row["mom_20"] > 0:
        score += 2.0
    if row["mom_60"] > 0:
        score += 2.0
    if row["above_ma20"] > 0:
        score += 1.0
    if row["fund_amount_5"] > 0:
        score += 1.0
    if -2.0 <= row["change_rate"] <= 5.0:
        score += 1.0
    if row["mom_20"] > 0.25:
        score -= min(5.0, 1.5 + (row["mom_20"] - 0.25) * 8.0)
    if row["mom_60"] > 0.60:
        score -= min(3.0, 1.0 + (row["mom_60"] - 0.60) * 3.0)
    if row["turnoverrate"] > 12.0 or row["change_rate"] > 7.5:
        score -= 2.0
    return max(0.0, min(10.0, score))


def _append_risk(existing: object, addition: str) -> str:
    return "；".join(
        part for part in (str(existing or "").strip("；"), addition) if part
    )


def score_candidates(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Score tradable candidates with industry prosperity as the largest block."""
    eligible = hard_filter_candidates(frame, config)
    if eligible.empty:
        return eligible
    industry_scores = build_industry_scores(eligible)
    eligible = eligible.merge(industry_scores, on="industry", how="left")
    # TTM ROE (PB/PE): companies on different report periods stay comparable.
    implied = 100.0 * eligible["pbnewmrq"] / eligible["pe9"].where(eligible["pe9"] != 0)
    eligible["roe_ttm"] = implied.where(eligible["pbnewmrq"] > 0).fillna(eligible["roe_weight"])
    roe_percentile = _within_industry_rank(eligible, "roe_ttm")
    eligible["roe_repair_room"] = (1.0 - roe_percentile).clip(0.0, 1.0)

    repair = eligible.apply(_score_roe_repair, axis=1, result_type="expand")
    repair.columns = ["roe_repair_score", "evidence_status", "repair_reason", "risk"]
    eligible = pd.concat([eligible, repair], axis=1)
    eligible["company_quality_score"] = _score_company_quality(eligible)
    eligible["liquidity_score"] = _score_liquidity(eligible)
    eligible["valuation_timing_score"] = eligible.apply(
        _score_valuation_timing, axis=1
    ).round(2)
    overheated = (
        eligible["mom_20"].gt(0.25)
        | eligible["mom_60"].gt(0.60)
        | eligible["turnoverrate"].gt(12.0)
        | eligible["change_rate"].gt(7.5)
    )
    eligible.loc[overheated, "risk"] = eligible.loc[overheated, "risk"].apply(
        lambda value: _append_risk(value, "动量过热")
    )
    eligible["total_score"] = (
        eligible["industry_prosperity_score"]
        + eligible["roe_repair_score"]
        + eligible["company_quality_score"]
        + eligible["liquidity_score"]
        + eligible["valuation_timing_score"]
    ).clip(0.0, 100.0).round(2)
    eligible["reason"] = eligible.apply(
        lambda row: "；".join(
            value
            for value in (
                "行业景气靠前" if row["industry_prosperity_score"] >= 30 else "",
                row["repair_reason"],
                "流动性达标",
            )
            if value
        ),
        axis=1,
    )
    return eligible.sort_values(
        ["total_score", "industry_prosperity_score", "liquidity_score", "code"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def select_fixed_count(
    scored: pd.DataFrame,
    *,
    count: int = 20,
    industry_cap: int = 4,
    strict_total: float = 70.0,
    strict_industry: float = 28.0,
    strict_repair: float = 15.0,
) -> pd.DataFrame:
    """Select exactly ``count`` rows without relaxing prior hard filters."""
    if len(scored) < count:
        raise ValueError(f"硬条件候选不足20只：仅{len(scored)}只")
    ranked = scored.sort_values(
        ["total_score", "industry_prosperity_score", "liquidity_score", "code"],
        ascending=[False, False, False, True],
    ).copy()
    strict = (
        ranked["total_score"].ge(strict_total)
        & ranked["industry_prosperity_score"].ge(strict_industry)
        & ranked["roe_repair_score"].ge(strict_repair)
        & ~ranked.get("risk", pd.Series("", index=ranked.index))
        .fillna("")
        .astype(str)
        .str.contains("动量过热|结构性低ROE风险", regex=True)
    )
    first_fallback = ~strict & ranked["industry_prosperity_score"].ge(strict_industry)
    ranked["selection_tier"] = "二级候补"
    ranked.loc[first_fallback, "selection_tier"] = "一级候补"
    ranked.loc[strict, "selection_tier"] = "严格入选"
    tier_order = {"严格入选": 0, "一级候补": 1, "二级候补": 2}
    ranked["_tier_order"] = ranked["selection_tier"].map(tier_order)
    ranked = ranked.sort_values(
        [
            "_tier_order",
            "total_score",
            "industry_prosperity_score",
            "liquidity_score",
            "code",
        ],
        ascending=[True, False, False, False, True],
    )

    selected_indices: list[int] = []
    industry_counts: dict[str, int] = {}
    for index, row in ranked.iterrows():
        industry = str(row["industry"])
        if industry_counts.get(industry, 0) >= industry_cap:
            continue
        selected_indices.append(index)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected_indices) == count:
            break

    selected_set = set(selected_indices)
    relaxed_indices: set[int] = set()
    if len(selected_indices) < count:
        for index in ranked.index:
            if index in selected_set:
                continue
            selected_indices.append(index)
            selected_set.add(index)
            relaxed_indices.add(index)
            if len(selected_indices) == count:
                break

    selected = ranked.loc[selected_indices].copy()
    selected["concentration_relaxed"] = selected.index.isin(relaxed_indices)
    if "risk" not in selected:
        selected["risk"] = ""
    selected["risk"] = selected["risk"].fillna("").astype(str)
    selected.loc[selected["concentration_relaxed"], "risk"] = selected.loc[
        selected["concentration_relaxed"], "risk"
    ].apply(
        lambda value: "；".join(
            part for part in (value.strip("；"), "行业集中度补位") if part
        )
    )
    selected = selected.drop(columns="_tier_order")
    selected["rank"] = range(1, len(selected) + 1)
    return selected.reset_index(drop=True)
