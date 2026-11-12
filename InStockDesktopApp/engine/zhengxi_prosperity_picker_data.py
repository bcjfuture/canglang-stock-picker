# -*- coding: utf-8 -*-
"""Desktop JSON engine for the Zhengxi-inspired prosperity strategy."""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from instock_tech_picker_data import (
    clean_value,
    connect,
    load_selection_snapshot,
    resolve_grouped_flows_wan,
    screener_lag_warning,
)
from daily_history import (
    FALLBACK_BAR_CACHE,
    align_to_calendar,
    append_spot_bar,
    extend_calendar,
    history_window_is_complete,
    latest_hist_dir,
    load_code_history,
    trading_calendar,
)
from picker_backtest import compute_factors
from picker_news import (
    AkshareNewsProvider,
    NewsFactorResult,
    evaluate_news_for_codes,
    shortlist_codes,
)
from zhengxi_prosperity_core import (
    point_in_time_fundamentals,
    score_candidates,
    select_fixed_count,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "zhengxi_prosperity_config.json"
FUNDAMENTALS_PATH = (
    ROOT.parent / "data" / "fundamentals" / "quarterly-point-in-time.pkl"
)
DAILY_BARS_PATH = FALLBACK_BAR_CACHE


def _append_explanation(existing: object, addition: str) -> str:
    parts = [part.strip() for part in str(existing or "").split("；") if part.strip()]
    if addition and addition not in parts:
        parts.append(addition)
    return "；".join(parts)


def apply_news_to_scored_frame(
    scored: pd.DataFrame,
    *,
    signal_date: str | date | pd.Timestamp,
    provider: object | None = None,
    cache_dir: Path | None = None,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply point-in-time news scores before fixed-count selection."""
    result_frame = scored.copy()
    cutoff = pd.Timestamp(signal_date).date()
    news_config = config.get("news_factor") or {}
    default = NewsFactorResult.not_evaluated()
    news_results: dict[str, NewsFactorResult] = {}
    if news_config.get("enabled", False) and not result_frame.empty:
        rows = result_frame[["code", "total_score"]].to_dict("records")
        codes = shortlist_codes(
            rows,
            code_key="code",
            score_key="total_score",
            top_n=int(config.get("top_n", 20)),
            min_shortlist=int(news_config.get("min_shortlist", 60)),
            positive_cap=float(news_config.get("positive_cap", 8.0)),
        )
        if cache_dir is None:
            data_dir = Path(os.environ.get("INSTOCK_DATA_DIR", ROOT.parent / "data"))
            cache_dir = data_dir / "cache"
        news_results = evaluate_news_for_codes(
            codes,
            signal_date=cutoff,
            provider=provider or AkshareNewsProvider(),
            cache_dir=Path(cache_dir),
            config=news_config,
        )

    for index, row in result_frame.iterrows():
        code = str(row.get("code") or "").zfill(6)
        result = news_results.get(code, default)
        result_frame.at[index, "total_score"] = round(
            float(_number(row.get("total_score"), 0.0) or 0.0) + result.score, 2
        )
        result_frame.at[index, "news_score"] = result.score
        result_frame.at[index, "news_summary"] = result.summary
        result_frame.at[index, "news_count"] = result.count
        result_frame.at[index, "news_status"] = result.status
        result_frame.at[index, "news_hard_risk"] = result.hard_risk
        result_frame.at[index, "news_as_of"] = cutoff.isoformat()
        if result.score > 0:
            result_frame.at[index, "reason"] = _append_explanation(
                row.get("reason"), f"新闻利好+{result.score:.1f}分：{result.summary}"
            )
        elif result.score < 0:
            label = "重大新闻风险" if result.hard_risk else "新闻风险"
            result_frame.at[index, "risk"] = _append_explanation(
                row.get("risk"), f"{label}{result.score:.1f}分：{result.summary}"
            )
        elif result.status == "资讯不可用":
            result_frame.at[index, "risk"] = _append_explanation(
                row.get("risk"), "新闻资讯不可用，本项按0分处理"
            )
    return result_frame.sort_values(
        ["total_score", "code"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def _number(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _money_wan(value: object) -> float:
    return round(float(_number(value, 0.0) or 0.0) / 10_000.0, 2)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else CONFIG_PATH
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_market_factor_frame(
    rows: list[dict],
    signal_date: str | date | pd.Timestamp,
    cache_dir: Path = DAILY_BARS_PATH,
    *,
    hist_dir: Path | str | None = "auto",
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Join current snapshot rows to signal-date-only daily factors."""
    cutoff = pd.Timestamp(signal_date).normalize()
    if hist_dir == "auto":
        # An explicit cache directory (tests, sandboxes) is used on its own.
        hist_dir = latest_hist_dir() if Path(cache_dir) == DAILY_BARS_PATH else None
    calendar = extend_calendar(trading_calendar() if calendar is None else calendar, [cutoff])
    records: list[dict[str, Any]] = []
    for source in rows:
        row = {key: clean_value(value) for key, value in source.items()}
        code = str(row.get("code") or "").zfill(6)
        if len(code) != 6 or not code.isdigit():
            continue
        try:
            bars, _meta = load_code_history(
                code, cutoff, calendar=calendar, hist_dir=hist_dir, fallback_dir=Path(cache_dir)
            )
            if bars.empty:
                continue
            bars = append_spot_bar(bars, row, cutoff, calendar)
            if len(bars) < 2:
                continue
            complete = history_window_is_complete(bars, calendar, cutoff)
            history_days = int(len(bars))
            bars = align_to_calendar(bars, calendar)
            factors = compute_factors(bars)
            latest = factors.iloc[-1]
            enriched = dict(row)
            # Momentum from a window with missing sessions measures the wrong horizon.
            enriched.update(
                {
                    "code": code,
                    "mom_20": _number(latest.get("mom_20")) if complete else None,
                    "mom_60": _number(latest.get("mom_60")) if complete else None,
                    "atr_14": _number(latest.get("atr_14")),
                    "above_ma20": float(
                        _number(bars.iloc[-1].get("close"), 0.0)
                        > _number(latest.get("ma_20"), float("inf"))
                    ) if complete else None,
                    "average_amount_20": _number(
                        pd.to_numeric(bars["amount"], errors="coerce").tail(20).mean()
                    ),
                    "history_days": history_days,
                    "history_complete": complete,
                }
            )
            records.append(enriched)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return pd.DataFrame(records)


def _prepare_scoring_frame(
    snapshot: dict,
    statements: pd.DataFrame,
    config: dict,
    cache_dir: Path,
) -> pd.DataFrame:
    signal_date = snapshot["selection_date"]
    market = load_market_factor_frame(snapshot["rows"], signal_date, cache_dir)
    if market.empty:
        raise ValueError("没有可用的点时日线因子")
    fundamentals = point_in_time_fundamentals(statements, signal_date)
    if fundamentals.empty:
        for column in (
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
        ):
            market[column] = float("nan")
        market["fundamental_missing"] = 1.0
        return market
    keep = [
        "code",
        "announcement_date",
        "report_date",
        "fundamental_missing",
        "fundamental_age_days",
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
    ]
    keep = [column for column in keep if column in fundamentals]
    result = market.merge(fundamentals[keep], on="code", how="left")
    result["fundamental_missing"] = result["fundamental_missing"].fillna(1.0)
    return result


def _output_row(row: pd.Series) -> dict[str, Any]:
    large_flow, retail_flow, order_flow_source = resolve_grouped_flows_wan(
        fund_amount=row.get("fund_amount"),
        fund_amount_super=row.get("fund_amount_super"),
        fund_amount_large=row.get("fund_amount_large"),
        fund_amount_medium=row.get("fund_amount_medium"),
        fund_amount_small=row.get("fund_amount_small"),
    )
    tier = str(row["selection_tier"])
    reason = "；".join(
        part for part in (tier, str(row.get("reason") or "")) if part
    )
    return {
        "排名": int(row["rank"]),
        "代码": str(row["code"]).zfill(6),
        "名称": str(row.get("name") or ""),
        "行业": str(row.get("industry") or ""),
        "概念": str(row.get("concept") or ""),
        "最新价": round(float(_number(row.get("new_price"), 0.0) or 0.0), 2),
        "涨跌幅%": round(float(_number(row.get("change_rate"), 0.0) or 0.0), 2),
        "换手率%": round(float(_number(row.get("turnoverrate"), 0.0) or 0.0), 2),
        "成交额_万元": _money_wan(row.get("deal_amount")),
        "综合评分": round(float(row["total_score"]), 2),
        "行业景气分": round(float(row["industry_prosperity_score"]), 2),
        "ROE修复代理分": round(float(row["roe_repair_score"]), 2),
        "公司质量分": round(float(row["company_quality_score"]), 2),
        "流动性分": round(float(row["liquidity_score"]), 2),
        "估值时机分": round(float(row["valuation_timing_score"]), 2),
        "新闻评分": round(float(_number(row.get("news_score"), 0.0) or 0.0), 1),
        "新闻摘要": str(row.get("news_summary") or ""),
        "新闻数量": int(_number(row.get("news_count"), 0.0) or 0),
        "新闻状态": str(row.get("news_status") or "未评估"),
        "新闻重大风险": bool(row.get("news_hard_risk", False)),
        "新闻截止日": str(row.get("news_as_of") or ""),
        "入选层级": tier,
        "证据状态": str(row.get("evidence_status") or "财务证据不足"),
        "今日主力净流入_万元": _money_wan(row.get("fund_amount")),
        "3日主力净流入_万元": _money_wan(row.get("fund_amount_3")),
        "5日主力净流入_万元": _money_wan(row.get("fund_amount_5")),
        "今日大单净额_万元": large_flow,
        "今日散户净额_万元": retail_flow,
        "订单资金日期": str(clean_value(row.get("date")) or ""),
        "订单资金来源": order_flow_source,
        "PE": _number(row.get("pe9")),
        "PB": _number(row.get("pbnewmrq")),
        "ROE%": (
            round(value, 2)
            if (value := _number(row.get("roe_ttm", row.get("roe_weight")))) is not None
            else None
        ),
        "净利同比%": _number(row.get("net_profit_yoy")),
        "ATR14": _number(row.get("atr_14")),
        "加分理由": reason,
        "风险提示": str(row.get("risk") or ""),
    }


def build_payload(
    snapshot: dict,
    statements: pd.DataFrame,
    config: dict,
    cache_dir: Path = DAILY_BARS_PATH,
    *,
    news_provider: object | None = None,
    news_cache_dir: Path | None = None,
) -> dict[str, Any]:
    scoring_frame = _prepare_scoring_frame(snapshot, statements, config, cache_dir)
    scored = score_candidates(scoring_frame, config)
    scored = apply_news_to_scored_frame(
        scored,
        signal_date=snapshot["selection_date"],
        provider=news_provider,
        cache_dir=news_cache_dir,
        config=config,
    )
    selected = select_fixed_count(
        scored,
        count=int(config.get("top_n", 20)),
        industry_cap=int(config.get("industry_cap", 4)),
        strict_total=float(config.get("strict_total_score", 70.0)),
        strict_industry=float(config.get("strict_industry_score", 28.0)),
        strict_repair=float(config.get("strict_repair_score", 15.0)),
    )
    output_by_code = {
        str(row["代码"]): row
        for row in (_output_row(row) for _, row in scored.assign(rank=range(1, len(scored) + 1), selection_tier="未入选").iterrows())
    }
    top_rows = [_output_row(row) for _, row in selected.iterrows()]
    all_rows = list(output_by_code.values())[:300]
    industry_rows = sorted(
        all_rows,
        key=lambda item: (item["行业景气分"], item["综合评分"]),
        reverse=True,
    )[:100]
    repair_rows = sorted(
        all_rows,
        key=lambda item: (item["ROE修复代理分"], item["综合评分"]),
        reverse=True,
    )[:100]
    risk_rows = [
        row
        for row in top_rows
        if row["入选层级"] != "严格入选" or row["风险提示"]
    ]
    tier_counts = pd.Series(selected["selection_tier"]).value_counts().to_dict()
    industry_counts = pd.Series(selected["industry"]).value_counts()
    top_sheet = config.get("top_sheet_name", "郑希景气Top")
    industry_sheet = config.get("industry_sheet_name", "行业景气Top")
    repair_sheet = config.get("repair_sheet_name", "ROE修复Top")
    risk_sheet = config.get("risk_sheet_name", "候补与风险")
    all_sheet = config.get("all_sheet_name", "全部郑希候选")
    signal_date = clean_value(snapshot["selection_date"])
    return {
        "order_flow_schema": "grouped-order-flow-v2",
        "title": config["title"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_dates": {
            "综合选股": clean_value(snapshot.get("screener_date", snapshot["selection_date"])),
            "选股计算日": signal_date,
            "每日行情": clean_value(snapshot.get("spot_date")),
            "资金流向": clean_value(snapshot.get("fund_date")),
        },
        "summary": {
            "全市场记录数": len(snapshot["rows"]),
            "硬条件候选数": len(scored),
            "输出Top数": len(top_rows),
            "严格入选数": int(tier_counts.get("严格入选", 0)),
            "一级候补数": int(tier_counts.get("一级候补", 0)),
            "二级候补数": int(tier_counts.get("二级候补", 0)),
            "行业数": int(industry_counts.size),
            "行业最大占比": round(float(industry_counts.max() / len(selected)), 4),
            "active_count": 20,
            "abstain_reason": "",
            "data_warning": screener_lag_warning(snapshot),
            "market_regime": "always_invested",
        },
        "strategy_gate_passed": False,
        "trade_policy": None,
        "rules_note": config.get("rules_note", []),
        "top_sheet_name": top_sheet,
        "preview_sheet": top_sheet,
        "sheet_meta": {
            top_sheet: {
                "title": config.get("top_sheet_title", top_sheet),
                "subtitle": config.get("top_sheet_subtitle", "按行业景气和修复证据排序"),
            },
            industry_sheet: {"title": industry_sheet, "subtitle": "按行业景气分排序"},
            repair_sheet: {"title": repair_sheet, "subtitle": "按ROE修复代理分排序"},
            risk_sheet: {"title": risk_sheet, "subtitle": "候补与风险提示"},
            all_sheet: {"title": all_sheet, "subtitle": "全部硬条件候选"},
        },
        "sheets": {
            top_sheet: top_rows,
            industry_sheet: industry_rows,
            repair_sheet: repair_rows,
            risk_sheet: risk_rows,
            all_sheet: all_rows,
        },
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: zhengxi_prosperity_picker_data.py output.json [config.json] [as_of_date]",
            file=sys.stderr,
        )
        return 2
    output_path = Path(sys.argv[1])
    config = load_config(sys.argv[2] if len(sys.argv) >= 3 else None)
    as_of_date = pd.Timestamp(sys.argv[3]).date() if len(sys.argv) >= 4 else None
    with connect() as connection:
        snapshot = load_selection_snapshot(connection.cursor(), as_of_date)
    statements = pd.read_pickle(FUNDAMENTALS_PATH)
    payload = build_payload(snapshot, statements, config, DAILY_BARS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_path}")
    print(f"Selected {len(payload['sheets'][config['top_sheet_name']])} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
