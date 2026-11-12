# -*- coding: utf-8 -*-
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import pymysql
import pandas as pd

from picker_backtest import compute_factors
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
from market_policy import MarketPolicy, classify_regime, pick_count_for_regime
from selection_refinement import (
    attach_peer_momentum,
    historical_features,
    operating_cash_flow_positive,
    attach_evidence_scores,
    score_groups,
    score_groups_v2,
    ttm_roe_pct,
)
from picker_news import (
    AkshareNewsProvider,
    NewsFactorResult,
    evaluate_news_for_codes,
    shortlist_codes,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "tech_attack_config.json"

DB = {
    "host": "127.0.0.1",  # "localhost" tries IPv6 first: ~2 s per connection
    "user": "root",
    "password": "root",
    "database": "instockdb",
    "port": 3306,
    "charset": "utf8mb4",
}

STRATEGY_LABELS = {
    "cn_stock_strategy_enter": "放量上涨",
    "cn_stock_strategy_keep_increasing": "均线多头",
    "cn_stock_strategy_parking_apron": "停机坪",
    "cn_stock_strategy_backtrace_ma250": "回踩年线",
    "cn_stock_strategy_breakthrough_platform": "突破平台",
    "cn_stock_strategy_low_backtrace_increase": "无大幅回撤",
    "cn_stock_strategy_turtle_trade": "海龟交易法则",
    "cn_stock_strategy_high_tight_flag": "高而窄旗形",
    "cn_stock_strategy_climax_limitdown": "放量跌停",
    "cn_stock_strategy_low_atr": "低ATR成长",
}


def clean_value(value):
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bytes):
        return int.from_bytes(value, "big")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def bit(value):
    if value is None:
        return False
    if isinstance(value, bytes):
        return any(value)
    return bool(value)


def f(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def calculate_market_earnings_v3(
    *, pe, pb, historical_roe_pct, dividend_yield=None
):
    """Return Ding Ning's dividend-adjusted market-earnings metrics.

    ``dividend_yield`` and the returned ``payout_ratio`` use decimal units.
    When dividend evidence is unavailable, the correction is conservatively
    capped at 2 rather than treating missing data as a generous payout.
    """
    pe_value = f(pe, None)
    pb_value = f(pb, None)
    historical_roe_value = f(historical_roe_pct, None)
    if (
        pe_value is None
        or pb_value is None
        or historical_roe_value is None
        or not math.isfinite(pe_value)
        or not math.isfinite(pb_value)
        or not math.isfinite(historical_roe_value)
        or pe_value <= 0
        or pb_value <= 0
        or historical_roe_value <= 0
    ):
        return None

    implied_roe_pct = 100.0 * pb_value / pe_value
    basic_ratio = pb_value * 100.0 / (historical_roe_value * historical_roe_value)
    yield_value = f(dividend_yield, None)
    if yield_value is not None and math.isfinite(yield_value) and yield_value > 0:
        payout_ratio = yield_value * pe_value
        dividend_correction = min(2.0, max(1.0, 0.5 / payout_ratio))
    else:
        payout_ratio = None
        dividend_correction = 2.0
    return {
        "historical_roe_pct": historical_roe_value,
        "implied_roe_pct": implied_roe_pct,
        "payout_ratio": payout_ratio,
        "dividend_correction": dividend_correction,
        "market_earnings_ratio": basic_ratio * dividend_correction,
    }


def _rank_market_earnings_candidates(rows, rules, *, tier):
    ranked = []
    for source in rows:
        if bool(source.get("新闻重大风险")):
            continue
        pe = f(source.get("PE"), None)
        pb = f(source.get("PB"), None)
        if pe is None or pb is None or pe <= 0 or not 0 < pb < 1:
            continue
        dividend_yield_pct = f(source.get("股息率%"), None)
        metrics = calculate_market_earnings_v3(
            pe=pe,
            pb=pb,
            historical_roe_pct=source.get("历史最低ROE%"),
            dividend_yield=(
                dividend_yield_pct / 100.0
                if dividend_yield_pct is not None
                else None
            ),
        )
        if metrics is None:
            continue
        max_ratio = f(rules.get("max_market_earnings_ratio"), 1.0)
        if metrics["market_earnings_ratio"] >= max_ratio:
            continue
        industry = str(source.get("行业") or "")
        financial_industries = {
            str(value) for value in rules.get("financial_industries", [])
        }
        row = dict(source)
        quality_penalty = 0
        profit_growth = f(source.get("净利同比%"), None)
        quality_penalty += 1 if profit_growth is None else 2 if profit_growth < 0 else 0
        three_year_growth = f(source.get("三年净利增速%"), None)
        quality_penalty += (
            1 if three_year_growth is None else 2 if three_year_growth < 0 else 0
        )
        if industry not in financial_industries:
            debt = f(source.get("资产负债率%"), None)
            quality_penalty += 1 if debt is None or debt >= 70 else 0
            operating_cash_flow = f(source.get("每股经营现金流"), None)
            # Prefer the trailing sign; the per-share figure is year-to-date.
            cash_positive = source.get("经营现金流为正")
            if cash_positive is None and operating_cash_flow is not None and operating_cash_flow != 0:
                cash_positive = operating_cash_flow > 0
            quality_penalty += 0 if cash_positive else 1
        row.update(
            {
                "入选层级": tier,
                "基本面风险分": quality_penalty,
                "历史最低ROE%": round(metrics["historical_roe_pct"], 2),
                "隐含ROE%": round(metrics["implied_roe_pct"], 2),
                "估算分红率%": (
                    round(metrics["payout_ratio"] * 100.0, 2)
                    if metrics["payout_ratio"] is not None
                    else None
                ),
                "分红修正系数": round(metrics["dividend_correction"], 4),
                "市赚率3.0": round(metrics["market_earnings_ratio"], 4),
            }
        )
        row["加分理由"] = _append_explanation(
            f"质量风险分{quality_penalty}（越低越优先）；市赚率3.0={metrics['market_earnings_ratio']:.3f}；通过估值硬门槛",
            str(row.get("加分理由") or ""),
        )
        if metrics["payout_ratio"] is None:
            row["风险提示"] = _append_explanation(
                row.get("风险提示"),
                "分红数据缺失，修正系数按保守上限2计算",
            )
        if profit_growth is not None and profit_growth < 0:
            row["风险提示"] = _append_explanation(
                row.get("风险提示"),
                "净利润同比下降",
            )
        if three_year_growth is not None and three_year_growth < 0:
            row["风险提示"] = _append_explanation(
                row.get("风险提示"),
                "三年净利润增速为负",
            )
        if profit_growth is not None and profit_growth < 0 and three_year_growth is not None and three_year_growth < 0:
            row["风险提示"] = _append_explanation(row.get("风险提示"), "持续盈利走弱，关注价值陷阱")
        row["质量层级"] = "质量风险较低" if quality_penalty == 0 else "质量待核查"
        row["加分理由"] = _append_explanation(row.get("加分理由"), row["质量层级"])
        if industry not in financial_industries:
            if debt is not None and debt >= 70:
                row["风险提示"] = _append_explanation(
                    row.get("风险提示"),
                    "资产负债率偏高",
                )
            if cash_positive is False:
                row["风险提示"] = _append_explanation(
                    row.get("风险提示"),
                    "经营现金流为负",
                )
        ranked.append(row)
    ranked.sort(
        key=lambda row: (
            f(row.get("市赚率3.0"), float("inf")),
            -f(row.get("综合评分")),
            str(row.get("代码") or ""),
        )
    )
    return ranked


def select_market_earnings_v3(rows, cfg):
    """Build the PB-discounted all-market shortlist and rank it by PR 3.0."""
    rules = cfg.get("market_earnings_v3") or {}
    top_n = int(cfg.get("top_n", 20))
    eligible = _rank_market_earnings_candidates(rows, rules, tier="市赚率<1")
    # Quality is a coarse gate (low risk vs. needs review); within a tier the
    # valuation metric decides, instead of only breaking ties on a 0-6 count.
    low_risk_max = f(rules.get("low_risk_max_penalty"), 1)
    ranked = sorted(
        eligible,
        key=lambda row: (
            0 if f(row.get("基本面风险分"), float("inf")) <= low_risk_max else 1,
            f(row.get("市赚率3.0"), float("inf")),
            f(row.get("基本面风险分"), float("inf")),
            -f(row.get("综合评分")),
            str(row.get("代码") or ""),
        ),
    )
    industry_cap = int(rules.get("industry_cap", 0) or 0)
    if industry_cap <= 0:
        return ranked[:top_n]
    selected, counts = [], defaultdict(int)
    for row in ranked:
        industry = str(row.get("行业") or "未知行业")
        if counts[industry] >= industry_cap:
            continue
        counts[industry] += 1
        selected.append(row)
        if len(selected) == top_n:
            break
    return selected


def minimum_disclosed_annual_roe(records, *, as_of_date, years=3):
    """Return the minimum ROE from the latest disclosed full-year reports."""
    cutoff = pd.Timestamp(as_of_date).date()
    by_report_date = {}
    for source in records:
        report_date = pd.to_datetime(source.get("REPORT_DATE"), errors="coerce")
        notice_date = pd.to_datetime(source.get("NOTICE_DATE"), errors="coerce")
        roe = f(source.get("ROEJQ"), None)
        if (
            pd.isna(report_date)
            or pd.isna(notice_date)
            or roe is None
            or report_date.month != 12
            or report_date.day != 31
            or notice_date.date() > cutoff
        ):
            continue
        key = report_date.date()
        existing = by_report_date.get(key)
        if existing is None or notice_date > existing[0]:
            by_report_date[key] = (notice_date, roe)
    recent = [by_report_date[key][1] for key in sorted(by_report_date, reverse=True)[:years]]
    if len(recent) < years:
        return {"minimum_roe_pct": None, "annual_count": len(recent)}
    return {"minimum_roe_pct": min(recent), "annual_count": len(recent)}


def _market_symbol(code):
    code = str(code)
    if code.startswith(("4", "8", "9")):
        return f"{code}.BJ"
    if code.startswith(("5", "6")):
        return f"{code}.SH"
    return f"{code}.SZ"


def _fetch_annual_roe_records(code):
    import akshare as ak

    frame = ak.stock_financial_analysis_indicator_em(symbol=_market_symbol(code))
    if frame is None or frame.empty:
        return []
    required = {"REPORT_DATE", "NOTICE_DATE", "ROEJQ"}
    if not required.issubset(frame.columns):
        return []
    return frame[list(required)].to_dict("records")


def _serializable_roe_records(records):
    normalized = []
    for source in records or []:
        report_date = pd.to_datetime(source.get("REPORT_DATE"), errors="coerce")
        notice_date = pd.to_datetime(source.get("NOTICE_DATE"), errors="coerce")
        roe = f(source.get("ROEJQ"), None)
        if pd.isna(report_date) or pd.isna(notice_date) or roe is None:
            continue
        normalized.append(
            {
                "REPORT_DATE": report_date.date().isoformat(),
                "NOTICE_DATE": notice_date.date().isoformat(),
                "ROEJQ": roe,
            }
        )
    return normalized


def load_market_earnings_roe_minimums(
    rows,
    *,
    as_of_date,
    years=3,
    fetcher=None,
    cache_path=None,
    max_workers=8,
):
    """Load and cache annual ROE histories, returning a minimum per candidate."""
    cache_file = Path(cache_path or ROOT.parent / "data" / "market_earnings_roe_history.json")
    cache = {"version": 1, "fetched_at": None, "records": {}}
    try:
        loaded = json.loads(cache_file.read_text(encoding="utf-8"))
        if loaded.get("version") == 1 and isinstance(loaded.get("records"), dict):
            cache = loaded
    except (FileNotFoundError, OSError, ValueError, TypeError):
        pass

    needed_codes = sorted(
        {
            str(row.get("代码") or "")
            for row in rows
            if f(row.get("PE"), None) is not None
            and f(row.get("PE"), None) > 0
            and f(row.get("PB"), None) is not None
            and 0 < f(row.get("PB"), None) < 1
            and str(row.get("代码") or "")
        }
    )
    records_by_code = dict(cache.get("records") or {})
    missing_codes = [code for code in needed_codes if code not in records_by_code]
    fetch_records = fetcher or _fetch_annual_roe_records
    if missing_codes:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            jobs = {executor.submit(fetch_records, code): code for code in missing_codes}
            for index, future in enumerate(as_completed(jobs), start=1):
                code = jobs[future]
                try:
                    records_by_code[code] = _serializable_roe_records(future.result())
                except Exception as exc:
                    print(f"历史ROE获取失败 {code}: {type(exc).__name__}", file=sys.stderr)
                    records_by_code.setdefault(code, [])
                if index % 50 == 0:
                    print(f"历史ROE已获取 {index}/{len(missing_codes)}")
        cache = {
            "version": 1,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "records": records_by_code,
        }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        code: minimum_disclosed_annual_roe(
            records_by_code.get(code, []),
            as_of_date=as_of_date,
            years=years,
        )
        for code in needed_codes
    }


def attach_market_earnings_roe_minimums(rows, minimums):
    """Attach formula-ready historical ROE evidence to scored candidates."""
    for row in rows:
        evidence = minimums.get(str(row.get("代码") or ""), {})
        row["历史最低ROE%"] = evidence.get("minimum_roe_pct")
        row["历史ROE年度数"] = int(evidence.get("annual_count") or 0)
    return rows


def select_output_rows(scored, cfg, *, active_count, regime):
    if cfg.get("selection_mode") == "market_earnings_v3":
        selected = select_market_earnings_v3(scored, cfg)
        for index, row in enumerate(selected, start=1):
            row["排名"] = index
        return selected

    if str(cfg.get("scoring_version", "")).startswith("grouped"):
        safe = [row for row in scored if not row.get("新闻重大风险")]
        qualified = [row for row in safe if row["综合评分"] >= cfg["min_score"] and row.get("形态达标", False)]
        qualified_codes = {row["代码"] for row in qualified}
        # Filling empty slots with unconfirmed names dilutes the signal; strategies
        # can opt out and output only what passes the strict rule.
        fallback = (
            [row for row in safe if row["代码"] not in qualified_codes]
            if cfg.get("fallback_fill", True)
            else []
        )
        selected = [dict(row) for row in (qualified + fallback)[:active_count]]
        for index, row in enumerate(selected, 1):
            strict = row["代码"] in qualified_codes
            row["排名"] = index
            row["入选层级"] = "严格入选" if strict else "候补入选"
            row["加分理由"] = _append_explanation(row["入选层级"], str(row.get("加分理由") or ""))
            if not strict:
                row["风险提示"] = _append_explanation(row.get("风险提示"), "候补补位：分数或形态未达到严格标准")
        return selected

    qualified = [row for row in scored if row["综合评分"] >= cfg["min_score"]]
    selected = qualified[:active_count]
    if regime == "legacy" and len(selected) < min(30, cfg["top_n"]):
        selected = scored[: cfg["top_n"]]
    return selected


def select_fundamental_rows(scored, cfg):
    if cfg.get("selection_mode") == "market_earnings_v3":
        return select_market_earnings_v3(scored, {**cfg, "top_n": 100})
    return sorted(
        [
            row
            for row in scored
            if (row.get("PE") or 0) > 0
            and (row.get("PE") or 999) <= 80
            and (row.get("ROE%") or 0) >= 8
        ],
        key=lambda row: (row.get("ROE%") or 0, row["综合评分"]),
        reverse=True,
    )[:100]


def money_wan(value):
    return round(f(value) / 10000, 2)


def _append_explanation(existing, addition):
    parts = [part.strip() for part in str(existing or "").split("；") if part.strip()]
    if addition and addition not in parts:
        parts.append(addition)
    return "；".join(parts)


def apply_news_factor(row, result, signal_date):
    """Apply an already point-in-time-filtered news result to one scored row."""
    row["综合评分"] = round(f(row.get("综合评分")) + result.score, 1)
    row["新闻评分"] = round(result.score, 1)
    row["新闻摘要"] = result.summary
    row["新闻数量"] = result.count
    row["新闻状态"] = result.status
    row["新闻重大风险"] = result.hard_risk
    row["新闻截止日"] = signal_date.isoformat()
    if result.score > 0:
        row["加分理由"] = _append_explanation(
            row.get("加分理由"), f"新闻利好+{result.score:.1f}分：{result.summary}"
        )
    elif result.score < 0:
        label = "重大新闻风险" if result.hard_risk else "新闻风险"
        row["风险提示"] = _append_explanation(
            row.get("风险提示"), f"{label}{result.score:.1f}分：{result.summary}"
        )
    elif result.status == "资讯不可用":
        row["风险提示"] = _append_explanation(
            row.get("风险提示"), "新闻资讯不可用，本项按0分处理"
        )
    return row


def _news_signal_date(value):
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"无效新闻截止日：{value}")
    return parsed.date()


def apply_news_to_scored_rows(scored, cfg, signal_date, provider=None, cache_dir=None):
    news_cfg = cfg.get("news_factor") or {}
    signal_date = _news_signal_date(signal_date)
    if not news_cfg.get("enabled", False):
        for row in scored:
            apply_news_factor(row, NewsFactorResult.not_evaluated(), signal_date)
        return scored
    codes = shortlist_codes(
        scored,
        code_key="代码",
        score_key="综合评分",
        top_n=int(cfg.get("top_n", 20)),
        min_shortlist=int(news_cfg.get("min_shortlist", 60)),
        positive_cap=float(news_cfg.get("positive_cap", 8.0)),
    )
    if cache_dir is None:
        data_dir = Path(os.environ.get("INSTOCK_DATA_DIR", ROOT.parent / "data"))
        cache_dir = data_dir / "cache"
    results = evaluate_news_for_codes(
        codes,
        signal_date=signal_date,
        provider=provider or AkshareNewsProvider(),
        cache_dir=Path(cache_dir),
        config=news_cfg,
    )
    for row in scored:
        result = results.get(str(row.get("代码") or ""), NewsFactorResult.not_evaluated())
        apply_news_factor(row, result, signal_date)
    return scored


def resolve_grouped_flows_wan(
    *,
    fund_amount: object,
    fund_amount_super: object,
    fund_amount_large: object,
    fund_amount_medium: object,
    fund_amount_small: object,
) -> tuple[float | None, float | None, str]:

    def grouped(first, second):
        if any(value is None or str(value).strip() == "" for value in (first, second)):
            return None
        return money_wan(f(first) + f(second))

    large = grouped(fund_amount_super, fund_amount_large)
    retail = grouped(fund_amount_medium, fund_amount_small)
    if large is not None or retail is not None:
        return large, retail, "数据库分项"
    if fund_amount is not None and str(fund_amount).strip() != "":
        main = money_wan(f(fund_amount))
        return main, round(-main, 2), "数据库汇总推导"
    return None, None, "分项暂缺"


def score_fund_flows(
    fund_amount: object,
    fund_amount_3: object,
    fund_amount_5: object,
) -> tuple[float, list[str], list[str]]:
    """Score 1/3/5-day main flow once, with symmetric outflow penalties."""
    values = (
        ("1日", f(fund_amount), 12.0, 5.0, 100_000_000.0),
        ("3日", f(fund_amount_3), 7.0, 4.0, 300_000_000.0),
        ("5日", f(fund_amount_5), 5.0, 3.0, 500_000_000.0),
    )
    score = 0.0
    reasons: list[str] = []
    risks: list[str] = []
    for label, value, cap, base, scale in values:
        if value > 0:
            score += min(cap, base + value / scale)
            reasons.append(f"{label}资金净流入")
        elif value < 0:
            score -= min(cap, base + abs(value) / scale)
            risks.append(f"{label}资金净流出")
    signs = [value for _label, value, _cap, _base, _scale in values]
    if all(value > 0 for value in signs):
        score += 7.0
        reasons.append("资金连续净流入")
    elif all(value < 0 for value in signs):
        score -= 7.0
        risks.append("资金连续净流出")
    return score, reasons, risks


def score_shareholder_structure(
    *,
    institutional_ratio: object,
    institutional_count: object,
    holder_count_change: object,
    top10_ratio: object,
    survey_count: object,
) -> tuple[float, list[str], list[str], str]:
    """Score disclosed shareholder structure without assuming more is always better."""
    def optional(value):
        if value is None or str(value).strip() == "":
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    ratio = optional(institutional_ratio)
    count = optional(institutional_count)
    holder_change = optional(holder_count_change)
    concentration = optional(top10_ratio)
    surveys = optional(survey_count)
    values = (ratio, count, holder_change, concentration, surveys)
    if all(value is None for value in values):
        return 0.0, [], [], "股东结构数据不足"

    score = 0.0
    reasons: list[str] = []
    risks: list[str] = []
    summary_parts: list[str] = []

    if ratio is not None:
        summary_parts.append(
            f"机构持股{ratio:.2f}%"
            + (f"（{int(count)}家）" if count is not None else "")
        )
        if 5 <= ratio <= 50:
            score += 4
            reasons.append("机构持股适中")
        elif 1 <= ratio <= 75:
            score += 2
            reasons.append("机构参与持股")
        elif ratio > 85:
            score -= 2
            risks.append("机构持股较拥挤")
    elif count is not None:
        summary_parts.append(f"机构持股{int(count)}家")

    if count is not None:
        if count >= 30:
            score += 3
            reasons.append("机构覆盖较广")
        elif count >= 10:
            score += 2
            reasons.append("机构覆盖")
        elif count >= 3:
            score += 1

    if holder_change is not None:
        summary_parts.append(f"股东户数变化{holder_change:+.2f}%")
        if holder_change <= -5:
            score += 5
            reasons.append("股东户数下降")
        elif holder_change <= 0:
            score += 2
            reasons.append("筹码趋于集中")
        elif holder_change > 10:
            score -= 6
            risks.append("股东户数明显增加")
        elif holder_change > 5:
            score -= 3
            risks.append("股东户数增加")

    if concentration is not None:
        summary_parts.append(f"前十大股东{concentration:.2f}%")
        if 25 <= concentration <= 70:
            score += 2
            reasons.append("股权集中度适中")
        elif concentration > 85:
            score -= 5
            risks.append("股权集中度过高")
        elif concentration < 15:
            score -= 2
            risks.append("股权结构较分散")

    if surveys is not None:
        summary_parts.append(f"近3月机构调研{int(surveys)}次")
        if surveys >= 20:
            score += 3
            reasons.append("机构调研活跃")
        elif surveys >= 5:
            score += 2
            reasons.append("机构关注度较高")
        elif surveys >= 1:
            score += 1

    score = max(-15.0, min(15.0, score))
    return score, reasons, risks, "；".join(summary_parts) or "股东结构数据不足"


def text_join(*parts):
    return " ".join(str(p or "") for p in parts)


def load_config(config_path=None):
    path = Path(config_path) if config_path else CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def connect():
    return pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **DB)


def table_exists(cur, table_name):
    cur.execute(
        """
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_schema=%s AND table_name=%s
        """,
        (DB["database"], table_name),
    )
    return cur.fetchone()["n"] > 0


def latest_date(cur, table_name, as_of_date=None):
    if not table_exists(cur, table_name):
        return None
    if as_of_date is None:
        cur.execute(f"SELECT MAX(`date`) AS d FROM `{table_name}`")
    else:
        cur.execute(
            f"SELECT MAX(`date`) AS d FROM `{table_name}` WHERE `date` <= %s",
            (as_of_date,),
        )
    row = cur.fetchone()
    return row["d"] if row else None


def fetch_code_set(cur, table_name, as_of_date=None):
    d = latest_date(cur, table_name, as_of_date)
    if not d:
        return set()
    cur.execute(f"SELECT code FROM `{table_name}` WHERE `date`=%s", (d,))
    return {str(row["code"]) for row in cur.fetchall()}


def fetch_strategy_hits(cur, as_of_date=None):
    hits = defaultdict(list)
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema=%s AND table_name LIKE 'cn_stock_strategy%%'
        """,
        (DB["database"],),
    )
    tables = [r["table_name"] for r in cur.fetchall()]
    for table in tables:
        d = latest_date(cur, table, as_of_date)
        if not d:
            continue
        label = STRATEGY_LABELS.get(table, table.replace("cn_stock_strategy_", "策略_"))
        cur.execute(f"SELECT code FROM `{table}` WHERE `date`=%s", (d,))
        for row in cur.fetchall():
            hits[str(row["code"])].append(label)
    return hits


def tech_matches(row, cfg):
    haystack = text_join(row.get("industry"), row.get("concept"), row.get("style")).upper()
    matched = []
    keywords = cfg.get("focus_keywords", cfg.get("core_tech_keywords", []))
    industries = cfg.get("focus_industries", cfg.get("tech_industries", []))
    for kw in keywords:
        if kw.upper() in haystack:
            matched.append(kw)
    industry = str(row.get("industry") or "")
    if industry in industries and industry not in matched:
        matched.insert(0, industry)
    return matched


def reject_name(row, cfg):
    name = str(row.get("name") or "")
    return any(k in name for k in cfg["reject_name_keywords"])


def _safe_call(function, *args):
    try:
        return function(*args)
    except Exception:
        return None


def load_historical_context(rows, weights, cache_dir=FALLBACK_BAR_CACHE, *, selection_features=False, hist_dir="auto", calendar=None):
    """Load signal-date-only factors, ATR and cross-sectional market context."""
    factor_rows = []
    selection_context = {}
    if not weights:
        for row in rows:
            code = str(row.get("code") or "").zfill(6)
            close = f(row.get("new_price"), 0.0)
            high = f(row.get("high_price"), 0.0)
            low = f(row.get("low_price"), 0.0)
            previous_close = f(row.get("pre_close_price"), 0.0)
            true_ranges = [high - low] if high > 0 and low > 0 and high >= low else []
            if previous_close > 0 and high > 0 and low > 0:
                true_ranges.extend((abs(high - previous_close), abs(low - previous_close)))
            current_true_range = max(true_ranges, default=0.0)
            if len(code) == 6 and code.isdigit() and close > 0 and current_true_range > 0:
                factor_rows.append(
                    {
                        "code": code,
                        "atr_14": current_true_range,
                        "ma_20": float("nan"),
                        "close": close,
                        "mom_20_context": float("nan"),
                        "drawdown_60_context": float("nan"),
                    }
                )
    if hist_dir == "auto":
        # An explicit cache directory (tests, sandboxes) is used on its own.
        hist_dir = latest_hist_dir() if Path(cache_dir) == FALLBACK_BAR_CACHE else None
    if calendar is None:
        calendar = trading_calendar()
    signal_dates = {pd.Timestamp(row.get("date")).normalize() for row in rows if row.get("date")}
    calendar = extend_calendar(calendar, signal_dates)

    def load(row):
        code = str(row.get("code") or "").zfill(6)
        selection_date = pd.Timestamp(row.get("date")).normalize()
        bars, meta = load_code_history(
            code, selection_date, calendar=calendar, hist_dir=hist_dir, fallback_dir=cache_dir
        )
        if bars.empty:
            return None
        bars = append_spot_bar(bars, row, selection_date, calendar)
        complete = history_window_is_complete(bars, calendar, selection_date)
        return code, align_to_calendar(bars, calendar), complete, meta

    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded = list(executor.map(lambda row: _safe_call(load, row), rows))

    for result in loaded:
        if result is None:
            continue
        code, bars, complete, meta = result
        try:
            if len(bars) < 60:
                continue
            if selection_features:
                feature = historical_features(bars)
                # Missing sessions shift every rolling window, so they invalidate the shape.
                feature["selection_history_valid"] = bool(feature["selection_history_valid"] and complete)
                feature["history_source"] = meta["source"]
                feature["ex_rights_adjustments"] = meta["ex_rights_events"]
                selection_context[code] = feature
            latest = compute_factors(bars).iloc[-1]
            item = {
                "code": code,
                "atr_14": f(latest.get("atr_14"), float("nan")),
                "ma_20": f(latest.get("ma_20"), float("nan")),
                "close": f(bars.iloc[-1].get("close"), float("nan")),
                "mom_20_context": f(latest.get("mom_20"), float("nan")),
                "drawdown_60_context": f(latest.get("drawdown_60"), float("nan")),
            }
            for factor in weights:
                # Factors from gapped windows would rank stocks on mismatched horizons.
                item[factor] = f(latest.get(factor), float("nan")) if complete else float("nan")
            factor_rows.append(item)
        except Exception:
            continue
    if not factor_rows:
        return {}
    frame = pd.DataFrame(factor_rows).drop_duplicates("code", keep="last").set_index("code")
    normalized = None
    if weights:
        weighted = pd.Series(0.0, index=frame.index)
        total_weight = sum(abs(float(value)) for value in weights.values()) or 1.0
        for factor, weight in weights.items():
            ranked = pd.to_numeric(frame[factor], errors="coerce").rank(pct=True, method="average").fillna(0.5)
            weighted += (ranked - 0.5) * float(weight)
        normalized = (2.0 * weighted / total_weight).clip(-1.0, 1.0)
    breadth = float((frame["close"] > frame["ma_20"]).mean())
    momentum_values = pd.to_numeric(frame["mom_20_context"], errors="coerce")
    drawdown_values = pd.to_numeric(frame["drawdown_60_context"], errors="coerce")
    median_momentum = float(momentum_values.median()) if momentum_values.notna().any() else float("nan")
    median_drawdown = float(drawdown_values.median()) if drawdown_values.notna().any() else float("nan")
    return {
        str(code): {
            **selection_context.get(str(code), {}),
            "historical_factor_score": (
                round(float(normalized.loc[code]), 8) if normalized is not None else None
            ),
            "atr_14": float(frame.loc[code, "atr_14"]),
            "breadth_above_ma20": breadth,
            "market_median_mom20": median_momentum,
            "market_median_drawdown60": median_drawdown,
        }
        for code in frame.index
    }


def load_historical_factor_scores(rows, weights, cache_dir=Path(r"C:\InStockLocal\backtest_cache\daily_bars")):
    """Compatibility wrapper for callers that only consume normalized scores."""
    context = load_historical_context(rows, weights, cache_dir)
    return {code: item["historical_factor_score"] for code, item in context.items()}


def rebound_trend_bonus(row, cfg, strategy_hits):
    if cfg.get("score_profile") != "rebound_trend":
        return 0, [], []

    code = str(row.get("code") or "")
    score = 0
    reasons = []
    risks = []
    signal_count = 0

    change_rate = f(row.get("change_rate"))
    turnoverrate = f(row.get("turnoverrate"))
    volume_ratio = f(row.get("volume_ratio"))
    fund_amount = f(row.get("fund_amount"))
    fund_amount_3 = f(row.get("fund_amount_3"))
    fund_amount_5 = f(row.get("fund_amount_5"))

    if -2.5 <= change_rate <= 5:
        score += 8
        signal_count += 1
        reasons.append("反弹趋势:涨幅温和")
    elif 5 < change_rate <= 7.5:
        score += 2
        risks.append("短线涨幅偏高，注意回撤")
    elif change_rate > 7.5:
        score -= 18
        risks.append("涨幅过高，避免追高")
    elif change_rate < -4:
        score -= 8
        risks.append("仍在下跌趋势")

    if 0.8 <= volume_ratio <= 2.8:
        score += 6
        signal_count += 1
        reasons.append("反弹趋势:量能温和放大")
    elif volume_ratio > 3.8:
        score -= 4
        risks.append("放量过猛，可能冲高回落")

    if 1.5 <= turnoverrate <= 12:
        score += 4
        signal_count += 1
        reasons.append("反弹趋势:换手健康")
    elif turnoverrate > 18:
        score -= 4
        risks.append("换手过热")

    signal_bits = [
        ("low_funds_inflow", "低位资金流入"),
        ("macd_golden_fork", "MACD金叉"),
        ("kdj_golden_fork", "KDJ金叉"),
        ("breakup_ma_5days", "突破5日线"),
        ("breakup_ma_10days", "突破10日线"),
        ("breakup_ma_20days", "突破20日线"),
        ("break_through", "放量突破"),
    ]
    hit_labels = []
    for col, label in signal_bits:
        if bit(row.get(col)):
            score += 4
            signal_count += 1
            hit_labels.append(label)
    if hit_labels:
        reasons.append("反弹趋势信号:" + ",".join(hit_labels[:5]))

    strategy_text = " ".join(strategy_hits.get(code, []))
    preferred_strategy_hits = [
        label
        for label in strategy_hits.get(code, [])
        if any(key in label for key in ["回踩", "突破", "均线", "海龟", "低ATR", "成长"])
    ]
    if preferred_strategy_hits:
        score += min(12, 5 * len(preferred_strategy_hits))
        signal_count += 1
        reasons.append("反弹趋势策略:" + ",".join(preferred_strategy_hits[:3]))
    elif strategy_text:
        score += 2

    if signal_count >= 4:
        score += 8
        reasons.append("反弹趋势确认")
    elif signal_count <= 1:
        score -= 8
        risks.append("反弹确认信号偏少")

    return score, reasons, risks


def score_row(row, cfg, spot_buy, indicator_buy, indicator_sell, strategy_hits):
    code = str(row.get("code") or "")
    reasons = []
    risks = []
    matched = tech_matches(row, cfg)
    score = 0

    historical_profile = cfg.get("historical_profile")
    historical_score = row.get("historical_factor_score")
    if historical_profile and historical_score is None and cfg.get("require_historical_factors", False):
        return None

    require_focus_match = cfg.get("require_focus_match", True)
    if not matched and require_focus_match:
        return None
    if matched:
        score += f(cfg.get("focus_base_score"), 30)
        reasons.append(cfg.get("focus_label", "科技方向"))

        core_hit_count = min(len(matched), 5)
        score += core_hit_count * f(cfg.get("keyword_score"), 3)
        if core_hit_count:
            reasons.append("概念命中:" + ",".join(matched[:5]))
    else:
        score += f(cfg.get("all_market_base_score"), 18)
        reasons.append(cfg.get("all_market_label", "全市场候选"))

    industry = str(row.get("industry") or "")
    focus_industries = cfg.get("focus_industries", cfg.get("tech_industries", []))
    if industry in focus_industries:
        score += f(cfg.get("focus_industry_bonus"), 8)
        reasons.append(cfg.get("focus_industry_label", "科技/高端制造行业"))

    change_rate = f(row.get("change_rate"))
    turnoverrate = f(row.get("turnoverrate"))
    volume_ratio = f(row.get("volume_ratio"))
    deal_amount = f(row.get("deal_amount"))

    if change_rate > 0:
        score += 6
        reasons.append("当日上涨")
    if change_rate >= 3:
        score += 8
        reasons.append("进攻涨幅")
    if change_rate > 9.5:
        score -= 4
        risks.append("涨幅接近涨停，追高风险")
    if change_rate < -2:
        score -= 6
        risks.append("短线走弱")

    if volume_ratio >= 1.2:
        score += 6
        reasons.append("量比放大")
    if 2 <= turnoverrate <= 15:
        score += 6
        reasons.append("换手活跃")
    elif turnoverrate > 20:
        score -= 3
        risks.append("换手过热")
    if deal_amount >= 200000000:
        score += 5
        reasons.append("成交额充足")

    fund_amount = f(row.get("fund_amount"))
    fund_amount_3 = f(row.get("fund_amount_3"))
    fund_amount_5 = f(row.get("fund_amount_5"))
    fund_score, fund_reasons, fund_risks = score_fund_flows(
        fund_amount, fund_amount_3, fund_amount_5
    )
    score += fund_score
    reasons.extend(fund_reasons)
    risks.extend(fund_risks)

    shareholder_score, shareholder_reasons, shareholder_risks, shareholder_summary = (
        score_shareholder_structure(
            institutional_ratio=row.get("allcorp_ratio"),
            institutional_count=row.get("allcorp_num"),
            holder_count_change=row.get("holder_ratio"),
            top10_ratio=row.get("hold_ratio_count"),
            survey_count=row.get("org_survey_3m"),
        )
    )
    score += shareholder_score
    reasons.extend(shareholder_reasons)
    risks.extend(shareholder_risks)

    pe = f(row.get("pe9"))
    pb = f(row.get("pbnewmrq"))
    # Year-to-date ROE drifts with the reporting season; prefer the TTM figure.
    roe_ttm = ttm_roe_pct(row)
    roe = roe_ttm if roe_ttm is not None else f(row.get("roe_weight"))
    cash_positive = operating_cash_flow_positive(row)
    profit_growth_value = f(row.get("netprofit_yoy_ratio"), None)
    profit_growth = profit_growth_value if profit_growth_value is not None else 0.0
    debt = f(row.get("debt_asset_ratio"))
    three_year_profit_growth = f(row.get("netprofit_growthrate_3y"), None)
    operating_cash_flow = f(row.get("per_netcash_operate"), None)
    dividend_yield_pct = f(row.get("zxgxl"), None)

    if 0 < pe <= 80:
        score += 5
        reasons.append("PE可接受")
    elif pe > 120:
        score -= 6
        risks.append("估值偏高")
    if 0 < pb <= 10:
        score += 4
    if roe >= 15:
        score += 8
        reasons.append("ROE较强")
    elif roe >= 8:
        score += 4
    if profit_growth > 0:
        score += 5
        reasons.append("净利同比增长")
    if debt and debt < 70:
        score += 2

    positive_bits = [
        ("macd_golden_fork", "MACD金叉"),
        ("kdj_golden_fork", "KDJ金叉"),
        ("break_through", "放量突破"),
        ("low_funds_inflow", "低位资金流入"),
        ("breakup_ma_5days", "突破5日线"),
        ("breakup_ma_10days", "突破10日线"),
        ("breakup_ma_20days", "突破20日线"),
        ("long_avg_array", "均线多头"),
        ("upper_large_volume", "连涨放量"),
        ("upside_volume", "放量上攻"),
        ("now_newhigh", "近期新高")
    ]
    for col, label in positive_bits:
        if bit(row.get(col)):
            score += 4
            reasons.append(label)

    if bit(row.get("high_funds_outflow")):
        score -= 8
        risks.append("高位资金流出")

    if code in spot_buy:
        score += 12
        reasons.append("基本面买入池")
    if code in indicator_buy:
        score += 18
        reasons.append("指标买入信号")
    if code in indicator_sell:
        score -= 18
        risks.append("指标卖出信号")
    if strategy_hits.get(code):
        score += min(20, 8 * len(strategy_hits[code]))
        reasons.append("策略命中:" + ",".join(strategy_hits[code][:3]))

    if historical_profile and historical_score is not None:
        historical_score = f(historical_score)
        score += historical_score * f(cfg.get("historical_factor_score_scale"), 30)
        if historical_score > 0:
            reasons.append("历史样本外因子排名靠前")
        elif historical_score < -0.3:
            risks.append("历史趋势/风险因子排名偏弱")

    bonus, profile_reasons, profile_risks = rebound_trend_bonus(row, cfg, strategy_hits)
    score += bonus
    reasons.extend(profile_reasons)
    risks.extend(profile_risks)

    if reject_name(row, cfg):
        return None

    if pe <= 0:
        risks.append("PE为负或缺失")
    if not risks:
        risks.append("仍需人工复核公告/业绩/位置")

    grouped = {}
    if cfg.get("scoring_version") == "grouped-v2":
        grouped = score_groups_v2(row, cfg, shareholder_score=shareholder_score, indicator_sell=code in indicator_sell)
        score, reasons, risks = grouped["score"], grouped["reasons"], grouped["risks"]
    elif cfg.get("scoring_version") == "grouped-v1":
        grouped = score_groups(row, cfg, shareholder_score=shareholder_score, indicator_sell=code in indicator_sell)
        score, reasons, risks = grouped["score"], grouped["reasons"], grouped["risks"]

    large_order_flow, retail_flow, order_flow_source = resolve_grouped_flows_wan(
        fund_amount=row.get("fund_amount"),
        fund_amount_super=row.get("fund_amount_super"),
        fund_amount_large=row.get("fund_amount_large"),
        fund_amount_medium=row.get("fund_amount_medium"),
        fund_amount_small=row.get("fund_amount_small"),
    )

    return {
        **{key: value for key, value in grouped.items() if key not in ("score", "reasons", "risks")},
        "排名": 0,
        "综合评分": round(score, 1),
        "代码": code,
        "名称": row.get("name"),
        "行业": industry,
        "最新价": f(row.get("new_price"), None),
        "涨跌幅%": round(change_rate, 2),
        "换手率%": round(turnoverrate, 2),
        "量比": round(volume_ratio, 2),
        "成交额_万元": money_wan(row.get("deal_amount")),
        "PE": round(pe, 2) if pe else None,
        "PB": round(pb, 2) if pb else None,
        "ROE%": round(roe, 2) if roe else None,
        "ROE口径": "TTM" if roe_ttm is not None else "报告期累计",
        "经营现金流为正": cash_positive,
        "净利同比%": (
            round(profit_growth_value, 2)
            if profit_growth_value is not None
            else None
        ),
        "三年净利增速%": (
            round(three_year_profit_growth, 2)
            if three_year_profit_growth is not None
            else None
        ),
        "资产负债率%": round(debt, 2) if debt else None,
        "每股经营现金流": (
            round(operating_cash_flow, 4)
            if operating_cash_flow is not None
            else None
        ),
        "股息率%": (
            round(dividend_yield_pct, 2)
            if dividend_yield_pct is not None
            else None
        ),
        "今日主力净流入_万元": money_wan(row.get("fund_amount")),
        "今日大单净额_万元": large_order_flow,
        "今日散户净额_万元": retail_flow,
        "订单资金日期": row.get("fund_component_date"),
        "订单资金来源": order_flow_source,
        "3日主力净流入_万元": money_wan(row.get("fund_amount_3")),
        "5日主力净流入_万元": money_wan(row.get("fund_amount_5")),
        "股东结构评分": round(shareholder_score, 1),
        "机构持股比例%": f(row.get("allcorp_ratio"), None),
        "机构持股家数": f(row.get("allcorp_num"), None),
        "股东户数变化%": f(row.get("holder_ratio"), None),
        "十大股东持股比例%": f(row.get("hold_ratio_count"), None),
        "近3月机构调研数": f(row.get("org_survey_3m"), None),
        "股东结构说明": shareholder_summary,
        "命中关键词": ",".join(matched[:10]),
        "加分理由": "；".join(dict.fromkeys(reasons)),
        "风险提示": "；".join(dict.fromkeys(risks)),
        "概念": row.get("concept"),
        "历史因子评分": round(f(historical_score), 4) if historical_score is not None else None,
        "ATR14": round(f(row.get("atr_14")), 6) if row.get("atr_14") is not None else None,
    }


def screener_lag_warning(snapshot):
    screener = snapshot.get("screener_date")
    signal = snapshot.get("selection_date")
    if screener and signal and str(screener) < str(signal):
        return (
            f"综合选股数据停在{clean_value(screener)}，价格与估值已用{clean_value(signal)}行情；"
            "概念、财务与股东数据沿用旧快照"
        )
    return ""


def load_selection_snapshot(cur, as_of_date=None):
    selection_date = latest_date(cur, "cn_stock_selection", as_of_date)
    spot_date = latest_date(cur, "cn_stock_spot", as_of_date)
    fund_date = latest_date(cur, "cn_stock_fund_flow", as_of_date)
    if not selection_date:
        raise RuntimeError(
            "cn_stock_selection has no data. Run 沧浪之水 data update first."
        )
    spot_buy = fetch_code_set(cur, "cn_stock_spot_buy", as_of_date)
    indicator_buy = fetch_code_set(cur, "cn_stock_indicators_buy", as_of_date)
    indicator_sell = fetch_code_set(
        cur, "cn_stock_indicators_sell", as_of_date
    )
    strategy_hits = fetch_strategy_hits(cur, as_of_date)
    # The screener table (cn_stock_selection) comes from a flaky source and can
    # lag the daily quotes.  Prices, valuation ratios and the signal date must
    # then come from the newer quote table; slower-moving fundamentals,
    # concepts and shareholder data keep the screener's last snapshot.
    quotes_newer = bool(spot_date and str(spot_date) > str(selection_date))
    market = "COALESCE(p.{0}, s.{0}) AS {0}" if quotes_newer else "s.{0}"
    market_columns = ", ".join(
        market.format(column)
        for column in (
            "new_price", "change_rate", "turnoverrate", "volume_ratio", "deal_amount",
            "high_price", "low_price", "pre_close_price", "volume", "pe9", "pbnewmrq",
        )
    )
    date_column = "%s AS date" if quotes_newer else "s.date"
    quote_join = (
        "LEFT JOIN cn_stock_spot p ON p.code = s.code AND p.date = %s" if quotes_newer else ""
    )
    cur.execute(
        f"""
        SELECT
          {date_column}, s.code, s.name, s.industry, s.concept, s.style,
          {market_columns},
          s.roe_weight, s.zxgxl, s.netprofit_yoy_ratio,
          s.netprofit_growthrate_3y, s.debt_asset_ratio, s.per_netcash_operate, s.pcfjyxjl9,
          s.macd_golden_fork, s.kdj_golden_fork, s.break_through, s.low_funds_inflow,
          s.high_funds_outflow, s.breakup_ma_5days, s.breakup_ma_10days,
          s.breakup_ma_20days, s.long_avg_array, s.upper_large_volume,
          s.upside_volume, s.now_newhigh,
          s.holder_ratio, s.hold_ratio_count, s.org_survey_3m,
          s.allcorp_num, s.allcorp_ratio,
          f.date AS fund_component_date,
          f.fund_amount, f.fund_rate, f.fund_amount_super,
          f.fund_amount_large, f.fund_amount_medium, f.fund_amount_small,
          f.fund_amount_3, f.fund_rate_3, f.fund_amount_5, f.fund_rate_5
        FROM cn_stock_selection s
        LEFT JOIN cn_stock_fund_flow f
          ON f.code = s.code AND f.date = %s
        {quote_join}
        WHERE s.date = %s
        """,
        (
            (spot_date, fund_date, spot_date, selection_date)
            if quotes_newer
            else (fund_date, selection_date)
        ),
    )
    return {
        "selection_date": spot_date if quotes_newer else selection_date,
        "screener_date": selection_date,
        "spot_date": spot_date,
        "fund_date": fund_date,
        "spot_buy": spot_buy,
        "indicator_buy": indicator_buy,
        "indicator_sell": indicator_sell,
        "strategy_hits": strategy_hits,
        "rows": cur.fetchall(),
    }


def main():
    if len(sys.argv) < 2:
        print(
            "usage: instock_tech_picker_data.py output.json [config.json] [as_of_date]",
            file=sys.stderr,
        )
        return 2
    output_path = Path(sys.argv[1])
    cfg = load_config(sys.argv[2] if len(sys.argv) >= 3 else None)
    as_of_date = None
    if len(sys.argv) >= 4:
        try:
            as_of_date = pd.Timestamp(sys.argv[3]).date()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无效历史选股日期：{sys.argv[3]}") from exc

    with connect() as conn:
        cur = conn.cursor()
        snapshot = load_selection_snapshot(cur, as_of_date)
        selection_date = snapshot["selection_date"]
        spot_date = snapshot["spot_date"]
        fund_date = snapshot["fund_date"]
        spot_buy = snapshot["spot_buy"]
        indicator_buy = snapshot["indicator_buy"]
        indicator_sell = snapshot["indicator_sell"]
        strategy_hits = snapshot["strategy_hits"]
        rows = snapshot["rows"]

    refined = str(cfg.get("scoring_version", "")).startswith("grouped")
    historical_context = load_historical_context(rows, cfg.get("historical_factor_weights", {}), selection_features=refined)
    if refined:
        for row in rows:
            row.update(historical_context.get(str(row.get("code") or ""), {}))
        attach_peer_momentum(rows)
        if cfg.get("scoring_version") == "grouped-v2":
            attach_evidence_scores(rows)

    scored = []
    for row in rows:
        row = {k: clean_value(v) for k, v in row.items()}
        code = str(row.get("code") or "")
        if code in historical_context:
            row.update(historical_context[code])
        item = score_row(row, cfg, spot_buy, indicator_buy, indicator_sell, strategy_hits)
        if item:
            scored.append(item)

    if cfg.get("selection_mode") == "market_earnings_v3":
        market_earnings_rules = cfg.get("market_earnings_v3") or {}
        roe_minimums = load_market_earnings_roe_minimums(
            scored,
            as_of_date=selection_date,
            years=int(market_earnings_rules.get("historical_roe_years", 3)),
            max_workers=int(market_earnings_rules.get("history_fetch_workers", 8)),
        )
        attach_market_earnings_roe_minimums(scored, roe_minimums)

    apply_news_to_scored_rows(scored, cfg, selection_date)
    scored.sort(key=lambda x: (x["综合评分"], x["今日主力净流入_万元"], x["涨跌幅%"]), reverse=True)
    for idx, row in enumerate(scored, start=1):
        row["排名"] = idx

    regime = "legacy"
    active_count = int(cfg["top_n"])
    abstain_reason = ""
    market_policy_data = cfg.get("market_policy")
    if isinstance(market_policy_data, dict) and historical_context:
        market_policy = MarketPolicy.from_dict(market_policy_data)
        context = next(iter(historical_context.values()))
        regime = classify_regime(
            context["breadth_above_ma20"],
            context["market_median_mom20"],
            context["market_median_drawdown60"],
            market_policy,
        )
        active_count = pick_count_for_regime(regime, market_policy)
        if active_count == 0:
            abstain_reason = "当前市场未通过策略风险门槛，本次保持空仓观察"
    selected = select_output_rows(
        scored,
        cfg,
        active_count=active_count,
        regime=regime,
    )
    if not selected and not abstain_reason:
        abstain_reason = (
            "今日没有通过严格条件的候选，保持观望"
            if cfg.get("fallback_fill", True) is False
            else "本次没有通过基础条件的候选，请检查历史行情、风险提示与筛选范围"
        )

    fund_flow = sorted(
        [r for r in scored if f(r.get("今日主力净流入_万元")) > 0],
        key=lambda x: x["今日主力净流入_万元"],
        reverse=True,
    )[:100]
    fundamentals = select_fundamental_rows(scored, cfg)
    sell_signal = [
        r for r in scored
        if "指标卖出信号" in r["风险提示"] or r.get("新闻重大风险")
    ][:100]

    top_sheet_name = cfg.get("top_sheet_name", "科技进攻Top")
    fund_sheet_name = cfg.get("fund_sheet_name", "资金流入Top")
    fundamental_sheet_name = cfg.get("fundamental_sheet_name", "基本面科技")
    risk_sheet_name = cfg.get("risk_sheet_name", "卖出风险")
    all_sheet_name = cfg.get("all_sheet_name", "全部科技候选")
    candidate_label = cfg.get("candidate_label", "科技候选数")
    all_sample_title = cfg.get("all_sample_title", all_sheet_name)

    payload = {
        "order_flow_schema": "grouped-order-flow-v2",
        "title": cfg["title"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_dates": {
            "综合选股": clean_value(snapshot.get("screener_date", selection_date)),
            "选股计算日": clean_value(selection_date),
            "每日行情": clean_value(spot_date),
            "资金流向": clean_value(fund_date),
        },
        "summary": {
            "全市场记录数": len(rows),
            candidate_label: len(scored),
            "输出Top数": len(selected),
            "基本面买入池": len(spot_buy),
            "指标买入信号": len(indicator_buy),
            "指标卖出信号": len(indicator_sell),
            "策略命中股票数": len(strategy_hits),
            "market_regime": regime,
            "active_count": active_count,
            "abstain_reason": abstain_reason,
            "data_warning": screener_lag_warning(snapshot),
            "policy_version": cfg.get("policy_version"),
            "selection_version": cfg.get("selection_version", "legacy"),
            "严格入选数": sum(row.get("入选层级") == "严格入选" for row in selected),
            "候补入选数": sum(row.get("入选层级") == "候补入选" for row in selected),
        },
        "strategy_gate_passed": bool(cfg.get("strategy_gate_passed")),
        "trade_policy": cfg.get("trade_policy"),
        "rules_note": cfg["rules_note"],
        "top_sheet_name": top_sheet_name,
        "preview_sheet": top_sheet_name,
        "preview_file": cfg.get("preview_file", "tech_attack_preview.png"),
        "sheet_meta": {
            top_sheet_name: {
                "title": cfg.get("top_sheet_title", top_sheet_name),
                "subtitle": cfg.get("top_sheet_subtitle", "按综合评分排序"),
            },
            fund_sheet_name: {
                "title": cfg.get("fund_sheet_title", fund_sheet_name),
                "subtitle": cfg.get("fund_sheet_subtitle", cfg["title"]),
            },
            fundamental_sheet_name: {
                "title": cfg.get("fundamental_sheet_title", fundamental_sheet_name),
                "subtitle": cfg.get("fundamental_sheet_subtitle", cfg["title"]),
            },
            risk_sheet_name: {
                "title": cfg.get("risk_sheet_title", risk_sheet_name),
                "subtitle": cfg.get("risk_sheet_subtitle", cfg["title"]),
            },
            all_sheet_name: {
                "title": cfg.get("all_sheet_title", all_sample_title),
                "subtitle": cfg.get("all_sheet_subtitle", cfg["title"]),
            },
        },
        "sheets": {
            top_sheet_name: selected,
            fund_sheet_name: fund_flow,
            fundamental_sheet_name: fundamentals,
            risk_sheet_name: sell_signal,
            all_sheet_name: scored[:300],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Selected {len(selected)} rows from {len(scored)} candidates.")


if __name__ == "__main__":
    raise SystemExit(main())
