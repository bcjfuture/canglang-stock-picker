from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from instock_app.settings import AppPaths, STRATEGIES, StrategySpec
from workers.common import write_json_atomic
from workers.top5_reranker import rerank_top_five, sector_context_from_payload


def resolve_engine_module_dir(app_root: Path) -> Path:
    candidates = (app_root / "resources" / "engine", app_root / "engine")
    for candidate in candidates:
        if (candidate / "trade_policy.py").exists():
            return candidate
    raise FileNotFoundError("trade policy engine module is missing")


ENGINE_DIR = resolve_engine_module_dir(Path(__file__).resolve().parents[1])
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
from trade_policy import TradePolicy, calculate_advice
from picker_order_flow import (
    ORDER_FLOW_SCHEMA,
    enrich_selected_rows,
)


Emitter = Callable[[str, str, int, dict | None], None]

REFERENCE_TRADE_POLICY = TradePolicy(
    policy_version="atr-reference-v1",
    holding_days=7,
    atr_multiplier=1.5,
    reward_risk=2.0,
)

HEAD_RERANK_VERSION = "top5-validated-core-v3"
HEAD_RERANK_POLICIES: dict[str, dict[str, object]] = {
    "all_market": {
        "enabled": False,
        "reason": "市赚率3.0使用PB与基本面硬门槛，保留公式排序",
        "oos_gate": "formula_order",
    },
    "tech_attack": {
        "enabled": False,
        "reason": "近期样本外胜率未改善，保留原策略前5",
        "oos_gate": "failed",
    },
    "rebound_trend": {
        "enabled": False,
        "reason": "近期样本外胜率未稳定改善，保留原策略前5",
        "oos_gate": "failed",
    },
    "zhengxi_prosperity": {
        "enabled": False,
        "reason": "近期样本外胜率未稳定改善，保留原策略前5",
        "oos_gate": "failed",
    },
}


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def optional_number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _display_number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    if signed and value > 0:
        text = "+" + text
    return text


def format_performance_review(
    pe: object,
    roe: object,
    profit_growth: object,
    *,
    pb: object = None,
    historical_roe: object = None,
    implied_roe: object = None,
    dividend_yield: object = None,
    payout_ratio: object = None,
    dividend_correction: object = None,
    market_earnings_ratio: object = None,
) -> str:
    values = tuple(optional_number(value) for value in (pe, roe, profit_growth))
    if all(value is None for value in values):
        return "数据不足，需人工复核"
    pe_value, roe_value, growth_value = values
    roe_text = _display_number(roe_value)
    growth_text = _display_number(growth_value, signed=True)
    if roe_value is not None:
        roe_text += "%"
    if growth_value is not None:
        growth_text += "%"
    review = (
        f"PE {_display_number(pe_value)}｜"
        f"ROE {roe_text}｜"
        f"净利同比 {growth_text}"
    )
    market_ratio_value = optional_number(market_earnings_ratio)
    if market_ratio_value is None:
        return review
    return (
        review
        + f"｜PB {_display_number(optional_number(pb))}"
        + f"｜三年最低ROE {_display_number(optional_number(historical_roe))}%"
        + f"｜股息率 {_display_number(optional_number(dividend_yield))}%"
        + f"｜估算分红率 {_display_number(optional_number(payout_ratio))}%"
        + f"｜N {_display_number(optional_number(dividend_correction))}"
        + f"｜市赚率3.0 {_display_number(market_ratio_value)}"
    )


def _latest_trade_date(payload: dict) -> str:
    values = [str(value)[:10] for value in payload.get("latest_dates", {}).values() if value]
    return max(values) if values else str(payload.get("generated_at", ""))[:10]


def probability_context_from_payload(payload: dict, model: dict) -> dict[str, object]:
    del model
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    sheets = payload.get("sheets", {})
    rows = sheets.get("全部概率候选", []) if isinstance(sheets, dict) else []
    selected_rows = sheets.get("高胜率精选", []) if isinstance(sheets, dict) else []
    probabilities_by_code: dict[str, float] = {}
    precision_scores_by_code: dict[str, float] = {}
    gate_passed_by_code: dict[str, bool] = {}
    if isinstance(rows, list):
        denominator = max(1, len(rows) - 1)
        for index, row in enumerate(rows):
            code = str(row.get("代码", "")).strip().zfill(6)
            probability = optional_number(row.get("5日上涨概率"))
            if len(code) == 6 and code.isdigit() and probability is not None:
                probabilities_by_code[code] = probability
                precision_scores_by_code[code] = 1.0 - index / denominator
                if bool(row.get("通过高胜率门槛")):
                    gate_passed_by_code[code] = True
    if isinstance(selected_rows, list):
        for row in selected_rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("代码", "")).strip().zfill(6)
            if len(code) == 6 and code.isdigit():
                gate_passed_by_code[code] = True
    threshold = optional_number(summary.get("probability_threshold"))
    regime = str(summary.get("market_regime") or "").strip()
    trade_date = _latest_trade_date(payload)
    unavailable_codes = sorted(
        {
            str(code).strip().zfill(6)
            for code in summary.get("unavailable_probability_codes", [])
            if str(code).strip()
        }
    )
    if threshold is None or not regime or not trade_date:
        raise ValueError("高胜率概率上下文不完整")
    return {
        "trade_date": trade_date,
        "market_regime": regime,
        "probability_threshold": threshold,
        "probabilities_by_code": probabilities_by_code,
        "precision_scores_by_code": precision_scores_by_code,
        "gate_passed_by_code": gate_passed_by_code,
        "selected_rows": [
            dict(row)
            for row in selected_rows
            if isinstance(row, dict) and bool(row.get("通过高胜率门槛", True))
        ] if isinstance(selected_rows, list) else [],
        "unavailable_codes": unavailable_codes,
    }


def apply_head_reranking(
    spec: StrategySpec,
    payload: dict,
    precision_context: dict[str, object],
    sector_context: dict[str, object],
    *,
    all_sheet_name: str,
    policy: dict[str, object] | None = None,
) -> None:
    sheets = payload.get("sheets")
    if not isinstance(sheets, dict):
        raise ValueError("选股结果缺少 sheets")
    source_rows = sheets.get(all_sheet_name)
    if not isinstance(source_rows, list) or not source_rows:
        source_rows = sheets.get(spec.top_sheet)
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError(f"选股结果缺少 {spec.top_sheet}")
    policy = dict(policy or {})
    summary = payload.setdefault("summary", {})
    if sheets.get(spec.top_sheet) == [] and summary.get("abstain_reason"):
        return
    if not bool(policy.get("enabled", True)):
        top_rows = sheets.get(spec.top_sheet)
        preserved_rows = (
            top_rows
            if isinstance(top_rows, list) and top_rows
            else source_rows
        )
        original = [dict(row) for row in preserved_rows[:20]]
        for index, row in enumerate(original, start=1):
            row["排名"] = index
            row["原策略排名"] = index
            row["头部重排"] = False
            row["头部重排评分"] = None
            row["板块因子权重"] = 0.0
            row["热门板块"] = ""
        sheets[spec.top_sheet] = original
        if isinstance(summary, dict):
            summary["head_rerank_version"] = HEAD_RERANK_VERSION
            summary["head_rerank_enabled"] = False
            summary["head_rerank_gate"] = str(policy.get("oos_gate", "failed"))
            summary["head_rerank_reason"] = str(policy.get("reason", ""))
        return
    reranked = rerank_top_five(
        source_rows[:100],
        precision_context,
        sector_context,
        total_count=20,
        head_count=5,
        industry_cap=int(policy.get("industry_cap", 2)),
        candidate_limit=int(policy.get("candidate_limit", min(100, len(source_rows)))),
        precision_weight=float(policy.get("precision_weight", 0.25)),
        probability_weight=float(policy.get("probability_weight", 0.10)),
        sector_weight_scale=float(policy.get("sector_weight_scale", 1.0)),
        core_rows=(
            list(precision_context.get("selected_rows", []))
            if bool(policy.get("use_global_core"))
            and isinstance(precision_context.get("selected_rows"), list)
            else []
        ),
    )
    if len(reranked) < min(20, len(source_rows)):
        raise ValueError("头部重排没有保留足够候选")
    for row in reranked[:5]:
        hot_sector = str(row.get("热门板块", "")).strip()
        if not hot_sector:
            continue
        weight = optional_number(row.get("板块因子权重")) or 0.0
        label = "热门板块加分" if weight >= 0.15 else "板块参考平分"
        reason = str(row.get("加分理由", "")).strip()
        addition = f"{label}：{hot_sector}"
        row["加分理由"] = f"{reason}；{addition}" if reason else addition
    sheets[spec.top_sheet] = reranked
    if isinstance(summary, dict):
        summary["head_rerank_version"] = HEAD_RERANK_VERSION
        summary["head_rerank_enabled"] = True
        summary["head_rerank_gate"] = str(policy.get("oos_gate", "unknown"))
        summary["head_rerank_shortlist_count"] = min(100, len(source_rows))
        summary["head_rerank_candidate_limit"] = int(
            policy.get("candidate_limit", min(100, len(source_rows)))
        )
        summary["head_rerank_industry_cap"] = int(policy.get("industry_cap", 2))


def _is_hard_exit(row: dict) -> bool:
    risk = str(row.get("风险提示", ""))
    return bool(row.get("新闻重大风险")) or "指标卖出信号" in risk


def apply_holding_buffer(
    spec: StrategySpec,
    payload: dict,
    previous: dict | None,
    holding: dict | None,
    *,
    rank_sheet_name: str,
    top_n: int = 20,
) -> None:
    """Keep the previous picks that still rank inside a wider buffer.

    Re-picking the top names from scratch every day replaced most of the list
    daily; the round-trip cost of that turnover exceeded the signal.  A name
    stays while it ranks within ``buffer_rank`` and has been held fewer than
    ``max_hold_runs`` runs; free slots go to today's best new names.
    """
    holding = holding or {}
    buffer_rank = int(holding.get("buffer_rank", 0) or 0)
    max_runs = int(holding.get("max_hold_runs", 0) or 0)
    sheets = payload.get("sheets")
    if not isinstance(sheets, dict):
        return
    top = [dict(row) for row in sheets.get(spec.top_sheet) or [] if isinstance(row, dict)]
    trade_date = _latest_trade_date(payload)
    previous_runs: dict[str, int] = {}
    if (
        isinstance(previous, dict)
        and str(previous.get("trade_date", "")) < trade_date
        and isinstance(previous.get("rows"), list)
    ):
        for row in previous["rows"]:
            code = str(row.get("code", "")).strip().zfill(6)
            previous_runs[code] = int(_number(row.get("holding_runs"), 1.0)) or 1

    ranking = [row for row in sheets.get(rank_sheet_name) or [] if isinstance(row, dict)]
    rank_by_code: dict[str, int] = {}
    row_by_code: dict[str, dict] = {}
    for index, row in enumerate(ranking, start=1):
        code = str(row.get("代码", "")).strip().zfill(6)
        rank_by_code.setdefault(code, index)
        row_by_code.setdefault(code, row)

    top_codes = {str(row.get("代码", "")).strip().zfill(6) for row in top}
    retained: list[dict] = []
    if buffer_rank > 0:
        for code, runs in previous_runs.items():
            if code in top_codes or code not in row_by_code:
                continue
            rank = rank_by_code[code]
            if rank > buffer_rank or (max_runs and runs >= max_runs):
                continue
            row = dict(row_by_code[code])
            if _is_hard_exit(row):
                continue
            row["入选层级"] = "延续持有"
            note = f"延续持有：今日排名{rank}，仍在前{buffer_rank}名"
            reason = str(row.get("加分理由", "")).strip()
            row["加分理由"] = f"{note}；{reason}" if reason else note
            retained.append(row)

    retained_codes = {str(row.get("代码", "")).strip().zfill(6) for row in retained}
    # Retained names keep their slots; today's new names fill what is left.
    new_rows = [row for row in top if str(row.get("代码", "")).strip().zfill(6) not in retained_codes]
    combined = retained + new_rows[: max(0, top_n - len(retained))]
    combined.sort(key=lambda row: rank_by_code.get(str(row.get("代码", "")).strip().zfill(6), 10**6))
    for index, row in enumerate(combined, start=1):
        code = str(row.get("代码", "")).strip().zfill(6)
        row["排名"] = index
        row["持有次数"] = previous_runs.get(code, 0) + 1
    sheets[spec.top_sheet] = combined
    summary = payload.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["holding_buffer_rank"] = buffer_rank
        summary["holding_retained"] = len(retained)
        summary["holding_new"] = len(combined) - len(retained)
        if combined and summary.get("abstain_reason"):
            summary["abstain_reason"] = ""


def load_previous_result(paths: AppPaths, spec: StrategySpec, trade_date: str) -> dict | None:
    """Latest stored run from an earlier trading day.

    Same-day re-runs must not become their own predecessor, or the holding
    count would restart at every re-run.
    """
    best: dict | None = None
    history_dir = paths.data_dir / "store" / "history" / spec.key
    candidates = list(history_dir.glob("*.json")) if history_dir.exists() else []
    candidates.append(paths.data_dir / "results" / spec.result_name)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        date_value = str(payload.get("trade_date", ""))
        if not date_value or date_value >= trade_date:
            continue
        key = (date_value, str(payload.get("generated_at", "")))
        if best is None or key > (str(best.get("trade_date", "")), str(best.get("generated_at", ""))):
            best = payload
    return best


def load_sector_context(paths: AppPaths, trade_date: str) -> dict[str, object]:
    candidates: list[Path] = [paths.data_dir / "sector-forecast-latest.json"]
    history_root = paths.data_dir / "sector_forecast" / "history"
    candidates.extend(sorted(history_root.glob(f"{trade_date}-*.json"), reverse=True))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            context = sector_context_from_payload(payload, trade_date)
            if context.get("industries") or context.get("concepts"):
                return context
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return {}


def apply_probability_context(
    spec: StrategySpec, payload: dict, context: dict[str, object]
) -> None:
    trade_date = _latest_trade_date(payload)
    if trade_date != str(context.get("trade_date", "")):
        raise ValueError("概率上下文与选股交易日不一致")
    sheets = payload.get("sheets")
    source_rows = sheets.get(spec.top_sheet) if isinstance(sheets, dict) else None
    if not isinstance(source_rows, list):
        raise ValueError(f"选股结果缺少 {spec.top_sheet}")
    probabilities_by_code = context.get("probabilities_by_code", {})
    if not isinstance(probabilities_by_code, dict):
        raise ValueError("概率上下文缺少按代码索引的数据")
    gate_passed_by_code = context.get("gate_passed_by_code", {})
    if not isinstance(gate_passed_by_code, dict):
        gate_passed_by_code = {}
    for row in source_rows:
        code = str(row.get("代码", "")).strip().zfill(6)
        row["5日上涨概率"] = probabilities_by_code.get(code)
        row["市场状态"] = context["market_regime"]
        row["概率门槛"] = context["probability_threshold"]
        row["通过高胜率门槛"] = bool(gate_passed_by_code.get(code, False))
    summary = payload.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["probability_context_trade_date"] = trade_date
        summary["probability_threshold"] = context["probability_threshold"]


def load_live_probability_context(
    paths: AppPaths,
    engine_dir: Path,
    trade_date: str,
    emit: Emitter,
    requested_codes: set[str],
) -> dict[str, object]:
    model_path = engine_dir / "high_precision_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model_run = str(model.get("run_id", ""))
    candidates = (
        paths.data_dir / "raw" / "high_precision.json",
        paths.data_dir / "raw" / "probability_context.json",
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            summary = payload.get("summary", {})
            if (
                _latest_trade_date(payload) == trade_date
                and isinstance(summary, dict)
                and str(summary.get("run_id", "")) == model_run
            ):
                context = probability_context_from_payload(payload, model)
                covered = set(context["probabilities_by_code"]) | set(
                    context["unavailable_codes"]
                )
                if requested_codes <= covered:
                    return context
        except (OSError, json.JSONDecodeError, ValueError):
            continue

    emit("progress", "正在补充5日概率和市场状态", 70, {"scope": "probability"})
    output_path = paths.data_dir / "raw" / "probability_context.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(paths.python_exe),
        "-X",
        "utf8",
        str(engine_dir / "high_precision_picker_data.py"),
        str(output_path),
        str(engine_dir / "high_precision_config.json"),
        str(model_path),
        trade_date,
        ",".join(sorted(requested_codes)),
    ]
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(
        command,
        cwd=engine_dir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
        timeout=600,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-1500:]
        raise RuntimeError(f"5日概率上下文计算失败：{detail}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    context = probability_context_from_payload(payload, model)
    covered = set(context["probabilities_by_code"]) | set(
        context["unavailable_codes"]
    )
    missing = sorted(requested_codes - covered)
    if missing:
        raise RuntimeError(f"概率上下文未返回请求股票：{missing[:10]}")
    return context


def normalize_picker_payload(spec: StrategySpec, payload: dict) -> dict[str, object]:
    sheets = payload.get("sheets")
    if not isinstance(sheets, dict):
        raise ValueError("选股结果缺少 sheets")
    source_rows = sheets.get(spec.top_sheet)
    abstain_reason = str(payload.get("summary", {}).get("abstain_reason", "")).strip()
    if not isinstance(source_rows, list) or (not source_rows and not abstain_reason):
        raise ValueError(f"选股结果缺少或清空了 {spec.top_sheet}")
    gate_passed = bool(payload.get("strategy_gate_passed"))
    policy_data = payload.get("trade_policy")
    policy = None
    if gate_passed and isinstance(policy_data, dict):
        try:
            policy = TradePolicy.from_dict(policy_data)
        except (KeyError, TypeError, ValueError):
            policy = None
    using_reference_policy = policy is None
    advice_policy = REFERENCE_TRADE_POLICY if using_reference_policy else policy
    advice_as_of = _latest_trade_date(payload)
    grouped_flow_schema = payload.get("order_flow_schema") == ORDER_FLOW_SCHEMA
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in source_rows:
        code = str(item.get("代码", "")).strip().zfill(6)
        if len(code) != 6 or not code.isdigit() or code in seen:
            continue
        score = _number(item.get("综合评分"), float("nan"))
        if not math.isfinite(score):
            continue
        pe = optional_number(item.get("PE"))
        roe = optional_number(item.get("ROE%"))
        profit_growth = optional_number(item.get("净利同比%"))
        pb = optional_number(item.get("PB"))
        historical_roe = optional_number(item.get("历史最低ROE%"))
        implied_roe = optional_number(item.get("隐含ROE%"))
        dividend_yield = optional_number(item.get("股息率%"))
        payout_ratio = optional_number(item.get("估算分红率%"))
        dividend_correction = optional_number(item.get("分红修正系数"))
        market_earnings_ratio = optional_number(item.get("市赚率3.0"))
        order_flow_date = str(item.get("订单资金日期") or "").strip()
        order_flow_source = str(item.get("订单资金来源") or "").strip()
        exact_grouped_flow = (
            grouped_flow_schema
            and order_flow_date == advice_as_of
            and order_flow_source
            in {"数据库分项", "数据库汇总推导", "当日分项缓存", "实时分项"}
        )
        large_order_flow = (
            optional_number(item.get("今日大单净额_万元"))
            if exact_grouped_flow
            else None
        )
        retail_flow = (
            optional_number(item.get("今日散户净额_万元"))
            if exact_grouped_flow
            else None
        )
        if not grouped_flow_schema:
            normalized_flow_source = "历史口径不兼容"
        elif large_order_flow is None and retail_flow is None:
            normalized_flow_source = "分项暂缺"
        else:
            normalized_flow_source = order_flow_source
        advice = calculate_advice(item.get("最新价"), item.get("ATR14"), advice_policy)
        if using_reference_policy and advice.get("status") == "ready":
            advice = {**advice, "status": "reference_only"}
        has_advice = advice.get("status") in {"ready", "reference_only"}
        seen.add(code)
        rows.append(
            {
                "rank": len(rows) + 1,
                "score": score,
                "code": code,
                "name": str(item.get("名称", "")),
                "industry": str(item.get("行业", "")),
                "price": _number(item.get("最新价")),
                "change_pct": _number(item.get("涨跌幅%")),
                "fund_flow": _number(item.get("今日主力净流入_万元")),
                "fund_flow_1d": optional_number(item.get("今日主力净流入_万元")),
                "fund_flow_3d": optional_number(item.get("3日主力净流入_万元")),
                "fund_flow_5d": optional_number(item.get("5日主力净流入_万元")),
                "shareholder_score": optional_number(item.get("股东结构评分")),
                "institutional_holding_ratio": optional_number(item.get("机构持股比例%")),
                "institutional_holder_count": optional_number(item.get("机构持股家数")),
                "shareholder_count_change": optional_number(item.get("股东户数变化%")),
                "top10_holding_ratio": optional_number(item.get("十大股东持股比例%")),
                "institutional_survey_count": optional_number(item.get("近3月机构调研数")),
                "shareholder_review": str(
                    item.get("股东结构说明") or "股东结构数据不足"
                ),
                "news_score": optional_number(item.get("新闻评分")),
                "news_summary": str(item.get("新闻摘要") or ""),
                "news_count": int(_number(item.get("新闻数量"), 0.0)),
                "news_status": str(item.get("新闻状态") or "未评估"),
                "news_hard_risk": bool(item.get("新闻重大风险", False)),
                "news_as_of": str(item.get("新闻截止日") or ""),
                "large_order_flow": large_order_flow,
                "retail_flow": retail_flow,
                "order_flow_as_of": (
                    order_flow_date
                    if large_order_flow is not None or retail_flow is not None
                    else None
                ),
                "order_flow_source": normalized_flow_source,
                "pe": pe,
                "pb": pb,
                "roe": roe,
                "historical_roe": historical_roe,
                "implied_roe": implied_roe,
                "dividend_yield": dividend_yield,
                "payout_ratio": payout_ratio,
                "dividend_correction": dividend_correction,
                "market_earnings_ratio": market_earnings_ratio,
                "profit_growth": profit_growth,
                "performance_review": format_performance_review(
                    pe,
                    roe,
                    profit_growth,
                    pb=pb,
                    historical_roe=historical_roe,
                    implied_roe=implied_roe,
                    dividend_yield=dividend_yield,
                    payout_ratio=payout_ratio,
                    dividend_correction=dividend_correction,
                    market_earnings_ratio=market_earnings_ratio,
                ),
                "reason": str(item.get("加分理由", "")),
                "risk": str(item.get("风险提示", "")),
                "concept": str(item.get("概念", "")),
                "advice_holding_days": advice.get("advice_holding_days") if has_advice else None,
                "take_profit_price": advice.get("take_profit_price") if has_advice else None,
                "take_profit_pct": advice.get("take_profit_pct") if has_advice else None,
                "stop_loss_price": advice.get("stop_loss_price") if has_advice else None,
                "stop_loss_pct": advice.get("stop_loss_pct") if has_advice else None,
                "position_fraction": advice.get("position_fraction") if has_advice else None,
                "advice_as_of": advice_as_of,
                "policy_version": advice.get("policy_version") if has_advice else None,
                "advice_status": str(advice.get("status", "insufficient_data")),
                "win_probability": optional_number(item.get("5日上涨概率")),
                "market_regime": str(item.get("市场状态", "")),
                "probability_threshold": optional_number(item.get("概率门槛")),
                "high_precision_gate_passed": bool(
                    item.get("通过高胜率门槛")
                ),
                "head_reranked": bool(item.get("头部重排", False)),
                "high_precision_core_slot": bool(item.get("高胜率核心席位", False)),
                "head_rerank_score": optional_number(item.get("头部重排评分")),
                "original_rank": int(_number(item.get("原策略排名"), len(rows) + 1)),
                "hot_sector": str(item.get("热门板块", "")),
                "sector_factor_weight": optional_number(item.get("板块因子权重")),
                "positive_factors": str(item.get("主要正向因子", "")),
                "negative_factors": str(item.get("主要负向因子", "")),
                "selection_tier": str(item.get("入选层级", "")),
                "industry_prosperity_score": optional_number(item.get("行业景气分")),
                "roe_repair_score": optional_number(item.get("ROE修复代理分")),
                "company_quality_score": optional_number(item.get("公司质量分")),
                "liquidity_score": optional_number(item.get("流动性分")),
                "valuation_timing_score": optional_number(item.get("估值时机分")),
                "evidence_status": str(item.get("证据状态", "")),
                "holding_runs": int(_number(item.get("持有次数"), 1.0)) or 1,
            }
        )
        if len(rows) == 20:
            break
    if not rows and not abstain_reason:
        raise ValueError("选股 Top 结果没有有效股票")
    return {
        "strategy": spec.key,
        "title": spec.title,
        "generated_at": str(payload.get("generated_at", datetime.now().isoformat(timespec="seconds"))),
        "trade_date": _latest_trade_date(payload),
        "rows": rows,
        "source_summary": payload.get("summary", {}),
        "all_sheets": sheets,
        "abstain_reason": abstain_reason,
        "market_regime": str(
            payload.get("summary", {}).get("market_regime", "")
        ),
        "probability_threshold": optional_number(
            payload.get("summary", {}).get("probability_threshold")
        ),
        "order_flow_schema": (
            ORDER_FLOW_SCHEMA if grouped_flow_schema else None
        ),
    }


def _development_engine_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "engine"


def engine_script_for(strategy_key: str) -> str:
    return {
        "high_precision": "high_precision_picker_data.py",
        "zhengxi_prosperity": "zhengxi_prosperity_picker_data.py",
    }.get(strategy_key, "instock_tech_picker_data.py")


def run_picker(
    strategy_key: str,
    paths: AppPaths,
    emit: Emitter,
    as_of_date: str | None = None,
) -> dict[str, object]:
    if strategy_key not in STRATEGIES:
        raise ValueError(f"未知策略：{strategy_key}")
    spec = STRATEGIES[strategy_key]
    engine_dir = paths.engine_dir if paths.engine_dir.exists() else _development_engine_dir()
    if not engine_dir.exists():
        raise FileNotFoundError(f"选股引擎目录不存在：{engine_dir}")
    paths = replace(paths, engine_dir=engine_dir)
    raw_path = paths.data_dir / "raw" / f"{strategy_key}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    engine_script = engine_script_for(strategy_key)
    command = [
        str(paths.python_exe),
        "-X",
        "utf8",
        str(engine_dir / engine_script),
        str(raw_path),
        str(engine_dir / spec.config_name),
    ]
    if strategy_key == "high_precision":
        command.append(str(engine_dir / "high_precision_model.json"))
    if as_of_date:
        command.append(str(as_of_date))
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    env["INSTOCK_DATA_DIR"] = str(paths.data_dir)
    emit("progress", f"正在运行{spec.title}", 10, {"strategy": strategy_key})
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(
        command,
        cwd=engine_dir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
        timeout=600,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-1500:]
        raise RuntimeError(f"{spec.title}运行失败：{detail}")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if strategy_key != "high_precision" and payload.get("sheets", {}).get(spec.top_sheet):
        config = json.loads((engine_dir / spec.config_name).read_text(encoding="utf-8"))
        all_sheet_name = str(config.get("all_sheet_name") or spec.top_sheet)
        sheets = payload.get("sheets", {})
        source_rows = sheets.get(all_sheet_name, []) if isinstance(sheets, dict) else []
        if not isinstance(source_rows, list) or not source_rows:
            source_rows = sheets.get(spec.top_sheet, []) if isinstance(sheets, dict) else []
        requested_codes = {
            str(row.get("代码", "")).strip().zfill(6)
            for row in source_rows[:100]
            if str(row.get("代码", "")).strip()
        }
        probability_context: dict[str, object] = {}
        try:
            probability_context = load_live_probability_context(
                paths,
                engine_dir,
                _latest_trade_date(payload),
                emit,
                requested_codes,
            )
        except Exception as exc:
            emit(
                "progress",
                f"高胜率上下文暂不可用，保留原策略排序：{exc}",
                76,
                {"scope": "probability", "degraded": True},
            )
        sector_context = load_sector_context(paths, _latest_trade_date(payload))
        apply_head_reranking(
            spec,
            payload,
            probability_context,
            sector_context,
            all_sheet_name=all_sheet_name,
            policy=HEAD_RERANK_POLICIES.get(strategy_key, {"enabled": False}),
        )
        if probability_context:
            apply_probability_context(spec, payload, probability_context)
    if strategy_key != "high_precision" and not as_of_date:
        # Historical re-runs have no live predecessor, so the buffer only
        # applies to the daily run that follows yesterday's stored result.
        config = json.loads((engine_dir / spec.config_name).read_text(encoding="utf-8"))
        holding = config.get("holding") or {}
        if holding:
            apply_holding_buffer(
                spec,
                payload,
                load_previous_result(paths, spec, _latest_trade_date(payload)),
                holding,
                rank_sheet_name=str(
                    holding.get("rank_sheet") or config.get("all_sheet_name") or spec.top_sheet
                ),
                top_n=int(config.get("top_n", 20)),
            )
    if payload.get("order_flow_schema") == ORDER_FLOW_SCHEMA:
        source_rows = payload.get("sheets", {}).get(spec.top_sheet, [])
        if isinstance(source_rows, list) and source_rows:
            emit(
                "progress",
                "正在补充大单与散户分项资金",
                88,
                {"scope": "order_flow"},
            )
            enriched_rows = enrich_selected_rows(
                source_rows,
                _latest_trade_date(payload),
                cache_path=(
                    paths.data_dir
                    / "cache"
                    / "picker-order-flow"
                    / f"{_latest_trade_date(payload)}.json"
                ),
            )
            payload["sheets"][spec.top_sheet] = enriched_rows
    result = normalize_picker_payload(spec, payload)
    if as_of_date and str(result.get("trade_date", "")) > str(as_of_date):
        raise RuntimeError(
            f"{spec.title}历史选股返回未来日期：{result.get('trade_date')} > {as_of_date}"
        )
    normalized_path = paths.data_dir / "results" / spec.result_name
    write_json_atomic(normalized_path, result)
    warning = str(result.get("source_summary", {}).get("data_warning") or "")
    message = f"{spec.title}完成，共 {len(result['rows'])} 只"
    emit("result", f"{message}（注意：{warning}）" if warning else message, 100, result)
    return result
