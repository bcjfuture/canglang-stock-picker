# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from high_precision_picker import (
    HighPrecisionPolicy,
    apply_probability_calibrator,
    build_high_precision_features,
    classify_high_precision_regime,
    predict_raw_probability,
    select_daily_candidates,
)
from high_precision_v2 import (
    build_v2_features,
    combine_probabilities,
    predict_submodels,
)
from high_precision_v3 import build_pullback_scores, select_pullback_candidates
from high_precision_v4 import build_regime_scores, select_dynamic_candidates
from high_precision_v5 import score_v5_probabilities, select_v5_candidates
from picker_backtest import compute_factors
from instock_tech_picker_data import (
    clean_value,
    connect,
    f,
    load_historical_context,
    load_selection_snapshot,
    resolve_grouped_flows_wan,
    score_shareholder_structure,
)


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT.parent / "data" / "high_precision-backtest-latest.json"
PROFILES_PATH = ROOT / "picker_factor_profiles.json"
DISPLAY_CONFIG_PATH = ROOT / "all_market_config.json"
V2_CONFIG_PATH = ROOT / "high_precision_v2_config.json"
V3_CONFIG_PATH = ROOT / "high_precision_v3_config.json"
V4_CONFIG_PATH = ROOT / "high_precision_v4_config.json"
V5_CONFIG_PATH = ROOT / "high_precision_v5_config.json"
V2_FUNDAMENTALS_PATH = (
    ROOT.parent / "data" / "fundamentals" / "quarterly-point-in-time.pkl"
)
BENCHMARK_CACHE_PATH = Path(
    r"C:\InStockLocal\backtest_cache\indices\000300.csv"
)
DAILY_BARS_CACHE = Path(r"C:\InStockLocal\backtest_cache\daily_bars")
INSTOCK_HISTORY_CACHE = Path(r"C:\InStockLocal\InStock\instock\cache\hist")


def _fetch_sina_daily_bars(code: str, selection_date: pd.Timestamp) -> pd.DataFrame:
    normalized_code = str(code).zfill(6)
    if normalized_code.startswith(("4", "8", "92")):
        market_code = f"bj{normalized_code}"
    elif normalized_code.startswith(("5", "6", "9")):
        market_code = f"sh{normalized_code}"
    else:
        market_code = f"sz{normalized_code}"
    response = requests.get(
        "https://quotes.sina.cn/cn/api/jsonp_v2.php/"
        f"var%20_{market_code}=/CN_MarketDataService.getKLineData",
        params={
            "symbol": market_code,
            "scale": "240",
            "ma": "no",
            "datalen": "1023",
        },
        headers={"User-Agent": "Mozilla/5.0 InStockDesktop/2.0"},
        timeout=(3.05, 12.0),
    )
    response.raise_for_status()
    match = re.search(r"(\[\{.*\}\])", response.text, flags=re.DOTALL)
    if not match:
        return pd.DataFrame()
    records = json.loads(match.group(1))
    bars = pd.DataFrame(records).rename(columns={"day": "date"})
    required = ["date", "open", "high", "low", "close", "volume"]
    if bars.empty or not set(required).issubset(bars.columns):
        return pd.DataFrame()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
    for column in required[1:]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(subset=required)
    bars = bars[bars["date"] <= pd.Timestamp(selection_date).normalize()].copy()
    bars["amount"] = bars["close"] * bars["volume"]
    return bars.loc[:, [*required, "amount"]].sort_values("date").reset_index(drop=True)


def load_runtime_artifacts(
    model_path: Path, report_path: Path
) -> tuple[dict, dict]:
    try:
        model = json.loads(Path(model_path).read_text(encoding="utf-8"))
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("高胜率模型或回测证据不可用") from exc
    if not bool(model.get("deployment", {}).get("passed")):
        raise RuntimeError("高胜率模型尚未通过部署门槛")
    if not bool(report.get("deployment", {}).get("passed")):
        raise RuntimeError("最新高胜率回测尚未通过部署门槛")
    model_run = str(model.get("run_id", ""))
    report_run = str(report.get("run_id", ""))
    if not model_run or model_run != report_run:
        raise RuntimeError("高胜率模型与最新回测运行编号不一致")
    return model, report


def _finite(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def high_precision_composite_score(probability, shareholder_score) -> float:
    """Keep probability dominant while shareholder quality contributes 15%."""
    probability_value = max(0.0, min(1.0, _finite(probability, 0.0)))
    shareholder_value = max(-15.0, min(15.0, _finite(shareholder_score, 0.0)))
    shareholder_quality = (shareholder_value + 15.0) / 30.0
    return round(probability_value * 85.0 + shareholder_quality * 15.0, 2)


def _live_panel(snapshot: dict, historical_context: dict, config: dict) -> pd.DataFrame:
    rows = []
    for source in snapshot["rows"]:
        code = str(source.get("code") or "").zfill(6)
        context = historical_context.get(code)
        if context is None:
            continue
        item = {
            "date": pd.Timestamp(snapshot["selection_date"]),
            "code": code,
            "name": str(source.get("name") or ""),
            "industry": str(source.get("industry") or ""),
            "concept": str(source.get("concept") or ""),
            "open": _finite(source.get("new_price"), float("nan")),
            "close": _finite(source.get("new_price"), float("nan")),
            "breadth_above_ma20": context.get("breadth_above_ma20"),
            "market_median_mom20": context.get("market_median_mom20"),
            "market_median_drawdown60": context.get(
                "market_median_drawdown60"
            ),
        }
        for factor in config["stock_rank_features"]:
            item[factor] = context.get(factor)
        rows.append(item)
    return pd.DataFrame(rows)


def _contribution_text(
    feature_row: pd.Series, model: dict
) -> tuple[str, str]:
    contributions = []
    if str(model.get("schema_version", "")).startswith("high-precision-v5"):
        contributions = [
            ("relative_benchmark_model", 1.0),
            (str(feature_row.get("selection_score_name", "v5_score")), 0.8),
            ("balanced_risk_reward", 0.6),
        ]
    elif str(model.get("schema_version", "")).startswith("high-precision-v4"):
        contributions = [
            ("dynamic_market_regime", 1.0),
            (str(feature_row.get("selection_score_name", "regime_score")), 0.8),
            ("controlled_position", 0.6),
        ]
    elif str(model.get("schema_version", "")).startswith("high-precision-v3"):
        contributions = [
            ("强市回调", 1.0),
            ("长期趋势", 0.8),
            ("短期不过热", 0.6),
        ]
    elif str(model.get("schema_version", "")).startswith(
        "high-precision-v2"
    ):
        weights = model["ensemble"]["weights"]
        for submodel_name in ("trend", "reversal", "quality"):
            specification = model["submodels"][submodel_name]
            weight = float(weights[submodel_name])
            for name, coefficient in zip(
                specification["feature_names"],
                specification["model"]["coef"],
            ):
                value = _finite(feature_row.get(name), 0.5)
                contributions.append(
                    (name, value * float(coefficient) * weight)
                )
    else:
        names = list(model["feature_names"])
        coefficients = model["model"]["coef"]
        for name, coefficient in zip(names, coefficients):
            value = _finite(feature_row.get(name), 0.0)
            contributions.append((name, value * float(coefficient)))
    positive = sorted(
        (item for item in contributions if item[1] > 0),
        key=lambda item: (-item[1], item[0]),
    )[:3]
    negative = sorted(
        (item for item in contributions if item[1] < 0),
        key=lambda item: (item[1], item[0]),
    )[:3]
    return (
        "；".join(f"{name} {value:+.3f}" for name, value in positive),
        "；".join(f"{name} {value:+.3f}" for name, value in negative),
    )


def _display_row(
    source: dict,
    probability: float,
    regime: str,
    threshold: float,
    positive: str,
    negative: str,
    atr: float | None,
    gate_passed: bool = True,
) -> dict:
    price = _finite(source.get("new_price"), 0.0)
    fund_amount = _finite(source.get("fund_amount"), 0.0) / 10000.0
    shareholder_score, shareholder_reasons, shareholder_risks, shareholder_summary = (
        score_shareholder_structure(
            institutional_ratio=source.get("allcorp_ratio"),
            institutional_count=source.get("allcorp_num"),
            holder_count_change=source.get("holder_ratio"),
            top10_ratio=source.get("hold_ratio_count"),
            survey_count=source.get("org_survey_3m"),
        )
    )
    large_order_flow, retail_flow, order_flow_source = resolve_grouped_flows_wan(
        fund_amount=source.get("fund_amount"),
        fund_amount_super=source.get("fund_amount_super"),
        fund_amount_large=source.get("fund_amount_large"),
        fund_amount_medium=source.get("fund_amount_medium"),
        fund_amount_small=source.get("fund_amount_small"),
    )
    return {
        "代码": str(source.get("code") or "").zfill(6),
        "名称": str(source.get("name") or ""),
        "行业": str(source.get("industry") or ""),
        "最新价": round(price, 3),
        "涨跌幅%": round(_finite(source.get("change_rate"), 0.0), 3),
        "综合评分": high_precision_composite_score(probability, shareholder_score),
        "今日主力净流入_万元": round(fund_amount, 2),
        "3日主力净流入_万元": round(_finite(source.get("fund_amount_3"), 0.0) / 10000.0, 2),
        "5日主力净流入_万元": round(_finite(source.get("fund_amount_5"), 0.0) / 10000.0, 2),
        "今日大单净额_万元": large_order_flow,
        "今日散户净额_万元": retail_flow,
        "订单资金日期": clean_value(source.get("fund_component_date")),
        "订单资金来源": order_flow_source,
        "PE": source.get("pe9"),
        "ROE%": source.get("roe_weight"),
        "净利同比%": source.get("netprofit_yoy_ratio"),
        "股东结构评分": round(shareholder_score, 1),
        "机构持股比例%": source.get("allcorp_ratio"),
        "机构持股家数": source.get("allcorp_num"),
        "股东户数变化%": source.get("holder_ratio"),
        "十大股东持股比例%": source.get("hold_ratio_count"),
        "近3月机构调研数": source.get("org_survey_3m"),
        "股东结构说明": shareholder_summary,
        "加分理由": "；".join(
            [f"5日上涨概率达到{threshold:.1%}门槛", *shareholder_reasons]
        ),
        "风险提示": "；".join(
            ["概率模型仅反映历史样本外规律，不保证未来收益", *shareholder_risks]
        ),
        "概念": source.get("concept"),
        "ATR14": round(_finite(atr), 6) if atr is not None else None,
        "5日上涨概率": round(probability, 6),
        "市场状态": regime,
        "概率门槛": round(threshold, 6),
        "主要正向因子": positive,
        "主要负向因子": negative,
        "通过高胜率门槛": bool(gate_passed),
    }


def predict_v2_live(features: pd.DataFrame, model: dict) -> pd.DataFrame:
    output = features.copy()
    predictions = predict_submodels(model["submodels"], output)
    for name in ("trend", "reversal", "quality"):
        output[f"{name}_probability"] = predictions[name].to_numpy()
    raw = combine_probabilities(
        predictions, model["ensemble"]["weights"]
    )
    output["raw_probability"] = raw
    output["calibrated_probability"] = apply_probability_calibrator(
        raw, model["ensemble"]["calibration"]
    )
    return output


def predict_v3_live(features: pd.DataFrame, model: dict) -> pd.DataFrame:
    scored = build_pullback_scores(features)
    selected = select_pullback_candidates(scored, model["rule"]).copy()
    probability = float(model["historical_win_rate"])
    selected["raw_probability"] = probability
    selected["calibrated_probability"] = probability
    selected["market_regime"] = "strong"
    selected["probability_threshold"] = probability
    return selected


def predict_v4_live(features: pd.DataFrame, model: dict) -> pd.DataFrame:
    scored = build_regime_scores(features)
    selected = select_dynamic_candidates(scored, model["rule"]).copy()
    probability = float(model["historical_win_rate"])
    selected["raw_probability"] = probability
    selected["calibrated_probability"] = probability
    selected["probability_threshold"] = probability
    return selected


def predict_v5_live(features: pd.DataFrame, model: dict) -> pd.DataFrame:
    scored = score_v5_probabilities(features, model)
    selected = select_v5_candidates(scored, model["rule"]).copy()
    if selected.empty:
        selected["probability_threshold"] = pd.Series(dtype="float64")
    else:
        selected["probability_threshold"] = float(
            selected["calibrated_probability"].min()
        )
    return selected


def _load_live_v3_features(
    snapshot: dict, universe_records: list[dict] | None = None
) -> pd.DataFrame:
    selection_date = pd.Timestamp(snapshot["selection_date"]).normalize()
    live_by_code = {
        str(item.get("code") or "").zfill(6): item
        for item in snapshot["rows"]
    }
    rows = []
    for record in universe_records or snapshot["rows"]:
        code = str(record.get("code") or "").zfill(6)
        source = {**record, **live_by_code.get(code, {})}
        path = DAILY_BARS_CACHE / f"{code}.csv"
        try:
            if path.exists():
                bars = pd.read_csv(path)
            else:
                date_start = (
                    selection_date - pd.Timedelta(days=365 * 3)
                ).strftime("%Y%m%d")
                history_path = (
                    INSTOCK_HISTORY_CACHE
                    / date_start[:6]
                    / date_start
                    / f"{code}qfq.gzip.pickle"
                )
                if history_path.exists():
                    bars = pd.read_pickle(history_path, compression="gzip")
                else:
                    bars = _fetch_sina_daily_bars(code, selection_date)
                    if bars.empty:
                        continue
                    try:
                        DAILY_BARS_CACHE.mkdir(parents=True, exist_ok=True)
                        bars.to_csv(path, index=False)
                    except OSError:
                        pass
            bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
            bars = bars[bars["date"] <= selection_date].tail(130).copy()
            if bars.empty:
                continue
            if bars["date"].max() < selection_date:
                price = _finite(source.get("new_price"), float("nan"))
                volume = _finite(source.get("volume"), float("nan"))
                if math.isfinite(price) and math.isfinite(volume):
                    bars = pd.concat(
                        [
                            bars,
                            pd.DataFrame(
                                [
                                    {
                                        "date": selection_date,
                                        "open": _finite(source.get("pre_close_price"), price),
                                        "high": _finite(source.get("high_price"), price),
                                        "low": _finite(source.get("low_price"), price),
                                        "close": price,
                                        "volume": volume,
                                        "amount": _finite(source.get("deal_amount"), price * volume),
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
            bars = bars.sort_values("date").reset_index(drop=True)
            factors = compute_factors(bars)
            latest = factors.iloc[-1]
            latest_bar = bars.iloc[-1]
            if pd.Timestamp(latest_bar["date"]).normalize() != selection_date:
                continue
            rows.append(
                {
                    "date": selection_date,
                    "code": code,
                    "name": str(source.get("name") or ""),
                    "industry": str(source.get("industry") or ""),
                    "concept": str(source.get("concept") or ""),
                    "open": latest_bar["open"],
                    "high": latest_bar["high"],
                    "low": latest_bar["low"],
                    "close": latest_bar["close"],
                    **latest.to_dict(),
                }
            )
        except Exception:
            continue
    features = pd.DataFrame(rows)
    if features.empty:
        return features
    features["breadth_above_ma20"] = (
        pd.to_numeric(features["close"], errors="coerce")
        > pd.to_numeric(features["ma_20"], errors="coerce")
    ).mean()
    features["market_median_mom20"] = pd.to_numeric(
        features["mom_20"], errors="coerce"
    ).median()
    features["market_median_drawdown60"] = pd.to_numeric(
        features["drawdown_60"], errors="coerce"
    ).median()
    return features.reset_index(drop=True)


def _load_live_v2_features(snapshot: dict) -> pd.DataFrame:
    selection_date = pd.Timestamp(snapshot["selection_date"]).normalize()
    frames = []
    for source in snapshot["rows"]:
        code = str(source.get("code") or "").zfill(6)
        path = DAILY_BARS_CACHE / f"{code}.csv"
        if not path.exists():
            continue
        try:
            bars = pd.read_csv(path)
            bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
            bars = bars[bars["date"] <= selection_date].tail(130).copy()
            if bars.empty:
                continue
            if bars["date"].max() < selection_date:
                price = _finite(source.get("new_price"), float("nan"))
                volume = _finite(source.get("volume"), float("nan"))
                if math.isfinite(price) and math.isfinite(volume):
                    bars = pd.concat(
                        [
                            bars,
                            pd.DataFrame(
                                [
                                    {
                                        "date": selection_date,
                                        "open": _finite(
                                            source.get("pre_close_price"),
                                            price,
                                        ),
                                        "high": _finite(
                                            source.get("high_price"), price
                                        ),
                                        "low": _finite(
                                            source.get("low_price"), price
                                        ),
                                        "close": price,
                                        "volume": volume,
                                        "amount": _finite(
                                            source.get("deal_amount"),
                                            price * volume,
                                        ),
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
            bars["code"] = code
            bars["name"] = str(source.get("name") or "")
            bars["industry"] = str(source.get("industry") or "")
            bars["concept"] = str(source.get("concept") or "")
            frames.append(bars)
        except Exception:
            continue
    if not frames or not BENCHMARK_CACHE_PATH.exists():
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    benchmark = pd.read_csv(BENCHMARK_CACHE_PATH)
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="coerce"
    )
    benchmark = benchmark[benchmark["date"] <= selection_date].copy()
    fundamentals = (
        pd.read_pickle(V2_FUNDAMENTALS_PATH)
        if V2_FUNDAMENTALS_PATH.exists()
        else pd.DataFrame()
    )
    trading_dates = pd.DatetimeIndex(
        sorted(benchmark["date"].dropna().dt.normalize().unique())
    )
    features = build_v2_features(
        raw, benchmark, fundamentals, trading_dates
    )
    return features[
        features["date"].eq(selection_date)
    ].reset_index(drop=True)


def build_live_payload(
    snapshot: dict,
    historical_context: dict,
    config: dict,
    model: dict,
    profiles: dict,
    display_config: dict,
    v2_features: pd.DataFrame | None = None,
) -> dict:
    del display_config
    schema_version = str(model.get("schema_version", ""))
    is_v2 = schema_version.startswith(
        "high-precision-v2"
    )
    is_v3 = schema_version.startswith("high-precision-v3")
    is_v4 = schema_version.startswith("high-precision-v4")
    is_v5 = schema_version.startswith("high-precision-v5")
    panel = (
        v2_features.copy()
        if (is_v2 or is_v3 or is_v4 or is_v5) and v2_features is not None
        else _live_panel(snapshot, historical_context, config)
    )
    selection_date = snapshot["selection_date"]
    if panel.empty:
        regime = "unknown"
        threshold = None
        selected = panel
        features = panel
        abstain_reason = "历史因子数据不足，无法形成可靠概率"
    else:
        if is_v5:
            features = score_v5_probabilities(panel, model)
            selected = select_v5_candidates(features, model["rule"]).copy()
            regime = str(features.iloc[0]["market_regime"])
            threshold = float(selected["calibrated_probability"].min())
            selected["probability_threshold"] = threshold
            features["probability_threshold"] = threshold
            abstain_reason = ""
        elif is_v4:
            selected = predict_v4_live(panel, model)
            features = selected
            regime = str(selected.iloc[0]["market_regime"])
            threshold = float(model["historical_win_rate"])
            abstain_reason = ""
        elif is_v3:
            selected = predict_v3_live(panel, model)
            features = selected
            regime = "strong" if not selected.empty else "weak"
            threshold = float(model["historical_win_rate"])
            abstain_reason = (
                "当前不满足强市回调条件，高胜率模式保持空仓"
                if selected.empty
                else ""
            )
        elif is_v2:
            features = predict_v2_live(panel, model)
        else:
            features, _ = build_high_precision_features(
                panel,
                profiles,
                config,
                train_statistics=model["market_statistics"],
            )
            matrix = (
                features.loc[:, model["feature_names"]]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype="float64")
            )
            raw = predict_raw_probability(model["model"], matrix)
            features["raw_probability"] = raw
            features[
                "calibrated_probability"
            ] = apply_probability_calibrator(raw, model["calibration"])
        if not (is_v3 or is_v4 or is_v5):
            policy = HighPrecisionPolicy.from_dict(model["policy"])
            regime = classify_high_precision_regime(features.iloc[0], policy)
            threshold = (
                None
                if regime == "weak"
                else policy.strong_threshold
                + (policy.normal_threshold_delta if regime == "normal" else 0.0)
            )
            selected = select_daily_candidates(features, policy)
            if regime == "weak":
                abstain_reason = "市场处于弱势，高胜率模式保持空仓"
            elif selected.empty:
                abstain_reason = "当前候选的5日上涨概率未达到市场门槛"
            else:
                abstain_reason = ""

    source_by_code = {
        str(row.get("code") or "").zfill(6): row for row in snapshot["rows"]
    }
    display_rows = []
    probability_rows = []
    probability_features = features
    for _index, selected_row in selected.iterrows():
        code = str(selected_row["code"]).zfill(6)
        source = source_by_code.get(code) or {
            "code": code,
            "name": selected_row.get("name", ""),
            "industry": selected_row.get("industry", ""),
            "concept": selected_row.get("concept", ""),
            "new_price": selected_row.get("close"),
            "pre_close_price": selected_row.get("open"),
            "high_price": selected_row.get("high"),
            "low_price": selected_row.get("low"),
            "volume": selected_row.get("volume"),
            "deal_amount": selected_row.get("amount"),
            "change_rate": 0.0,
        }
        positive, negative = _contribution_text(selected_row, model)
        context = historical_context.get(code, {})
        display_rows.append(
            _display_row(
                source,
                float(selected_row["calibrated_probability"]),
                str(selected_row["market_regime"]),
                float(selected_row["probability_threshold"]),
                positive,
                negative,
                (
                    selected_row.get("atr_14")
                    if (is_v2 or is_v3 or is_v4 or is_v5)
                    else context.get("atr_14")
                ),
            )
        )
        display_rows[-1]["_selection_order"] = len(display_rows)
    display_rows.sort(
        key=lambda row: (-row["综合评分"], -row["5日上涨概率"], row["代码"])
    )
    for index, row in enumerate(display_rows, start=1):
        row.pop("_selection_order", None)
        row["排名"] = index

    if is_v5 and not probability_features.empty:
        selected_codes = {
            str(code).zfill(6) for code in selected.get("code", pd.Series(dtype=str))
        }
        for _index, feature_row in probability_features.iterrows():
            code = str(feature_row["code"]).zfill(6)
            source = source_by_code.get(code) or {
                "code": code,
                "name": feature_row.get("name", ""),
                "industry": feature_row.get("industry", ""),
                "concept": feature_row.get("concept", ""),
                "new_price": feature_row.get("close"),
                "change_rate": 0.0,
            }
            positive, negative = _contribution_text(feature_row, model)
            probability_rows.append(
                _display_row(
                    source,
                    float(feature_row["calibrated_probability"]),
                    str(feature_row["market_regime"]),
                    float(threshold),
                    positive,
                    negative,
                    feature_row.get("atr_14"),
                    gate_passed=code in selected_codes,
                )
            )
        probability_rows.sort(
            key=lambda row: (-row["综合评分"], -row["5日上涨概率"], row["代码"])
        )
        for index, row in enumerate(probability_rows, start=1):
            row["排名"] = index

    target_count = (
        int(selected.iloc[0]["target_count"])
        if (is_v4 or is_v5) and not selected.empty
        else len(display_rows)
    )

    return {
        "order_flow_schema": "grouped-order-flow-v2",
        "title": "高胜率精选",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_dates": {
            "综合选股": clean_value(selection_date),
            "每日行情": clean_value(snapshot.get("spot_date")),
            "资金流向": clean_value(snapshot.get("fund_date")),
        },
        "summary": {
            "全市场记录数": len(snapshot["rows"]),
            "高胜率有效因子数": len(panel),
            "输出Top数": len(display_rows),
            "market_regime": regime,
            "target_count": target_count,
            "probability_threshold": threshold,
            "active_count": len(display_rows),
            "abstain_reason": abstain_reason,
            "policy_version": (
                "high-precision-5d-v5"
                if is_v5
                else (
                    "high-precision-5d-v4"
                    if is_v4
                    else (
                        "high-precision-5d-v3"
                        if is_v3
                        else ("high-precision-5d-v2" if is_v2 else "high-precision-5d-v1")
                    )
                )
            ),
            "run_id": model.get("run_id"),
        },
        "strategy_gate_passed": True,
        "trade_policy": {
            "policy_version": (
                "high-precision-5d-v5"
                if is_v5
                else (
                    "high-precision-5d-v4"
                    if is_v4
                    else (
                        "high-precision-5d-v3"
                        if is_v3
                        else ("high-precision-5d-v2" if is_v2 else "high-precision-5d-v1")
                    )
                )
            ),
            "holding_days": 5,
            "atr_multiplier": 1.5,
            "reward_risk": 2.0,
            **(
                {
                    "fixed_take_profit_pct": float(model["trade_policy"]["take_profit_pct"]),
                    "fixed_stop_loss_pct": float(model["trade_policy"]["stop_loss_pct"]),
                    "position_fraction": float(model["trade_policy"]["position_fraction"]),
                }
                if (is_v3 or is_v4 or is_v5)
                else {}
            ),
        },
        "rules_note": [
            "仅在模型概率和市场门槛同时通过时选股。",
            "弱势市场或无达标股票时保持空仓。",
            "历史样本外表现不代表未来收益。",
        ],
        "top_sheet_name": "高胜率精选",
        "preview_sheet": "高胜率精选",
        "sheets": {
            "高胜率精选": display_rows,
            "全部高胜率候选": display_rows,
            "全部概率候选": probability_rows if is_v5 else display_rows,
        },
    }


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: high_precision_picker_data.py output.json "
            "config.json model.json [as_of_date] [requested_codes_csv]",
            file=sys.stderr,
        )
        return 2
    output_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    model_path = Path(sys.argv[3])
    as_of_date = None
    if len(sys.argv) >= 5:
        as_of_date = pd.Timestamp(sys.argv[4]).date()
    requested_probability_codes = (
        {
            code.strip().zfill(6)
            for code in sys.argv[5].split(",")
            if code.strip()
        }
        if len(sys.argv) >= 6
        else set()
    )
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    display_config = json.loads(
        DISPLAY_CONFIG_PATH.read_text(encoding="utf-8")
    )
    model, _report = load_runtime_artifacts(model_path, REPORT_PATH)
    schema_version = str(model.get("schema_version", ""))
    is_v2 = schema_version.startswith(
        "high-precision-v2"
    )
    is_v3 = schema_version.startswith("high-precision-v3")
    is_v4 = schema_version.startswith("high-precision-v4")
    is_v5 = schema_version.startswith("high-precision-v5")
    config = json.loads(
        (
            V5_CONFIG_PATH
            if is_v5
            else (
                V4_CONFIG_PATH
                if is_v4
                else (V3_CONFIG_PATH if is_v3 else (V2_CONFIG_PATH if is_v2 else config_path))
            )
        ).read_text(encoding="utf-8")
    )
    with connect() as connection:
        snapshot = load_selection_snapshot(connection.cursor(), as_of_date)
    if is_v2 or is_v3 or is_v4 or is_v5:
        historical_context = {}
        universe_records = model.get("universe") if is_v5 else None
        if is_v5 and requested_probability_codes:
            universe_records = list(universe_records or [])
            existing = {
                str(item.get("code") or "").zfill(6) for item in universe_records
            }
            live_by_code = {
                str(item.get("code") or "").zfill(6): item
                for item in snapshot["rows"]
            }
            for code in sorted(requested_probability_codes - existing):
                universe_records.append(live_by_code.get(code, {"code": code}))
        v2_features = (
            _load_live_v3_features(
                snapshot, universe_records
            )
            if (is_v3 or is_v4 or is_v5)
            else _load_live_v2_features(snapshot)
        )
    else:
        weights = {name: 1.0 for name in config["stock_rank_features"]}
        historical_context = load_historical_context(
            snapshot["rows"], weights
        )
        v2_features = None
    payload = build_live_payload(
        snapshot,
        historical_context,
        config,
        model,
        profiles,
        display_config,
        v2_features=v2_features,
    )
    if is_v5:
        available_codes = (
            set(v2_features["code"].astype(str).str.zfill(6))
            if v2_features is not None and not v2_features.empty
            else set()
        )
        payload["summary"]["requested_probability_codes"] = sorted(
            requested_probability_codes
        )
        payload["summary"]["unavailable_probability_codes"] = sorted(
            requested_probability_codes - available_codes
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    print(
        f"Selected {len(payload['sheets']['高胜率精选'])} rows "
        f"from {len(snapshot['rows'])} candidates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
