from __future__ import annotations

import json
import math
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from instock_app.data_status import _default_connection, latest_complete_data_date
from instock_app.settings import AppPaths, STRATEGIES
from workers.backtest_worker import fetch_index_bars
from workers.common import write_json_atomic
from workers.picker_worker import HEAD_RERANK_VERSION, run_picker


ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from picker_backtest import direct_network_environment, fetch_daily_bars


Emitter = Callable[[str, str, int, dict | None], None]
SelectionDateLoader = Callable[[str], list[str]]
RECENT_OFFSETS = (1, 3, 5)
ROUND_TRIP_COST = 0.003
DAILY_BAR_CACHE = Path(r"C:\InStockLocal\backtest_cache\daily_bars")
INDEX_CACHE = Path(r"C:\InStockLocal\backtest_cache\indices")


def select_signal_dates(
    trading_dates,
    evaluation_date: str,
    offsets: tuple[int, ...] = RECENT_OFFSETS,
) -> dict[int, str]:
    evaluation = pd.Timestamp(evaluation_date).normalize()
    normalized = sorted(
        {
            pd.Timestamp(value).normalize()
            for value in trading_dates
            if pd.notna(value) and pd.Timestamp(value).normalize() <= evaluation
        }
    )
    if evaluation not in normalized:
        raise ValueError(f"评估日 {evaluation.date()} 不在交易日序列中")
    position = normalized.index(evaluation)
    result: dict[int, str] = {}
    for offset in offsets:
        target = position - int(offset)
        if target < 0:
            raise ValueError(f"交易日历史不足，无法取得前 {offset} 个交易日")
        result[int(offset)] = normalized[target].date().isoformat()
    return result


def _normalized_dates(values, end_date: str | None = None) -> list[pd.Timestamp]:
    end = pd.Timestamp(end_date).normalize() if end_date else None
    return sorted(
        {
            value
            for item in values
            if pd.notna(item)
            for value in [pd.Timestamp(item).normalize()]
            if end is None or value <= end
        }
    )


def latest_trading_date_on_or_before(
    trading_dates, requested_date: str
) -> str:
    candidates = _normalized_dates(trading_dates, requested_date)
    if not candidates:
        raise ValueError(f"{requested_date} 及之前没有可用交易日")
    return candidates[-1].date().isoformat()


def align_signal_dates(
    trading_dates,
    evaluation_date: str,
    requested_signal_dates: dict[int, str],
    snapshot_dates,
) -> dict[int, dict[str, object]]:
    calendar = _normalized_dates(trading_dates, evaluation_date)
    evaluation = pd.Timestamp(evaluation_date).normalize()
    position_by_date = {value: index for index, value in enumerate(calendar)}
    if evaluation not in position_by_date:
        raise ValueError(f"评估日 {evaluation.date()} 不在交易日序列中")
    available = [
        value
        for value in _normalized_dates(snapshot_dates, evaluation_date)
        if value in position_by_date
    ]
    result: dict[int, dict[str, object]] = {}
    for offset, requested_text in requested_signal_dates.items():
        requested = pd.Timestamp(requested_text).normalize()
        eligible = [value for value in available if value <= requested]
        actual = eligible[-1] if eligible else None
        result[int(offset)] = {
            "requested_signal_date": requested.date().isoformat(),
            "signal_date": (
                actual.date().isoformat() if actual is not None else None
            ),
            "requested_holding_trading_days": int(offset),
            "holding_trading_days": (
                position_by_date[evaluation] - position_by_date[actual]
                if actual is not None
                else None
            ),
            "snapshot_fallback": actual is not None and actual != requested,
        }
    return result


def _normalized_bars(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["open"] = pd.to_numeric(data["open"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    return (
        data.dropna(subset=["date", "open", "close"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


def _bucket_summary(
    details: list[dict[str, object]],
    minimum_rank: int,
    maximum_rank: int,
    *,
    rank_field: str = "rank",
) -> dict[str, object]:
    selected = [
        detail
        for detail in details
        if minimum_rank <= int(detail.get(rank_field) or 0) <= maximum_rank
    ]
    returns = [
        float(detail["net_return"])
        for detail in selected
        if detail.get("status") == "ok" and detail.get("net_return") is not None
    ]
    positive = sum(value > 0 for value in returns)
    return {
        "selected_count": len(selected),
        "evaluated_count": len(returns),
        "unavailable_count": len(selected) - len(returns),
        "positive_count": positive,
        "mean_net_return": float(statistics.fmean(returns)) if returns else None,
        "median_net_return": float(statistics.median(returns)) if returns else None,
        "win_rate": positive / len(returns) if returns else None,
    }


def evaluate_snapshot(
    rows: list[dict],
    bars_by_code: dict[str, pd.DataFrame],
    signal_date: str,
    evaluation_date: str,
    cost: float = ROUND_TRIP_COST,
) -> dict[str, object]:
    if not isinstance(rows, list):
        raise ValueError("历史选股结果 rows 必须是列表")
    signal = pd.Timestamp(signal_date).normalize()
    evaluation = pd.Timestamp(evaluation_date).normalize()
    details: list[dict[str, object]] = []
    evaluated_returns: list[float] = []
    for index, row in enumerate(rows, start=1):
        code = str(row.get("code", "")).zfill(6)
        original_rank_value = row.get("original_rank")
        if original_rank_value is None:
            original_rank_value = row.get("rank") or index
        detail: dict[str, object] = {
            "rank": int(row.get("rank") or index),
            "original_rank": int(original_rank_value),
            "code": code,
            "name": str(row.get("name", "")),
            "entry_date": None,
            "exit_date": evaluation.date().isoformat(),
            "entry_price": None,
            "exit_price": None,
            "gross_return": None,
            "net_return": None,
            "status": "unavailable",
            "status_text": "缺少有效行情",
        }
        frame = bars_by_code.get(code)
        if frame is None or frame.empty:
            details.append(detail)
            continue
        data = _normalized_bars(frame)
        entries = data[(data["date"] > signal) & (data["date"] <= evaluation)]
        exits = data[data["date"] == evaluation]
        if entries.empty:
            detail["status_text"] = "选股后无可用开盘价"
            details.append(detail)
            continue
        if exits.empty:
            detail["status_text"] = "评估日无收盘价"
            details.append(detail)
            continue
        entry_row = entries.iloc[0]
        exit_row = exits.iloc[-1]
        entry_price = float(entry_row["open"])
        exit_price = float(exit_row["close"])
        if not (
            math.isfinite(entry_price)
            and entry_price > 0
            and math.isfinite(exit_price)
            and exit_price > 0
        ):
            detail["status_text"] = "价格无效"
            details.append(detail)
            continue
        gross_return = exit_price / entry_price - 1.0
        net_return = gross_return - float(cost)
        detail.update(
            {
                "entry_date": entry_row["date"].date().isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "status": "ok",
                "status_text": "已评估",
            }
        )
        evaluated_returns.append(net_return)
        details.append(detail)

    evaluated_count = len(evaluated_returns)
    positive_count = sum(value > 0 for value in evaluated_returns)
    negative_count = sum(value < 0 for value in evaluated_returns)
    flat_count = evaluated_count - positive_count - negative_count
    summary = {
        "selected_count": len(rows),
        "evaluated_count": evaluated_count,
        "unavailable_count": len(rows) - evaluated_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "flat_count": flat_count,
        "mean_net_return": (
            float(statistics.fmean(evaluated_returns)) if evaluated_returns else None
        ),
        "median_net_return": (
            float(statistics.median(evaluated_returns)) if evaluated_returns else None
        ),
        "win_rate": positive_count / evaluated_count if evaluated_count else None,
        "buckets": {
            "top5": _bucket_summary(details, 1, 5),
            "rank6_20": _bucket_summary(details, 6, 20),
            "all20": _bucket_summary(details, 1, 20),
            "baseline_top5": _bucket_summary(
                details, 1, 5, rank_field="original_rank"
            ),
        },
    }
    return {"summary": summary, "details": details}


def evaluate_profiles(
    snapshots: dict[str, dict],
    bars_by_code: dict[str, pd.DataFrame],
    signal_date: str,
    evaluation_date: str,
    benchmark_return: float,
    cost: float = ROUND_TRIP_COST,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for strategy_key, snapshot in snapshots.items():
        try:
            evaluated = evaluate_snapshot(
                snapshot.get("rows"),
                bars_by_code,
                signal_date,
                evaluation_date,
                cost,
            )
            mean_return = evaluated["summary"]["mean_net_return"]
            evaluated["summary"]["benchmark_return"] = benchmark_return
            evaluated["summary"]["excess_return"] = (
                float(mean_return) - benchmark_return
                if mean_return is not None
                else None
            )
            for bucket in evaluated["summary"].get("buckets", {}).values():
                bucket_mean = bucket.get("mean_net_return")
                bucket["benchmark_return"] = benchmark_return
                bucket["excess_return"] = (
                    float(bucket_mean) - benchmark_return
                    if bucket_mean is not None
                    else None
                )
            buckets = evaluated["summary"].get("buckets", {})
            top5 = buckets.get("top5", {})
            baseline_top5 = buckets.get("baseline_top5", {})
            top5_mean = top5.get("mean_net_return")
            baseline_mean = baseline_top5.get("mean_net_return")
            top5_win = top5.get("win_rate")
            baseline_win = baseline_top5.get("win_rate")
            evaluated["summary"]["head_rerank_gate_passed"] = bool(
                int(top5.get("evaluated_count", 0) or 0) >= 5
                and int(baseline_top5.get("evaluated_count", 0) or 0) >= 5
                and top5_mean is not None
                and baseline_mean is not None
                and top5_win is not None
                and baseline_win is not None
                and float(top5_mean) > float(baseline_mean)
                and float(top5_win) > float(baseline_win)
            )
            results[strategy_key] = {
                "status": "ok",
                "title": STRATEGIES.get(strategy_key).title
                if strategy_key in STRATEGIES
                else strategy_key,
                "source": snapshot.get("_source", "unknown"),
                **evaluated,
            }
        except Exception as exc:
            results[strategy_key] = {
                "status": "error",
                "title": STRATEGIES.get(strategy_key).title
                if strategy_key in STRATEGIES
                else strategy_key,
                "error": str(exc),
                "summary": {},
                "details": [],
            }
    return results


def find_stored_snapshot(
    history_root: Path,
    strategy_key: str,
    signal_date: str,
    required_run_id: str | None = None,
    required_head_rerank_version: str | None = None,
    required_selection_version: str | None = None,
) -> dict | None:
    strategy_dir = Path(history_root) / strategy_key
    if not strategy_dir.exists():
        return None
    matches: list[dict] = []
    for path in strategy_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("trade_date")) == signal_date:
            if required_selection_version and payload.get("source_summary", {}).get("selection_version") != required_selection_version:
                continue
            if required_run_id and str(
                payload.get("source_summary", {}).get("run_id", "")
            ) != required_run_id:
                continue
            if required_head_rerank_version and str(
                payload.get("source_summary", {}).get("head_rerank_version", "")
            ) != required_head_rerank_version:
                continue
            payload["_source"] = "stored_history"
            matches.append(payload)
    if not matches:
        return None
    return max(matches, key=lambda item: str(item.get("generated_at", "")))


def _load_cached_snapshot(
    path: Path,
    signal_date: str,
    required_run_id: str | None = None,
    required_head_rerank_version: str | None = None,
    required_selection_version: str | None = None,
) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(payload.get("trade_date")) != signal_date:
        return None
    if required_selection_version and payload.get("source_summary", {}).get("selection_version") != required_selection_version:
        return None
    if required_run_id and str(
        payload.get("source_summary", {}).get("run_id", "")
    ) != required_run_id:
        return None
    if required_head_rerank_version and str(
        payload.get("source_summary", {}).get("head_rerank_version", "")
    ) != required_head_rerank_version:
        return None
    payload["_source"] = "rebuild_cache"
    return payload


def _rebuild_snapshot(
    paths: AppPaths,
    strategy_key: str,
    signal_date: str,
    cache_path: Path,
) -> dict:
    sandbox_data = cache_path.parent / "_work" / strategy_key
    sandbox_paths = replace(paths, data_dir=sandbox_data)
    result = run_picker(
        strategy_key,
        sandbox_paths,
        lambda *_args: None,
        as_of_date=signal_date,
    )
    if str(result.get("trade_date")) != signal_date:
        raise RuntimeError(
            f"{strategy_key} 在 {signal_date} 没有精确选股快照，实际为 {result.get('trade_date')}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cache_path, result)
    result["_source"] = "rebuilt"
    return result


def _resolve_snapshot(
    paths: AppPaths,
    strategy_key: str,
    signal_date: str,
) -> dict:
    required_run_id = None
    required_selection_version = None
    if strategy_key in ("all_market", "tech_attack", "rebound_trend"):
        config_path = paths.engine_dir / STRATEGIES[strategy_key].config_name
        if config_path.exists():
            required_selection_version = json.loads(config_path.read_text(encoding="utf-8")).get("selection_version")
    required_head_rerank_version = (
        HEAD_RERANK_VERSION if strategy_key != "high_precision" else None
    )
    if strategy_key == "high_precision":
        model_path = paths.engine_dir / "high_precision_model.json"
        try:
            required_run_id = str(
                json.loads(model_path.read_text(encoding="utf-8")).get("run_id", "")
            ) or None
        except (OSError, json.JSONDecodeError):
            required_run_id = None
    stored = find_stored_snapshot(
        paths.data_dir / "store" / "history",
        strategy_key,
        signal_date,
        required_run_id=required_run_id,
        required_head_rerank_version=required_head_rerank_version,
        required_selection_version=required_selection_version,
    )
    if stored is not None:
        return stored
    cache_path = (
        paths.data_dir
        / "recent_backtest"
        / "snapshots"
        / signal_date
        / f"{strategy_key}.json"
    )
    cached = _load_cached_snapshot(
        cache_path,
        signal_date,
        required_run_id,
        required_head_rerank_version,
        required_selection_version,
    )
    if cached is not None:
        return cached
    return _rebuild_snapshot(paths, strategy_key, signal_date, cache_path)


def _load_bars_for_codes(
    codes: set[str],
    evaluation_date: str,
    *,
    workers: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    loaded: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    def load(code: str) -> pd.DataFrame:
        return fetch_daily_bars(
            code,
            "2022-01-01",
            evaluation_date,
            DAILY_BAR_CACHE,
        )

    with direct_network_environment():
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            future_map = {pool.submit(load, code): code for code in sorted(codes)}
            for future in as_completed(future_map):
                code = future_map[future]
                try:
                    loaded[code] = future.result()
                except Exception as exc:
                    errors[code] = str(exc)
    return loaded, errors


def _benchmark_return(
    index_bars: pd.DataFrame,
    signal_date: str,
    evaluation_date: str,
    cost: float,
) -> float:
    evaluated = evaluate_snapshot(
        [{"rank": 1, "code": "000300", "name": "沪深300"}],
        {"000300": index_bars},
        signal_date,
        evaluation_date,
        cost,
    )
    value = evaluated["summary"]["mean_net_return"]
    if value is None:
        raise RuntimeError("沪深300在目标区间缺少有效行情")
    return float(value)


def load_selection_dates(
    evaluation_date: str, connection_factory=None
) -> list[str]:
    factory = connection_factory or _default_connection
    with factory() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT `date` FROM `cn_stock_selection` "
                "WHERE `date` <= %s ORDER BY `date`",
                (evaluation_date,),
            )
            rows = cursor.fetchall()
    return [
        pd.Timestamp(row[0]).date().isoformat()
        for row in rows
        if row and row[0]
    ]


def run_recent_backtest(
    paths: AppPaths,
    emit: Emitter,
    *,
    offsets: tuple[int, ...] = RECENT_OFFSETS,
    cost: float = ROUND_TRIP_COST,
    selection_date_loader: SelectionDateLoader = load_selection_dates,
) -> dict[str, object]:
    requested_evaluation_date = latest_complete_data_date()
    if not requested_evaluation_date:
        raise RuntimeError("无法确定最新完整交易日")
    index_start = (
        pd.Timestamp(requested_evaluation_date) - pd.Timedelta(days=400)
    ).date().isoformat()
    emit(
        "progress",
        "正在读取交易日序列",
        5,
        {"evaluation_date": requested_evaluation_date},
    )
    hs300 = fetch_index_bars(
        "000300", index_start, requested_evaluation_date, INDEX_CACHE
    )
    evaluation_date = latest_trading_date_on_or_before(
        hs300["date"], requested_evaluation_date
    )
    warnings: list[str] = []
    if evaluation_date != requested_evaluation_date:
        warnings.append(
            f"数据库最新完整日为 {requested_evaluation_date}，"
            f"指数交易日仅到 {evaluation_date}，本次按 {evaluation_date} 评估"
        )
    requested_signal_dates = select_signal_dates(
        hs300["date"], evaluation_date, offsets=offsets
    )
    aligned_dates = {
        int(offset): {
            "requested_signal_date": signal_date,
            "signal_date": signal_date,
            "requested_holding_trading_days": int(offset),
            "holding_trading_days": int(offset),
            "snapshot_fallback": False,
        }
        for offset, signal_date in requested_signal_dates.items()
    }

    snapshots_by_offset: dict[int, dict[str, dict]] = {}
    for position, (offset, date_info) in enumerate(aligned_dates.items(), start=1):
        signal_date = date_info["signal_date"]
        profiles: dict[str, dict] = {}
        if signal_date is None:
            message = (
                f"{date_info['requested_signal_date']} 及之前没有可用选股快照"
            )
            warnings.append(message)
            for strategy_key in STRATEGIES:
                profiles[strategy_key] = {
                    "rows": None,
                    "_source": "error",
                    "_error": message,
                }
        else:
            if date_info["snapshot_fallback"]:
                warnings.append(
                    f"目标信号日 {date_info['requested_signal_date']} "
                    f"缺少精确快照，使用 {signal_date}"
                )
            for strategy_key in STRATEGIES:
                try:
                    profiles[strategy_key] = _resolve_snapshot(
                        paths, strategy_key, str(signal_date)
                    )
                except Exception as exc:
                    warnings.append(f"{signal_date} {strategy_key}: {exc}")
                    profiles[strategy_key] = {
                        "rows": None,
                        "_source": "error",
                        "_error": str(exc),
                    }
        snapshots_by_offset[offset] = profiles
        emit(
            "progress",
            f"已准备前{offset}个交易日选股快照",
            10 + position * 12,
            {
                "offset": offset,
                "requested_signal_date": date_info["requested_signal_date"],
                "signal_date": signal_date,
            },
        )

    codes: set[str] = set()
    for profiles in snapshots_by_offset.values():
        for snapshot in profiles.values():
            for row in snapshot.get("rows") or []:
                code = str(row.get("code", "")).zfill(6)
                if len(code) == 6 and code.isdigit():
                    codes.add(code)
    emit("progress", f"正在加载 {len(codes)} 只股票行情", 50, {"codes": len(codes)})
    bars_by_code, bar_errors = _load_bars_for_codes(codes, evaluation_date)
    warnings.extend(f"{code}: {message}" for code, message in bar_errors.items())

    output_offsets: dict[str, dict] = {}
    for index, (offset, date_info) in enumerate(aligned_dates.items(), start=1):
        signal_date = date_info["signal_date"]
        if signal_date is None:
            benchmark = None
            profiles = {}
            for strategy_key, snapshot in snapshots_by_offset[offset].items():
                profiles[strategy_key] = {
                    "status": "error",
                    "title": (
                        STRATEGIES[strategy_key].title
                        if strategy_key in STRATEGIES
                        else strategy_key
                    ),
                    "error": snapshot.get("_error", "没有可用选股快照"),
                    "summary": {},
                    "details": [],
                }
        else:
            benchmark = _benchmark_return(
                hs300, str(signal_date), evaluation_date, cost
            )
            profiles = evaluate_profiles(
                snapshots_by_offset[offset],
                bars_by_code,
                str(signal_date),
                evaluation_date,
                benchmark,
                cost,
            )
        for strategy_key, snapshot in snapshots_by_offset[offset].items():
            if snapshot.get("_error"):
                profiles[strategy_key]["error"] = snapshot["_error"]
        actual_holding = date_info["holding_trading_days"]
        label = f"前{offset}个交易日"
        if date_info["snapshot_fallback"]:
            label = f"前{actual_holding}个交易日（目标前{offset}日）"
        output_offsets[str(offset)] = {
            "label": label,
            **date_info,
            "evaluation_date": evaluation_date,
            "benchmark_return": benchmark,
            "profiles": profiles,
        }
        emit(
            "progress",
            f"已完成前{offset}个交易日回测",
            65 + index * 10,
            {"offset": offset},
        )

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "requested_evaluation_date": requested_evaluation_date,
        "evaluation_date": evaluation_date,
        "cost": cost,
        "offsets": output_offsets,
        "warnings": warnings,
        "disclaimer": "历史表现不代表未来收益，不构成投资建议。",
    }
    emit(
        "result",
        "近期选股回测完成",
        100,
        {
            "recent_backtest": True,
            "evaluation_date": evaluation_date,
            "offsets": list(output_offsets),
        },
    )
    return result
