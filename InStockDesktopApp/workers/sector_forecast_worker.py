from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from instock_app.settings import AppPaths
from workers.common import write_json_atomic
from workers.sector_data import SectorDataBundle, SectorDataUnavailable, load_sector_bundle
from workers.sector_forecast_core import (
    build_feature_panel,
    rank_latest,
    walk_forward_validate,
)


MODEL_VERSION = "sector-short-momentum-v5"
Emitter = Callable[[str, str, int, dict | None], None]
BundleLoader = Callable[..., SectorDataBundle]
Validator = Callable[[pd.DataFrame], dict[str, Any]]


def _merge_current_signals(panel: pd.DataFrame, bundle: SectorDataBundle) -> pd.DataFrame:
    result = panel.copy()
    current = bundle.current_signals.drop_duplicates("sector_id").set_index("sector_id")
    live_mask = pd.to_datetime(result["date"]).dt.normalize() == bundle.as_of
    signal_columns = (
        "fund_score",
        "breadth_score",
        "heat_score",
        "up_ratio",
        "current_data_completeness",
    )
    live_ids = result.loc[live_mask, "sector_id"]
    for column in signal_columns:
        if column in current:
            result.loc[live_mask, column] = live_ids.map(current[column])
    counts = result.groupby("sector_id")["date"].nunique()
    result.loc[live_mask, "history_days"] = live_ids.map(counts)
    return result


def _unavailable(message: str, quality: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "status": "unavailable",
        "message": message,
        "quality": quality or {},
        "validation": {},
        "rows": [],
    }


def _evaluation_path(paths: AppPaths) -> Path:
    return Path(paths.data_dir) / "sector_forecast" / "evaluations.json"


def _load_evaluations(paths: AppPaths) -> dict[str, Any]:
    path = _evaluation_path(paths)
    if not path.exists():
        return {"runs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data.get("runs"), dict) else {"runs": {}}
    except (OSError, json.JSONDecodeError):
        return {"runs": {}}


def _read_price_return(path: Path, as_of: pd.Timestamp) -> tuple[str, float] | None:
    try:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
        positions = frame.index[frame["date"] == as_of].tolist()
        if not positions:
            return None
        position = frame.index.get_loc(positions[-1])
        if position + 5 >= len(frame):
            return None
        start = float(frame.iloc[position]["close"])
        end = float(frame.iloc[position + 5]["close"])
        return frame.iloc[position + 5]["date"].date().isoformat(), end / start - 1.0
    except (OSError, ValueError, KeyError, IndexError):
        return None


def _benchmark_return(as_of: pd.Timestamp) -> tuple[str, float] | None:
    return _read_price_return(Path(r"C:\InStockLocal\backtest_cache\indices\000300.csv"), as_of)


def _update_evaluations(paths: AppPaths) -> dict[str, Any]:
    evaluations = _load_evaluations(paths)
    runs: dict[str, Any] = evaluations.setdefault("runs", {})
    history_root = Path(paths.data_dir) / "sector_forecast" / "history"
    if not history_root.exists():
        return evaluations
    changed = False
    for snapshot_path in history_root.glob("*.json"):
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            as_of = pd.Timestamp(snapshot["as_of"]).normalize()
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue
        run_key = f"{snapshot.get('as_of')}|{snapshot.get('model_version')}"
        if run_key in runs:
            continue
        benchmark = _benchmark_return(as_of)
        if benchmark is None:
            continue
        exit_date, benchmark_value = benchmark
        run_result: dict[str, Any] = {"as_of": snapshot["as_of"], "exit_date": exit_date, "universes": {}}
        complete = True
        for sector_type, universe in snapshot.get("universes", {}).items():
            cache_root = Path(paths.data_dir) / "sector_forecast" / "cache" / sector_type
            all_returns: dict[str, float] = {}
            for cache_path in cache_root.glob("*.csv"):
                realized = _read_price_return(cache_path, as_of)
                if realized is not None and realized[0] == exit_date:
                    all_returns[cache_path.stem] = realized[1] - benchmark_value
            actual_top = {
                sector_id
                for sector_id, _value in sorted(all_returns.items(), key=lambda item: item[1], reverse=True)[:10]
            }
            details = []
            for row in universe.get("rows", []):
                sector_id = str(row.get("sector_id", ""))
                if sector_id not in all_returns:
                    complete = False
                    break
                details.append(
                    {
                        "sector_id": sector_id,
                        "sector_name": row.get("sector_name"),
                        "excess_return": all_returns[sector_id],
                        "actual_top10": sector_id in actual_top,
                    }
                )
            if not complete:
                break
            run_result["universes"][sector_type] = details
        if complete:
            runs[run_key] = run_result
            changed = True
    if changed:
        evaluations["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_json_atomic(_evaluation_path(paths), evaluations)
    return evaluations


def _attach_previous_realized(result: dict[str, Any], evaluations: dict[str, Any]) -> None:
    candidates = sorted(
        evaluations.get("runs", {}).values(),
        key=lambda item: str(item.get("as_of", "")),
        reverse=True,
    )
    for sector_type, universe in result["universes"].items():
        previous: dict[str, float] = {}
        for run in candidates:
            details = run.get("universes", {}).get(sector_type, [])
            if details:
                previous = {
                    str(item.get("sector_id")): float(item.get("excess_return", 0.0))
                    for item in details
                }
                break
        for row in universe.get("rows", []):
            row["previous_realized_excess"] = previous.get(str(row.get("sector_id")))


def run_sector_forecast(
    paths: AppPaths,
    emit: Emitter,
    refresh_history: bool = False,
    *,
    bundle_loader: BundleLoader = load_sector_bundle,
    validator: Validator = walk_forward_validate,
    ranker: Callable[[pd.DataFrame, dict[str, Any], int], list[dict[str, Any]]] | None = None,
) -> dict[str, object]:
    ranker = ranker or (lambda panel, validation, top_k: rank_latest(panel, validation, top_k))
    universes: dict[str, object] = {}
    as_of_dates: list[pd.Timestamp] = []
    steps = {"industry": (5, 35), "concept": (45, 75)}
    labels = {"industry": "行业", "concept": "概念"}

    for sector_type in ("industry", "concept"):
        load_progress, model_progress = steps[sector_type]
        label = labels[sector_type]
        emit("progress", f"正在加载{label}板块历史与资金信号", load_progress, {"sector_type": sector_type})
        try:
            bundle = bundle_loader(paths, sector_type, refresh=refresh_history)
            as_of_dates.append(bundle.as_of)
            if not bool(bundle.quality.get("runnable", True)):
                universes[sector_type] = _unavailable(
                    f"{label}板块数据质量门禁未通过", bundle.quality
                )
                continue
            emit("progress", f"正在滚动验证{label}板块模型", model_progress, {"sector_type": sector_type})
            panel = build_feature_panel(bundle.history)
            panel = _merge_current_signals(panel, bundle)
            validation_panel = panel
            if "derived_latest" in bundle.history:
                historical_only = bundle.history.loc[
                    ~bundle.history["derived_latest"].fillna(False).astype(bool)
                ].copy()
                validation_panel = build_feature_panel(historical_only)
                validation_panel = _merge_current_signals(validation_panel, bundle)
            validation = validator(validation_panel)
            rows = ranker(panel, validation, 10)
            if not bool(validation.get("passed")) and bool(validation.get("head_passed")):
                status = "head_ready" if rows else "insufficient_confidence"
                message = (
                    f"{label}板块前5名已通过样本外验证，其余候选仅供参考"
                    if rows
                    else f"{label}板块没有达到最低数据可信度的候选"
                )
            elif not bool(validation.get("passed")):
                status = "forced_fallback" if rows else "insufficient_confidence"
                message = (
                    f"{label}板块参考榜单已生成（模型验证未通过）"
                    if rows
                    else f"{label}板块没有达到最低数据可信度的候选"
                )
            else:
                status = "ready" if rows else "insufficient_confidence"
                message = (
                    f"{label}板块预测已生成"
                    if rows
                    else f"{label}板块没有达到可信度门槛的候选"
                )
            universes[sector_type] = {
                "status": status,
                "message": message,
                "quality": bundle.quality,
                "validation": validation,
                "rows": rows,
                "as_of": bundle.as_of.date().isoformat(),
            }
        except SectorDataUnavailable as exc:
            universes[sector_type] = _unavailable(str(exc))
        except Exception as exc:
            universes[sector_type] = _unavailable(f"{label}板块预测失败：{exc}")

    as_of = max(as_of_dates).date().isoformat() if as_of_dates else datetime.now().date().isoformat()
    result: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": as_of,
        "horizon_days": 5,
        "benchmark": "000300",
        "model_version": MODEL_VERSION,
        "signal_horizons": {
            "ranking_days": [3, 5, 10],
            "factor_weights": [0.45, 0.35, 0.20],
            "risk_days": [20],
        },
        "universes": universes,
        "disclaimer": "历史样本外结果不代表未来收益，预测仅供研究参考，不构成投资建议。",
    }
    emit("progress", "正在更新历史预测的真实表现", 90, None)
    evaluations = _update_evaluations(paths)
    _attach_previous_realized(result, evaluations)

    snapshot_path = (
        Path(paths.data_dir)
        / "sector_forecast"
        / "history"
        / f"{as_of}-{MODEL_VERSION}.json"
    )
    has_meaningful_run = any(
        universe.get("status") != "unavailable"
        for universe in result["universes"].values()
    )
    if has_meaningful_run and not snapshot_path.exists():
        write_json_atomic(snapshot_path, result)
    emit(
        "result",
        "未来一周热门板块预测完成",
        100,
        {
            "sector_forecast": True,
            "as_of": result["as_of"],
            "statuses": {
                key: value.get("status") for key, value in result["universes"].items()
            },
        },
    )
    return result
