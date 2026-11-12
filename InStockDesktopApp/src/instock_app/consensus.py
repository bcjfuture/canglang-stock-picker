from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


def _snapshot_sort_key(payload: dict, path: Path) -> tuple[float, str]:
    generated_at = str(payload.get("generated_at") or "").strip()
    try:
        timestamp = datetime.fromisoformat(generated_at).timestamp()
    except (ValueError, OSError):
        timestamp = path.stat().st_mtime
    return timestamp, path.name


def _trade_date(value: object) -> str | None:
    try:
        parsed = date.fromisoformat(str(value or "").strip())
    except ValueError:
        return None
    if parsed > date.today():
        return None
    return parsed.isoformat()


def _stock_code(value: object) -> str | None:
    text = str(value or "").strip()
    if not text.isdigit() or len(text) > 6:
        return None
    return text.zfill(6)


def build_consensus(
    history_root: Path,
    strategy_keys: Iterable[str],
    *,
    day_limit: int,
    result_limit: int = 20,
) -> dict:
    strategies = tuple(strategy_keys)
    latest_by_strategy_day: dict[tuple[str, str], tuple[tuple[float, str], dict]] = {}
    skipped_snapshot_count = 0

    for strategy in strategies:
        directory = Path(history_root) / strategy
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                skipped_snapshot_count += 1
                continue
            if not isinstance(payload, dict):
                skipped_snapshot_count += 1
                continue
            trade_date = _trade_date(payload.get("trade_date"))
            rows = payload.get("rows")
            if not trade_date or not isinstance(rows, list):
                skipped_snapshot_count += 1
                continue
            key = strategy, trade_date
            sort_key = _snapshot_sort_key(payload, path)
            current = latest_by_strategy_day.get(key)
            if current is None or sort_key > current[0]:
                latest_by_strategy_day[key] = (sort_key, payload)

    selected_dates = sorted(
        {trade_date for _, trade_date in latest_by_strategy_day}, reverse=True
    )[:day_limit]
    trade_dates = sorted(selected_dates)
    selected_date_set = set(selected_dates)
    appearances: dict[str, list[dict]] = defaultdict(list)

    for (strategy, trade_date), (_, payload) in latest_by_strategy_day.items():
        if trade_date not in selected_date_set:
            continue
        best_by_code: dict[str, dict] = {}
        for index, source_row in enumerate(payload.get("rows", []), start=1):
            if not isinstance(source_row, dict):
                continue
            code = _stock_code(source_row.get("code"))
            if code is None:
                continue
            try:
                rank = int(source_row.get("rank") or index)
            except (TypeError, ValueError):
                rank = index
            row = {
                "code": code,
                "name": str(source_row.get("name") or "—"),
                "rank": max(1, rank),
            }
            current = best_by_code.get(code)
            if current is None or row["rank"] < current["rank"]:
                best_by_code[code] = row
        for row in best_by_code.values():
            appearances[row["code"]].append(
                {
                    **row,
                    "strategy": strategy,
                    "trade_date": trade_date,
                }
            )

    rows: list[dict] = []
    strategy_order = {key: index for index, key in enumerate(strategies)}
    for code, hits in appearances.items():
        dates = {hit["trade_date"] for hit in hits}
        hit_strategies = {hit["strategy"] for hit in hits}
        latest_hit = max(hits, key=lambda hit: hit["trade_date"])
        daily_counts = [
            sum(hit["trade_date"] == trade_date for hit in hits)
            for trade_date in trade_dates
        ]
        details = []
        for trade_date in reversed(trade_dates):
            day_hits = [hit for hit in hits if hit["trade_date"] == trade_date]
            if not day_hits:
                continue
            day_hits.sort(key=lambda hit: (strategy_order.get(hit["strategy"], 999), hit["rank"]))
            details.append(
                {
                    "trade_date": trade_date,
                    "hits": [
                        {"strategy": hit["strategy"], "rank": hit["rank"]}
                        for hit in day_hits
                    ],
                }
            )
        rows.append(
            {
                "code": code,
                "name": latest_hit["name"],
                "appearance_count": len(hits),
                "day_count": len(dates),
                "strategy_count": len(hit_strategies),
                "average_rank": round(
                    sum(hit["rank"] for hit in hits) / len(hits), 2
                ),
                "daily_counts": daily_counts,
                "last_seen": max(dates),
                "details": details,
            }
        )

    rows.sort(
        key=lambda row: (
            -row["appearance_count"],
            -row["day_count"],
            row["average_rank"],
            -row["strategy_count"],
            row["code"],
        )
    )
    rows = rows[:result_limit]
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": trade_dates[-1] if trade_dates else "",
        "date_start": trade_dates[0] if trade_dates else "",
        "trade_dates": trade_dates,
        "covered_days": len(trade_dates),
        "day_limit": day_limit,
        "strategy_count": len(strategies),
        "available_strategy_days": sum(
            trade_date in selected_date_set
            for _, trade_date in latest_by_strategy_day
        ),
        "maximum_appearances": day_limit * len(strategies),
        "skipped_snapshot_count": skipped_snapshot_count,
        "rows": rows,
    }


def build_five_day_consensus(
    history_root: Path,
    strategy_keys: Iterable[str],
    *,
    day_limit: int = 5,
    result_limit: int = 20,
) -> dict:
    return build_consensus(
        history_root,
        strategy_keys,
        day_limit=day_limit,
        result_limit=result_limit,
    )
