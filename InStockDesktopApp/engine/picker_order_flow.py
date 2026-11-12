from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests


ORDER_FLOW_SCHEMA = "grouped-order-flow-v2"
EASTMONEY_H5_HISTORY_FLOW_URL = (
    "https://emdatah5.eastmoney.com/dc/ZJLX/getDBHistoryData"
)
EASTMONEY_DAILY_FLOW_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
)


@dataclass(frozen=True)
class OrderFlowComponents:
    code: str
    trade_date: str
    large_order_wan: float | None
    retail_order_wan: float | None
    source: str
    fetched_at: str


def _optional_wan(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number / 10000.0, 2)


def _optional_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _grouped_wan(first: object, second: object) -> float | None:
    first_number = _optional_number(first)
    second_number = _optional_number(second)
    if first_number is None or second_number is None:
        return None
    return round((first_number + second_number) / 10000.0, 2)


def parse_daily_flow_payload(
    code: str, trade_date: str, payload: dict
) -> OrderFlowComponents | None:
    data = payload.get("data")
    rows = data.get("klines") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    for raw_row in rows:
        fields = str(raw_row).split(",")
        if len(fields) < 6 or fields[0] != trade_date:
            continue
        # Eastmoney row: date, main, small, medium, large, super-large, ...
        large = _grouped_wan(fields[5], fields[4])
        retail = _grouped_wan(fields[3], fields[2])
        if large is None and retail is None:
            return None
        return OrderFlowComponents(
            code=str(code).zfill(6),
            trade_date=trade_date,
            large_order_wan=large,
            retail_order_wan=retail,
            source="实时分项",
            fetched_at=datetime.now().isoformat(timespec="seconds"),
        )
    return None


def read_order_flow_cache(
    path: Path, trade_date: str
) -> dict[str, OrderFlowComponents]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        document.get("schema") != ORDER_FLOW_SCHEMA
        or str(document.get("trade_date", "")) != trade_date
    ):
        return {}
    items = document.get("items")
    if not isinstance(items, dict):
        return {}
    result: dict[str, OrderFlowComponents] = {}
    for raw_code, item in items.items():
        if not isinstance(item, dict) or str(item.get("trade_date", "")) != trade_date:
            continue
        code = str(raw_code).zfill(6)
        large = _optional_number(item.get("large_order_wan"))
        retail = _optional_number(item.get("retail_order_wan"))
        if large is None and retail is None:
            continue
        result[code] = OrderFlowComponents(
            code=code,
            trade_date=trade_date,
            large_order_wan=large,
            retail_order_wan=retail,
            source="当日分项缓存",
            fetched_at=str(item.get("fetched_at", "")),
        )
    return result


def write_order_flow_cache(
    path: Path,
    trade_date: str,
    rows: dict[str, OrderFlowComponents],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": ORDER_FLOW_SCHEMA,
        "trade_date": trade_date,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "items": {
            str(code).zfill(6): asdict(row)
            for code, row in sorted(rows.items())
            if row.trade_date == trade_date
            and (row.large_order_wan is not None or row.retail_order_wan is not None)
        },
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _market_id(code: str) -> str:
    normalized = str(code).zfill(6)
    return "1" if normalized.startswith(("5", "6", "9")) else "0"


def fetch_order_flow_components(
    code: str,
    trade_date: str,
    timeout_seconds: float = 4.0,
) -> OrderFlowComponents | None:
    normalized = str(code).zfill(6)
    secid = f"{_market_id(normalized)}.{normalized}"
    shared_params = {
        "lmt": "0",
        "secid": secid,
        "ut": "b2884a393a59ad64002292a3e90d46a5",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }
    sources = (
        (
            EASTMONEY_H5_HISTORY_FLOW_URL,
            {
                "secid": secid,
                "fields1": "f1,f2,f3",
                "fields2": "f51,f52,f53,f54,f55,f56,f62,f63",
                "ut": shared_params["ut"],
            },
            "https://emdatah5.eastmoney.com/dc/zjlx/stock",
        ),
        (
            EASTMONEY_DAILY_FLOW_URL,
            {
                **shared_params,
                "klt": "101",
                "fields1": "f1,f2,f3,f7",
                "fields2": (
                    "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,"
                    "f63,f64,f65"
                ),
                "_": str(int(time.time() * 1000)),
            },
            "https://data.eastmoney.com/",
        ),
    )
    for url, params, referer in sources:
        try:
            response = requests.get(
                url,
                params=params,
                headers={**headers, "Referer": referer},
                timeout=(min(2.5, timeout_seconds), timeout_seconds),
            )
            response.raise_for_status()
            parsed = parse_daily_flow_payload(normalized, trade_date, response.json())
            if parsed is not None:
                return parsed
        except (requests.RequestException, ValueError):
            continue
    return None


def _database_components(
    row: dict, trade_date: str
) -> OrderFlowComponents | None:
    if (
        str(row.get("订单资金日期") or "") != trade_date
        or str(row.get("订单资金来源") or "")
        not in {"数据库分项", "数据库汇总推导"}
    ):
        return None
    large = _optional_number(row.get("今日大单净额_万元"))
    retail = _optional_number(row.get("今日散户净额_万元"))
    if large is None and retail is None:
        return None
    return OrderFlowComponents(
        code=str(row.get("代码") or "").zfill(6),
        trade_date=trade_date,
        large_order_wan=large,
        retail_order_wan=retail,
        source=str(row.get("订单资金来源")),
        fetched_at="",
    )


def _apply_components(row: dict, components: OrderFlowComponents | None) -> dict:
    enriched = dict(row)
    if components is None:
        enriched["今日大单净额_万元"] = None
        enriched["今日散户净额_万元"] = None
        enriched["订单资金日期"] = None
        enriched["订单资金来源"] = "分项暂缺"
        return enriched
    enriched["今日大单净额_万元"] = components.large_order_wan
    enriched["今日散户净额_万元"] = components.retail_order_wan
    enriched["订单资金日期"] = components.trade_date
    enriched["订单资金来源"] = components.source
    return enriched


def enrich_selected_rows(
    rows: list[dict],
    trade_date: str,
    *,
    cache: dict[str, OrderFlowComponents] | None = None,
    cache_path: Path | None = None,
    fetcher: Callable[[str, str], OrderFlowComponents | None] = fetch_order_flow_components,
    max_workers: int = 4,
) -> list[dict]:
    cached = dict(cache or {})
    if cache is None and cache_path is not None:
        cached = read_order_flow_cache(cache_path, trade_date)

    resolved: dict[int, OrderFlowComponents | None] = {}
    fetch_indexes: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        code = str(row.get("代码") or "").zfill(6)
        database = _database_components(row, trade_date)
        if database is not None:
            resolved[index] = database
        elif code in cached:
            resolved[index] = cached[code]
        elif len(code) == 6 and code.isdigit():
            fetch_indexes.append((index, code))
        else:
            resolved[index] = None

    if fetch_indexes:
        workers = max(1, min(max_workers, len(fetch_indexes)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetcher, code, trade_date): (index, code)
                for index, code in fetch_indexes
            }
            for future in as_completed(futures):
                index, code = futures[future]
                try:
                    components = future.result()
                except Exception:
                    components = None
                if components is not None and components.trade_date == trade_date:
                    resolved[index] = components
                    cached[code] = components
                else:
                    resolved[index] = None

    if cache_path is not None and any(
        item.source == "实时分项" for item in cached.values()
    ):
        write_order_flow_cache(cache_path, trade_date, cached)

    return [_apply_components(row, resolved.get(index)) for index, row in enumerate(rows)]
