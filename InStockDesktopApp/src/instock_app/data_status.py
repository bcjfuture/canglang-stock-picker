from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

import pymysql


HEALTH_TABLES = {
    "screener": "cn_stock_selection",
    "quotes": "cn_stock_spot",
    "fund_flow": "cn_stock_fund_flow",
}
CORE_DATA_TABLES = (
    "cn_stock_spot",
    "cn_etf_spot",
    "cn_stock_fund_flow",
    "cn_stock_fund_flow_industry",
    "cn_stock_fund_flow_concept",
    "cn_stock_indicators",
)


def _default_connection():
    return pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root",
        database="instockdb",
        connect_timeout=2,
        read_timeout=2,
        write_timeout=2,
    )


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()[:10]
    text = str(value).strip()
    return text[:10] if text else None


def latest_table_dates(
    connection_factory: Callable[[], Any] | None = None,
) -> dict[str, str | None]:
    """Latest date of each table the strategies read."""
    factory = connection_factory or _default_connection
    result: dict[str, str | None] = {}
    with factory() as connection:
        with connection.cursor() as cursor:
            for key, table in HEALTH_TABLES.items():
                cursor.execute(f"SELECT MAX(`date`) FROM `{table}`")
                row = cursor.fetchone()
                result[key] = _normalize_date(row[0] if row else None)
    return result


def _previous_weekday(today: date) -> date:
    day = today - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def assess_data_health(dates: dict[str, str | None], today: date) -> dict[str, str]:
    """Judge the data itself: a running database can still hold stale tables."""
    quotes = dates.get("quotes")
    if not quotes:
        return {"label": "需检查", "detail": "没有每日行情数据", "tone": "danger"}
    lagging = [
        f"{label}停在{value or '无数据'}"
        for label, value in (("综合选股", dates.get("screener")), ("资金流", dates.get("fund_flow")))
        if not value or value < quotes
    ]
    if lagging:
        return {
            "label": "需检查",
            "detail": f"{'，'.join(lagging)}，行情已到{quotes}",
            "tone": "warning",
        }
    if quotes < _previous_weekday(today).isoformat():
        return {"label": "需更新", "detail": f"行情停在{quotes}（节假日可忽略）", "tone": "warning"}
    return {"label": "正常", "detail": f"数据已到{quotes}", "tone": "success"}


def latest_complete_data_date(
    connection_factory: Callable[[], Any] | None = None,
) -> str | None:
    selects = " UNION ALL ".join(
        f"SELECT MAX(`date`) AS latest_date FROM `{table}`"
        for table in CORE_DATA_TABLES
    )
    sql = (
        "SELECT CASE "
        f"WHEN COUNT(latest_date) = {len(CORE_DATA_TABLES)} "
        "THEN MIN(latest_date) ELSE NULL END "
        f"FROM ({selects}) AS core_dates"
    )
    factory = connection_factory or _default_connection
    with factory() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
    return _normalize_date(row[0] if row else None)
