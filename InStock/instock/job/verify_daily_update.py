#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import logging
import math
import re
import sys

import instock.core.tablestructure as tbs
import instock.lib.database as mdb
import instock.lib.trade_time as trd


CORE_TABLES = (
    (tbs.TABLE_CN_STOCK_SPOT["name"], "code"),
    (tbs.TABLE_CN_ETF_SPOT["name"], "code"),
    (tbs.TABLE_CN_STOCK_FUND_FLOW["name"], "code"),
    (tbs.TABLE_CN_STOCK_FUND_FLOW_INDUSTRY["name"], "name"),
    (tbs.TABLE_CN_STOCK_FUND_FLOW_CONCEPT["name"], "name"),
    (tbs.TABLE_CN_STOCK_INDICATORS["name"], "code"),
)


def _validate_identifier(identifier):
    if not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
        raise ValueError(f"unsafe SQL identifier: {identifier}")


def verify_table(cursor, table_name, date_value, key_column):
    _validate_identifier(table_name)
    _validate_identifier(key_column)
    cursor.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT `{key_column}`),
            COALESCE(SUM(CASE WHEN `{key_column}` IS NULL THEN 1 ELSE 0 END), 0)
        FROM `{table_name}`
        WHERE `date` = %s
        """,
        (date_value,),
    )
    rows, unique_keys, null_keys = cursor.fetchone()
    result = {
        "table": table_name,
        "date": date_value,
        "rows": int(rows),
        "unique_keys": int(unique_keys),
        "null_keys": int(null_keys),
    }
    if result["rows"] <= 0:
        raise RuntimeError(f"{table_name} has no rows for {date_value}")
    if result["rows"] != result["unique_keys"]:
        raise RuntimeError(
            f"{table_name} contains duplicate keys for {date_value}: "
            f"rows={result['rows']} unique={result['unique_keys']}"
        )
    if result["null_keys"]:
        raise RuntimeError(
            f"{table_name} contains {result['null_keys']} null keys for {date_value}"
        )
    return result


def verify_stock_coverage(results, minimum_ratio=0.85, minimum_rows=1000):
    stock_rows = int(results[tbs.TABLE_CN_STOCK_SPOT["name"]]["rows"])
    fund_rows = int(results[tbs.TABLE_CN_STOCK_FUND_FLOW["name"]]["rows"])
    required = max(int(minimum_rows), int(fund_rows * float(minimum_ratio)))
    if stock_rows < required:
        ratio = stock_rows / fund_rows if fund_rows else 0.0
        raise RuntimeError(
            "cn_stock_spot coverage is incomplete: "
            f"stock_rows={stock_rows} fund_rows={fund_rows} "
            f"coverage={ratio:.1%} required={minimum_ratio:.1%}"
        )


def verify_indicator_coverage(results, minimum_ratio=0.95):
    stock_rows = int(results[tbs.TABLE_CN_STOCK_SPOT["name"]]["rows"])
    indicator_rows = int(results[tbs.TABLE_CN_STOCK_INDICATORS["name"]]["rows"])
    required = math.ceil(stock_rows * float(minimum_ratio))
    if indicator_rows < required:
        ratio = indicator_rows / stock_rows if stock_rows else 0.0
        raise RuntimeError(
            "cn_stock_indicators indicator coverage is incomplete: "
            f"indicator_rows={indicator_rows} stock_rows={stock_rows} "
            f"coverage={ratio:.1%} required={minimum_ratio:.1%}"
        )


def main():
    _, latest_trade_date = trd.get_trade_date_last()
    date_value = latest_trade_date.strftime("%Y-%m-%d")
    with mdb.get_connection() as connection:
        with connection.cursor() as cursor:
            results = {}
            for table_name, key_column in CORE_TABLES:
                result = verify_table(cursor, table_name, date_value, key_column)
                results[table_name] = result
                print(
                    f"{table_name}: date={date_value} rows={result['rows']} "
                    f"unique={result['unique_keys']} null_keys={result['null_keys']}"
                )
            verify_stock_coverage(results)
            verify_indicator_coverage(results)
    print("DAILY_UPDATE_HEALTH=PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        logging.error("DAILY_UPDATE_HEALTH_FAILED %s", error)
        sys.exit(1)
