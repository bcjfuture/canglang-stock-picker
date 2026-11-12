#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fallback real-time quotes used when Eastmoney's market-list API is down."""

import time

import pandas as pd
import requests


def quote_coverage_is_sufficient(rows, universe_rows, minimum_ratio=0.85, minimum_rows=1000):
    required = max(int(minimum_rows), int(int(universe_rows) * float(minimum_ratio)))
    return int(rows) >= required


def merge_quote_universe(previous, universe, columns):
    """Add missing universe codes without weakening rows already in the snapshot."""
    columns = [column for column in columns if column != "date"]
    base = previous.reindex(columns=columns).copy()
    if base.empty:
        base = pd.DataFrame(columns=columns)
    if "code" not in base or universe is None or universe.empty:
        return base.reset_index(drop=True)

    base["code"] = base["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    known = set(base["code"])
    additions = []
    for _, source in universe.iterrows():
        code = str(source.get("code", "")).removesuffix(".0").zfill(6)
        if len(code) != 6 or not code.isdigit() or code in known:
            continue
        row = {column: None for column in columns}
        for column in columns:
            if column in source.index:
                row[column] = source.get(column)
        row["code"] = code
        additions.append(row)
        known.add(code)
    if additions:
        added = pd.DataFrame(additions, columns=columns).astype(object)
        base = pd.concat([base.astype(object), added], ignore_index=True)
    return base.reset_index(drop=True)


def load_latest_stock_universe(columns):
    """Load a broad current stock list used to reseed a degraded quote snapshot."""
    from instock.lib import database as mdb

    available = [column for column in ("code", "name", "new_price") if column in columns]
    selected = ",".join("`{}`".format(column) for column in available)
    sql = (
        "SELECT {} FROM `cn_stock_fund_flow` "
        "WHERE `date`=(SELECT MAX(`date`) FROM `cn_stock_fund_flow`)"
    ).format(selected)
    return pd.read_sql(sql, mdb.engine())


def _number(value, multiplier=1.0):
    try:
        if value in (None, "", "-"):
            return None
        return float(value) * multiplier
    except (TypeError, ValueError):
        return None


def _symbol(code):
    return ("sh" if str(code).startswith(("5", "6", "9")) else "sz") + str(code)


def _parse_quote_text(text):
    quotes = {}
    for statement in text.replace("\r", "").replace("\n", "").split(";"):
        if "=" not in statement:
            continue
        payload = statement.split("=", 1)[1].strip().strip('"')
        fields = payload.split("~")
        if len(fields) < 58 or not fields[2]:
            continue
        quotes[fields[2]] = fields
    return quotes


def fetch_tencent_quote_text(symbols, retries=3):
    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://gu.qq.com/",
    }
    last_error = None
    for attempt in range(retries):
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            response.encoding = "gb18030"
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.0 * (2 ** attempt))
    raise last_error


def refresh_quotes(previous, quote_fetcher=fetch_tencent_quote_text, batch_size=60):
    """Refresh quote fields while retaining fundamentals from the last snapshot."""
    if previous is None or previous.empty:
        raise RuntimeError("No previous market snapshot is available for fallback quotes.")

    quotes = {}
    codes = previous["code"].astype(str).str.zfill(6).tolist()
    for start in range(0, len(codes), batch_size):
        batch = codes[start:start + batch_size]
        quotes.update(_parse_quote_text(quote_fetcher([_symbol(code) for code in batch])))

    rows = []
    for _, old_row in previous.iterrows():
        code = str(old_row["code"]).zfill(6)
        fields = quotes.get(code)
        if not fields or _number(fields[3]) is None or _number(fields[3]) <= 0:
            continue
        row = old_row.copy()
        price_volume_amount = fields[35].split("/")
        updates = {
            "code": code,
            "name": fields[1],
            "new_price": _number(fields[3]),
            "pre_close_price": _number(fields[4]),
            "open_price": _number(fields[5]),
            "volume": _number(price_volume_amount[1], 100) if len(price_volume_amount) > 1 else None,
            "deal_amount": _number(price_volume_amount[2]) if len(price_volume_amount) > 2 else None,
            "ups_downs": _number(fields[31]),
            "change_rate": _number(fields[32]),
            "high_price": _number(fields[33]),
            "low_price": _number(fields[34]),
            "turnoverrate": _number(fields[38]),
            "amplitude": _number(fields[43]),
            "free_cap": _number(fields[44], 100000000),
            "total_market_cap": _number(fields[45], 100000000),
            "pbnewmrq": _number(fields[46]),
            "volume_ratio": _number(fields[49]),
            "dtsyl": _number(fields[39]),
            "pe9": _number(fields[53]),
        }
        if len(fields) > 73:
            updates["free_shares"] = _number(fields[72])
            updates["total_shares"] = _number(fields[73])
        for column, value in updates.items():
            if column in row.index and value is not None:
                row[column] = value
        rows.append(row)

    coverage = len(rows) / len(previous.index)
    if coverage < 0.85:
        raise RuntimeError(
            "Fallback quote coverage is too low: {}/{} ({:.1%}).".format(
                len(rows), len(previous.index), coverage
            )
        )
    return pd.DataFrame(rows, columns=previous.columns).reset_index(drop=True)


def load_latest_snapshot(table_name, columns):
    from instock.lib import database as mdb

    selected = ",".join("`{}`".format(column) for column in columns if column != "date")
    sql = (
        "SELECT {} FROM `{}` WHERE `date`=(SELECT MAX(`date`) FROM `{}`)"
        .format(selected, table_name, table_name)
    )
    return pd.read_sql(sql, mdb.engine())
