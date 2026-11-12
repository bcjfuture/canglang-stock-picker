#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Independent Tonghuashun fallback for Eastmoney fund-flow rankings."""

import math
import re
import time

import pandas as pd
import requests


def _money(value, numeric_unit=1.0):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value) * numeric_unit
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--"}:
        return None
    multiplier = 1.0
    if text.endswith("亿"):
        multiplier, text = 100_000_000.0, text[:-1]
    elif text.endswith("万"):
        multiplier, text = 10_000.0, text[:-1]
    elif text.endswith("元"):
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def _percent(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _safe_rate(net, inflow, outflow):
    total = (inflow or 0.0) + (outflow or 0.0)
    return net / total * 100 if net is not None and total else None


def _individual_columns(index):
    suffix = ("", "_3", "_5", "_10")[index]
    return [
        "code", "name", "new_price", "change_rate" + suffix,
        "fund_amount" + suffix, "fund_rate" + suffix,
        "fund_amount_super" + suffix, "fund_rate_super" + suffix,
        "fund_amount_large" + suffix, "fund_rate_large" + suffix,
        "fund_amount_medium" + suffix, "fund_rate_medium" + suffix,
        "fund_amount_small" + suffix, "fund_rate_small" + suffix,
    ]


def build_individual_rank(raw, index):
    columns = _individual_columns(index)
    suffix = ("", "_3", "_5", "_10")[index]
    rows = []
    for _, source in raw.iterrows():
        code = re.sub(r"\.0$", "", str(source.get("股票代码", ""))).zfill(6)
        if len(code) != 6:
            continue
        if index == 0:
            net = _money(source.get("净额"))
            inflow = _money(source.get("流入资金"))
            outflow = _money(source.get("流出资金"))
            turnover = _money(source.get("成交额"))
            change = _percent(source.get("涨跌幅"))
            rate = net / turnover * 100 if net is not None and turnover else None
        else:
            net = _money(source.get("资金流入净额"))
            change = _percent(source.get("阶段涨跌幅"))
            rate = None
        row = {column: None for column in columns}
        row.update({
            "code": code,
            "name": source.get("股票简称"),
            "new_price": _percent(source.get("最新价")),
            "change_rate" + suffix: change,
            "fund_amount" + suffix: net,
            "fund_rate" + suffix: rate,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _sector_columns(index):
    suffix = ("", "_5", "_10")[index]
    return [
        "name", "change_rate" + suffix,
        "fund_amount" + suffix, "fund_rate" + suffix,
        "fund_amount_super" + suffix, "fund_rate_super" + suffix,
        "fund_amount_large" + suffix, "fund_rate_large" + suffix,
        "fund_amount_medium" + suffix, "fund_rate_medium" + suffix,
        "fund_amount_small" + suffix, "fund_rate_small" + suffix,
        "stock_name" + suffix,
    ]


def build_sector_rank(raw, index):
    columns = _sector_columns(index)
    suffix = ("", "_5", "_10")[index]
    rows = []
    for _, source in raw.iterrows():
        net = _money(source.get("净额"), numeric_unit=100_000_000.0)
        inflow = _money(source.get("流入资金"), numeric_unit=100_000_000.0)
        outflow = _money(source.get("流出资金"), numeric_unit=100_000_000.0)
        change_key = "行业-涨跌幅" if index == 0 else "阶段涨跌幅"
        row = {column: None for column in columns}
        row.update({
            "name": source.get("行业"),
            "change_rate" + suffix: _percent(source.get(change_key)),
            "fund_amount" + suffix: net,
            "fund_rate" + suffix: _safe_rate(net, inflow, outflow),
            "stock_name" + suffix: source.get("领涨股") if index == 0 else None,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def fetch_individual_rank(index):
    import akshare as ak

    symbol = ("即时", "3日排行", "5日排行", "10日排行")[index]
    return build_individual_rank(ak.stock_fund_flow_individual(symbol=symbol), index)


def fetch_sector_rank(index_sector, index_indicator):
    import akshare as ak

    symbol = ("即时", "5日排行", "10日排行")[index_indicator]
    loader = ak.stock_fund_flow_industry if index_sector == 0 else ak.stock_fund_flow_concept
    return build_sector_rank(loader(symbol=symbol), index_indicator)


def _fetch_xuangu_fund_flow_rank(index):
    suffix = ("", "_3", "_5", "_10")[index]
    net_field = ("NET_INFLOW", "NETINFLOW_3DAYS", "NETINFLOW_5DAYS", None)[index]
    columns = _individual_columns(index)
    market_filter = (
        '(MARKET in ("'
        + "\u4e0a\u4ea4\u6240\u4e3b\u677f"
        + '","'
        + "\u6df1\u4ea4\u6240\u4e3b\u677f"
        + '","'
        + "\u6df1\u4ea4\u6240\u521b\u4e1a\u677f"
        + '"))(NEW_PRICE>0)'
    )
    fields = [
        "SECURITY_CODE",
        "SECURITY_NAME_ABBR",
        "NEW_PRICE",
        "CHANGE_RATE",
        "NET_INFLOW",
        "NETINFLOW_3DAYS",
        "NETINFLOW_5DAYS",
        "SECUCODE",
        "MAX_TRADE_DATE",
    ]
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://data.eastmoney.com/",
        }
    )
    rows = []
    page = 1
    while True:
        response = session.get(
            "https://data.eastmoney.com/dataapi/xuangu/list",
            params={
                "sty": ",".join(fields),
                "filter": market_filter,
                "p": page,
                "ps": 100,
                "source": "SELECT_SECURITIES",
                "client": "WEB",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"Eastmoney xuangu returned error: {payload}")
        result = payload.get("result") or {}
        data = result.get("data") or []
        for source in data:
            row = {column: None for column in columns}
            row.update(
                {
                    "code": str(source.get("SECURITY_CODE", "")).zfill(6),
                    "name": source.get("SECURITY_NAME_ABBR"),
                    "new_price": source.get("NEW_PRICE"),
                    "change_rate" + suffix: source.get("CHANGE_RATE"),
                    "fund_amount" + suffix: source.get(net_field) if net_field else None,
                }
            )
            rows.append(row)
        if not result.get("nextpage") or not data:
            break
        page += 1
        time.sleep(0.2)
    return pd.DataFrame(rows, columns=columns)


def fetch_individual_rank(index):
    try:
        data = _fetch_xuangu_fund_flow_rank(index)
        if data is not None and len(data.index) > 0:
            return data
    except Exception as error:
        raise RuntimeError(f"Eastmoney xuangu fund-flow fetch failed: {error}") from error
