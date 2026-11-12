#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fallback daily stock history sourced from Tencent Finance."""

import datetime
import time

import pandas as pd
import requests


HISTORY_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "quote_change",
    "ups_downs",
    "turnover",
]

TENCENT_HISTORY_URLS = (
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
)


def _date_text(value):
    text = value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def _symbol(code):
    return ("sh" if str(code).startswith(("5", "6", "9")) else "sz") + str(code)


def _append_latest_spot(history, latest_spot):
    if latest_spot is None:
        return history
    row = {
        "date": _date_text(latest_spot["date"]),
        "open": float(latest_spot["open_price"]),
        "close": float(latest_spot["new_price"]),
        "high": float(latest_spot["high_price"]),
        "low": float(latest_spot["low_price"]),
        "volume": float(latest_spot["volume"]) / 100.0,
        "amount": float(latest_spot["deal_amount"]),
        "amplitude": float(latest_spot["amplitude"]),
        "quote_change": float(latest_spot["change_rate"]),
        "ups_downs": float(latest_spot["ups_downs"]),
        "turnover": float(latest_spot["turnoverrate"]),
    }
    result = pd.concat([history, pd.DataFrame([row], columns=HISTORY_COLUMNS)], ignore_index=True)
    result.drop_duplicates(subset="date", keep="last", inplace=True)
    result.sort_values("date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def fetch_tencent_stock_history(
    code,
    start_date,
    end_date=None,
    adjust="qfq",
    latest_spot=None,
    request_get=None,
    sleep_fn=time.sleep,
):
    symbol = _symbol(code)
    start = _date_text(start_date)
    end = _date_text(end_date or datetime.date.today())
    if request_get is None:
        session = requests.Session()
        session.trust_env = False
        request_get = session.get
    source_rows = None
    last_error = None
    for attempt in range(2):
        for url in TENCENT_HISTORY_URLS:
            try:
                response = request_get(
                    url,
                    params={"param": f"{symbol},day,{start},{end},1000,{adjust}"},
                    headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
                    timeout=15,
                )
                response.raise_for_status()
                payload = response.json()
                symbol_data = (payload.get("data") or {}).get(symbol) or {}
                source_rows = symbol_data.get(f"{adjust}day") or symbol_data.get("day") or []
                if source_rows:
                    break
                last_error = RuntimeError(f"Tencent returned no history rows for {code}")
            except Exception as error:
                last_error = error
        if source_rows:
            break
        if attempt == 0:
            sleep_fn(2.0)
    if not source_rows:
        raise last_error or RuntimeError(f"Tencent returned no history rows for {code}")

    history = pd.DataFrame(
        [row[:6] for row in source_rows],
        columns=["date", "open", "close", "high", "low", "volume"],
    )
    for column in ("open", "close", "high", "low", "volume"):
        history[column] = pd.to_numeric(history[column], errors="raise")
    previous_close = history["close"].shift(1)
    history["amount"] = history["close"] * history["volume"] * 100.0
    history["amplitude"] = ((history["high"] - history["low"]) / previous_close * 100.0).fillna(0.0)
    history["quote_change"] = ((history["close"] - previous_close) / previous_close * 100.0).fillna(0.0)
    history["ups_downs"] = (history["close"] - previous_close).fillna(0.0)
    history["turnover"] = 0.0
    history = history[HISTORY_COLUMNS]
    return _append_latest_spot(history, latest_spot)
