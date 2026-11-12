from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from instock_app.settings import AppPaths
from workers.backtest_worker import fetch_index_bars


class SectorDataUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SectorDataBundle:
    history: pd.DataFrame
    current_signals: pd.DataFrame
    as_of: pd.Timestamp
    quality: dict[str, object]


class SectorAPI(Protocol):
    def list_boards(self, sector_type: str) -> pd.DataFrame: ...

    def fetch_history(
        self, sector_type: str, name: str, start: str, end: str
    ) -> pd.DataFrame: ...


class AkshareSectorAPI:
    def list_boards(self, sector_type: str) -> pd.DataFrame:
        import requests

        if sector_type not in {"industry", "concept"}:
            raise ValueError(f"不支持的板块类型：{sector_type}")
        session = requests.Session()
        session.trust_env = False
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://quote.eastmoney.com/",
        }
        board_filter = "m:90 t:2 f:!50" if sector_type == "industry" else "m:90 t:3 f:!50"
        rows: list[dict[str, object]] = []
        for page in range(1, 10):
            response = session.get(
                "https://push2.eastmoney.com/api/qt/clist/get",
                params={
                    "pn": str(page),
                    "pz": "100",
                    "po": "1",
                    "np": "1",
                    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                    "fltt": "2",
                    "invt": "2",
                    "fid": "f3",
                    "fs": board_filter,
                    "fields": "f12,f14",
                },
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            data = (response.json().get("data") or {})
            page_rows = list(data.get("diff") or [])
            rows.extend(page_rows)
            if not page_rows or len(rows) >= int(data.get("total") or 0):
                break
        result = pd.DataFrame(
            {
                "sector_id": [item.get("f12") for item in rows],
                "sector_name": [item.get("f14") for item in rows],
            }
        )
        if result.empty:
            raise SectorDataUnavailable("东方财富板块列表返回空数据")
        return result.dropna().drop_duplicates("sector_id")

    def fetch_history(
        self, sector_type: str, name: str, start: str, end: str
    ) -> pd.DataFrame:
        import requests

        if sector_type not in {"industry", "concept"}:
            raise ValueError(f"不支持的板块类型：{sector_type}")
        session = requests.Session()
        session.trust_env = False
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://quote.eastmoney.com/",
        }
        params = {
            "secid": f"90.{name}",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "0",
            "beg": start,
            "end": end,
            "smplmt": "10000",
            "lmt": "1000000",
        }
        last_error: Exception | None = None
        host_numbers = ["5", "7", "28", "33", "58", "72", "1", "2", "17", "35", "46", "63", "79", "80", "82", "84", "91", "92"]
        offset = sum(ord(character) for character in name) % len(host_numbers)
        host_numbers = host_numbers[offset:] + host_numbers[:offset]
        for number in host_numbers[:3]:
            host = f"{number}.push2his.eastmoney.com"
            try:
                response = session.get(
                    f"https://{host}/api/qt/stock/kline/get",
                    params=params,
                    headers=headers,
                    timeout=5,
                )
                response.raise_for_status()
                lines = list(((response.json().get("data") or {}).get("klines") or []))
                if not lines:
                    raise SectorDataUnavailable(f"{name} 历史行情为空")
                values = [line.split(",") for line in lines]
                return pd.DataFrame(
                    values,
                    columns=[
                        "日期",
                        "开盘",
                        "收盘",
                        "最高",
                        "最低",
                        "成交量",
                        "成交额",
                        "振幅",
                        "涨跌幅",
                        "涨跌额",
                        "换手率",
                    ],
                )
            except Exception as exc:
                last_error = exc
        raise SectorDataUnavailable(f"{name} 历史行情获取失败：{last_error}")


@contextmanager
def _proxy_free_environment():
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is not None:
                os.environ[key] = value


def normalize_sector_history(
    raw: pd.DataFrame,
    sector_type: str,
    sector_id: str,
    sector_name: str,
) -> pd.DataFrame:
    aliases = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }
    data = raw.rename(columns=aliases).copy()
    required = ("date", "open", "close", "high", "low", "volume", "amount", "turnover")
    missing = [column for column in required if column not in data]
    if missing:
        raise SectorDataUnavailable(f"{sector_name} 历史行情缺少字段：{','.join(missing)}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in required[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "close"])
    data = data[data["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    data.insert(0, "sector_name", str(sector_name))
    data.insert(0, "sector_id", str(sector_id))
    data.insert(0, "sector_type", str(sector_type))
    columns = [
        "sector_type",
        "sector_id",
        "sector_name",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
    ]
    return data[columns].reset_index(drop=True)


def _safe_float(value: object) -> float:
    try:
        if isinstance(value, (bytes, bytearray)):
            return float(int.from_bytes(value, "little"))
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _normalized_name(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def build_breadth_signals(selection: pd.DataFrame, sector_type: str) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame(
            columns=[
                "sector_name",
                "up_ratio",
                "new_high_ratio",
                "long_ma_ratio",
                "heat_score_raw",
                "member_count",
            ]
        )
    data = selection.copy()
    if sector_type == "industry":
        data["sector_name"] = data.get("industry", "").map(_normalized_name)
    elif sector_type == "concept":
        data["sector_name"] = data.get("concept", "").fillna("").astype(str).str.split(r"[,，;；|]")
        data = data.explode("sector_name")
        data["sector_name"] = data["sector_name"].map(_normalized_name)
    else:
        raise ValueError(f"不支持的板块类型：{sector_type}")
    data = data[data["sector_name"] != ""].copy()
    data["is_up"] = pd.to_numeric(data.get("change_rate"), errors="coerce").gt(0).astype(float)
    data["is_high"] = data.get("now_newhigh", 0).map(_safe_float).fillna(0).gt(0).astype(float)
    data["is_long"] = data.get("long_avg_array", 0).map(_safe_float).fillna(0).gt(0).astype(float)
    data["popularity"] = pd.to_numeric(data.get("popularity_rank"), errors="coerce")
    result = (
        data.groupby("sector_name", as_index=False)
        .agg(
            up_ratio=("is_up", "mean"),
            new_high_ratio=("is_high", "mean"),
            long_ma_ratio=("is_long", "mean"),
            popularity_median=("popularity", "median"),
            member_count=("sector_name", "size"),
        )
    )
    result["heat_score_raw"] = -pd.to_numeric(result["popularity_median"], errors="coerce")
    return result.drop(columns=["popularity_median"])


def build_daily_change_signals(
    selection: pd.DataFrame, fund: pd.DataFrame, sector_type: str
) -> pd.DataFrame:
    if sector_type not in {"industry", "concept"}:
        raise ValueError(f"不支持的板块类型：{sector_type}")
    constituent = pd.DataFrame(
        columns=["sector_name", "constituent_change_rate"]
    )
    if not selection.empty:
        data = selection.copy()
        if sector_type == "industry":
            data["sector_name"] = data.get("industry", "").map(_normalized_name)
        else:
            data["sector_name"] = (
                data.get("concept", "")
                .fillna("")
                .astype(str)
                .str.split(r"[,，;；|]")
            )
            data = data.explode("sector_name")
            data["sector_name"] = data["sector_name"].map(_normalized_name)
        data["change_value"] = pd.to_numeric(
            data.get("change_rate"), errors="coerce"
        )
        weights = pd.to_numeric(data.get("free_cap"), errors="coerce")
        data["effective_weight"] = weights.where(
            weights.gt(0) & data["change_value"].notna(), 0.0
        )
        data["weighted_change"] = (
            data["change_value"].fillna(0.0) * data["effective_weight"]
        )
        data = data[(data["sector_name"] != "") & data["change_value"].notna()]
        grouped = data.groupby("sector_name", as_index=False).agg(
            weighted_change=("weighted_change", "sum"),
            effective_weight=("effective_weight", "sum"),
            simple_change=("change_value", "mean"),
        )
        grouped["constituent_change_rate"] = np.where(
            grouped["effective_weight"] > 0,
            grouped["weighted_change"] / grouped["effective_weight"],
            grouped["simple_change"],
        )
        constituent = grouped[["sector_name", "constituent_change_rate"]]

    fund_change = pd.DataFrame(columns=["sector_name", "fund_change_rate"])
    if not fund.empty and {"name", "change_rate"}.issubset(fund.columns):
        fund_change = pd.DataFrame(
            {
                "sector_name": fund["name"].map(_normalized_name),
                "fund_change_rate": pd.to_numeric(
                    fund["change_rate"], errors="coerce"
                ),
            }
        )
        fund_change = (
            fund_change.dropna(subset=["fund_change_rate"])
            .query("sector_name != ''")
            .drop_duplicates("sector_name", keep="last")
        )

    result = constituent.merge(fund_change, on="sector_name", how="outer")
    fund_values = pd.to_numeric(result["fund_change_rate"], errors="coerce")
    constituent_values = pd.to_numeric(
        result["constituent_change_rate"], errors="coerce"
    )
    result["daily_change_rate"] = fund_values.where(
        fund_values.notna(), constituent_values
    )
    result["daily_change_source"] = np.where(
        result["fund_change_rate"].notna(), "fund_flow", "constituents"
    )
    return result[
        ["sector_name", "daily_change_rate", "daily_change_source"]
    ].dropna(subset=["daily_change_rate"])


def append_derived_latest_rows(
    history: pd.DataFrame, signals: pd.DataFrame, as_of: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = history.copy()
    target = pd.Timestamp(as_of).normalize()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if "derived_latest" not in data:
        data["derived_latest"] = False
    if "derived_source" not in data:
        data["derived_source"] = "history"
    expected_count = int(data["sector_id"].nunique()) if not data.empty else 0
    stats: dict[str, object] = {
        "derived_latest_count": 0,
        "derived_latest_ratio": 0.0,
        "derived_latest_sources": {},
    }
    if data.empty or signals.empty:
        return data, stats

    latest = data.sort_values("date").groupby("sector_id", as_index=False).tail(1)
    signal_data = signals.copy()
    signal_data["sector_name_key"] = signal_data["sector_name"].map(_normalized_name)
    signal_data = signal_data.drop_duplicates("sector_name_key", keep="last").set_index(
        "sector_name_key"
    )
    derived_rows: list[pd.Series] = []
    source_counts: dict[str, int] = {}
    for _index, previous in latest.iterrows():
        if pd.Timestamp(previous["date"]).normalize() >= target:
            continue
        name_key = _normalized_name(previous["sector_name"])
        if name_key not in signal_data.index:
            continue
        signal = signal_data.loc[name_key]
        rate = pd.to_numeric(signal.get("daily_change_rate"), errors="coerce")
        previous_close = pd.to_numeric(previous.get("close"), errors="coerce")
        if pd.isna(rate) or pd.isna(previous_close) or float(previous_close) <= 0:
            continue
        source = str(signal.get("daily_change_source") or "constituents")
        row = previous.copy()
        row["date"] = target
        row["open"] = float(previous_close)
        row["close"] = float(previous_close) * (1.0 + float(rate) / 100.0)
        row["high"] = max(float(row["open"]), float(row["close"]))
        row["low"] = min(float(row["open"]), float(row["close"]))
        row["derived_latest"] = True
        row["derived_source"] = source
        derived_rows.append(row)
        source_counts[source] = source_counts.get(source, 0) + 1

    if derived_rows:
        data = pd.concat([data, pd.DataFrame(derived_rows)], ignore_index=True)
        data = data.sort_values("date").drop_duplicates(
            ["sector_id", "date"], keep="last"
        )
    count = len(derived_rows)
    stats.update(
        {
            "derived_latest_count": count,
            "derived_latest_ratio": float(count / expected_count)
            if expected_count
            else 0.0,
            "derived_latest_sources": source_counts,
        }
    )
    return data, stats


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    deviation = float(values.std(ddof=0))
    if not np.isfinite(deviation) or deviation < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (values - float(values.mean())) / deviation


def _fetch_dataframe(connection, sql: str, parameters: tuple[object, ...] = ()) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(sql, parameters)
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def load_current_signals(
    sector_type: str,
    connection_factory: Callable[[], object] | None = None,
    *,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if sector_type not in {"industry", "concept"}:
        raise ValueError(f"不支持的板块类型：{sector_type}")
    if connection_factory is None:
        import pymysql

        connection_factory = lambda: pymysql.connect(
            host="127.0.0.1",
            port=3306,
            user="root",
            password="root",
            database="instockdb",
            charset="utf8mb4",
        )
    connection = connection_factory()
    try:
        table = f"cn_stock_fund_flow_{sector_type}"
        requested = pd.Timestamp(as_of).date() if as_of is not None else date.max
        fund = _fetch_dataframe(
            connection,
            f"""
            SELECT date, name, change_rate, fund_amount, fund_rate,
                   change_rate_5, fund_amount_5, fund_rate_5,
                   change_rate_10, fund_amount_10, fund_rate_10
            FROM {table}
            WHERE date = (SELECT MAX(date) FROM {table} WHERE date <= %s)
            """,
            (requested,),
        )
        selection = _fetch_dataframe(
            connection,
            """
            SELECT industry, concept, change_rate, free_cap, now_newhigh,
                   long_avg_array, popularity_rank
            FROM cn_stock_selection
            WHERE date = (SELECT MAX(date) FROM cn_stock_selection WHERE date <= %s)
            """,
            (requested,),
        )
        spot_date = _fetch_dataframe(
            connection,
            "SELECT MAX(date) AS date FROM cn_stock_spot WHERE date <= %s",
            (requested,),
        )
    finally:
        connection.close()

    dates = pd.to_datetime(fund.get("date"), errors="coerce").dropna()
    fallback_dates = pd.to_datetime(spot_date.get("date"), errors="coerce").dropna()
    as_of = dates.max() if not dates.empty else fallback_dates.max()
    if pd.isna(as_of):
        raise SectorDataUnavailable("数据库没有可用的最新交易日")

    if fund.empty:
        fund_signal = pd.DataFrame(columns=["sector_name", "fund_score", "fund_present"])
    else:
        fund_signal = pd.DataFrame({"sector_name": fund["name"].map(_normalized_name)})
        fund_signal["fund_score"] = (
            0.50 * _zscore(fund["fund_rate"])
            + 0.30 * _zscore(fund["fund_rate_5"])
            + 0.20 * _zscore(fund["fund_rate_10"])
        )
        fund_signal["fund_present"] = 1.0

    breadth = build_breadth_signals(selection, sector_type)
    daily_change = build_daily_change_signals(selection, fund, sector_type)
    if breadth.empty:
        breadth_signal = pd.DataFrame(
            columns=["sector_name", "breadth_score", "heat_score", "up_ratio", "breadth_present"]
        )
    else:
        breadth_signal = breadth.copy()
        breadth_signal["breadth_score"] = (
            breadth_signal["up_ratio"]
            + breadth_signal["new_high_ratio"]
            + breadth_signal["long_ma_ratio"]
        ) / 3.0
        breadth_signal["heat_score"] = _zscore(breadth_signal["heat_score_raw"])
        breadth_signal["breadth_present"] = (breadth_signal["member_count"] >= 3).astype(float)
        breadth_signal = breadth_signal[
            ["sector_name", "breadth_score", "heat_score", "up_ratio", "breadth_present"]
        ]

    signals = fund_signal.merge(breadth_signal, on="sector_name", how="outer")
    signals = signals.merge(daily_change, on="sector_name", how="outer")
    signals["current_data_completeness"] = (
        signals.get("fund_present", 0).fillna(0) + signals.get("breadth_present", 0).fillna(0)
    ) / 2.0
    for column in (
        "fund_score",
        "breadth_score",
        "heat_score",
        "up_ratio",
        "daily_change_rate",
    ):
        if column not in signals:
            signals[column] = np.nan
    return signals[
        [
            "sector_name",
            "fund_score",
            "breadth_score",
            "heat_score",
            "up_ratio",
            "current_data_completeness",
            "daily_change_rate",
            "daily_change_source",
        ]
    ], pd.Timestamp(as_of).normalize()


def _default_benchmark_loader(start: str, end: str) -> pd.DataFrame:
    return fetch_index_bars(
        "000300", start, end, Path(r"C:\InStockLocal\backtest_cache\indices")
    )


def _read_cached_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    return data.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")


def _write_cache_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.date.astype(str)
    output.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def load_sector_bundle(
    paths: AppPaths,
    sector_type: str,
    refresh: bool = False,
    *,
    fetch_api: SectorAPI | None = None,
    benchmark_loader: Callable[[str, str], pd.DataFrame] | None = None,
    signal_loader: Callable[[str], tuple[pd.DataFrame, pd.Timestamp]] | None = None,
) -> SectorDataBundle:
    if sector_type not in {"industry", "concept"}:
        raise ValueError(f"不支持的板块类型：{sector_type}")
    api = fetch_api or AkshareSectorAPI()
    benchmark_loader = benchmark_loader or _default_benchmark_loader
    signal_loader = signal_loader or load_current_signals
    signals, as_of = signal_loader(sector_type)
    as_of = pd.Timestamp(as_of).normalize()

    cache_root = Path(paths.data_dir) / "sector_forecast" / "cache" / sector_type
    cache_root.mkdir(parents=True, exist_ok=True)
    boards_path = cache_root.parent / f"{sector_type}-boards.csv"
    errors: list[str] = []
    using_stale_cache = False

    with _proxy_free_environment():
        try:
            boards = api.list_boards(sector_type).copy()
            boards["sector_id"] = boards["sector_id"].astype(str)
            boards["sector_name"] = boards["sector_name"].map(_normalized_name)
            signal_names = set(signals["sector_name"].map(_normalized_name))
            matched = boards[boards["sector_name"].isin(signal_names)].copy()
            if len(matched) >= min(10, max(1, len(signal_names))):
                boards = matched
            boards.to_csv(boards_path, index=False, encoding="utf-8")
        except Exception as exc:
            errors.append(f"板块列表更新失败：{exc}")
            using_stale_cache = True
            if not boards_path.exists():
                raise SectorDataUnavailable(errors[-1]) from exc
            boards = pd.read_csv(boards_path, dtype=str)

        target_text = as_of.strftime("%Y%m%d")
        default_start = (as_of.date() - timedelta(days=4 * 366)).strftime("%Y%m%d")
        prefetched: dict[str, pd.DataFrame] = {}
        remote_history_available = True
        probe_error: Exception | None = None

        for record in boards.to_dict("records"):
            sector_id = str(record["sector_id"])
            sector_name = _normalized_name(record["sector_name"])
            cache_path = cache_root / f"{sector_id}.csv"
            cached = _read_cached_history(cache_path)
            latest = (
                pd.to_datetime(cached.get("date"), errors="coerce").max()
                if not cached.empty
                else pd.NaT
            )
            needs_probe = (
                refresh
                or pd.isna(latest)
                or pd.Timestamp(latest).normalize() < as_of
            )
            if not needs_probe:
                continue
            start = default_start
            if not cached.empty and not refresh:
                start = (pd.Timestamp(latest) + pd.Timedelta(days=1)).strftime("%Y%m%d")
            try:
                fresh_raw = api.fetch_history(
                    sector_type, sector_id, start, target_text
                )
                fresh = normalize_sector_history(
                    fresh_raw, sector_type, sector_id, sector_name
                )
                if fresh.empty:
                    raise SectorDataUnavailable(f"{sector_name} 历史行情为空")
                prefetched[sector_id] = fresh
            except Exception as exc:
                remote_history_available = False
                probe_error = exc
                using_stale_cache = True
                errors.append(f"板块历史更新暂不可用，已回退缓存：{exc}")
            break

        def load_one(record: dict[str, object]) -> tuple[pd.DataFrame, str | None, bool]:
            sector_id = str(record["sector_id"])
            sector_name = _normalized_name(record["sector_name"])
            cache_path = cache_root / f"{sector_id}.csv"
            cached = _read_cached_history(cache_path)
            latest = pd.to_datetime(cached.get("date"), errors="coerce").max() if not cached.empty else pd.NaT
            needs_fetch = (
                refresh
                or pd.isna(latest)
                or pd.Timestamp(latest).normalize() < as_of
            )
            if not needs_fetch:
                return cached, None, False
            if not remote_history_available:
                if len(cached) >= 250:
                    return cached, f"{sector_name} 使用旧缓存：{probe_error}", True
                return pd.DataFrame(), f"{sector_name} 无可用历史：{probe_error}", True
            start = default_start
            if not cached.empty and not refresh:
                start = (pd.Timestamp(latest) + pd.Timedelta(days=1)).strftime("%Y%m%d")
            last_error: Exception | None = None
            for delay in (0,):
                if delay:
                    time.sleep(delay)
                try:
                    fresh = prefetched.get(sector_id)
                    if fresh is None:
                        fresh_raw = api.fetch_history(
                            sector_type, sector_id, start, target_text
                        )
                        fresh = normalize_sector_history(
                            fresh_raw, sector_type, sector_id, sector_name
                        )
                    merged = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
                    merged = merged.sort_values("date").drop_duplicates("date", keep="last")
                    if not merged.empty:
                        _write_cache_atomic(cache_path, merged)
                        return merged, None, False
                except Exception as exc:
                    last_error = exc
            if len(cached) >= 250:
                return cached, f"{sector_name} 使用旧缓存：{last_error}", True
            return pd.DataFrame(), f"{sector_name} 无可用历史：{last_error}", True

        histories: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="sector-history") as executor:
            futures = [executor.submit(load_one, item) for item in boards.to_dict("records")]
            for future in as_completed(futures):
                frame, error, stale = future.result()
                if not frame.empty:
                    histories.append(frame)
                if error:
                    errors.append(error)
                using_stale_cache = using_stale_cache or stale

    if not histories:
        raise SectorDataUnavailable("没有任何可用的板块历史数据")
    history = pd.concat(histories, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history, derived_stats = append_derived_latest_rows(history, signals, as_of)
    start = history["date"].min().date().isoformat()
    benchmark = benchmark_loader(start, as_of.date().isoformat()).copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce").dt.normalize()
    benchmark["benchmark_close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    history = history.merge(benchmark[["date", "benchmark_close"]], on="date", how="inner")

    aligned_as_of = pd.Timestamp(history["date"].max()).normalize()
    if aligned_as_of < as_of:
        try:
            signals, signal_date = signal_loader(sector_type, as_of=aligned_as_of)
        except TypeError:
            signal_date = as_of
        if pd.Timestamp(signal_date).normalize() == aligned_as_of:
            as_of = aligned_as_of

    boards = boards.copy()
    boards["sector_name_key"] = boards["sector_name"].map(_normalized_name)
    signals = signals.copy()
    signals["sector_name_key"] = signals["sector_name"].map(_normalized_name)
    current = boards.merge(signals.drop(columns=["sector_name"]), on="sector_name_key", how="left")
    current = current.drop(columns=["sector_name_key"])
    current["sector_type"] = sector_type
    current["current_data_completeness"] = pd.to_numeric(
        current["current_data_completeness"], errors="coerce"
    ).fillna(0.0)

    per_board_days = history.groupby("sector_id")["date"].nunique()
    board_count = int(per_board_days.size)
    expected_count = max(1, int(len(boards)))
    coverage_ratio = float((per_board_days >= 250).sum() / expected_count)
    valid_days = int(history["date"].nunique())
    latest_date = pd.Timestamp(history["date"].max()).normalize()
    latest_board_count = int(
        history.loc[history["date"] == latest_date, "sector_id"].nunique()
    )
    latest_board_coverage_ratio = float(latest_board_count / expected_count)
    missing_signal_ratio = float((current["current_data_completeness"] < 0.5).mean())
    runnable = (
        valid_days >= 250
        and coverage_ratio >= 0.75
        and latest_date >= as_of
        and latest_board_coverage_ratio >= 0.75
    )
    quality: dict[str, object] = {
        "valid_days": valid_days,
        "latest_date": latest_date.date().isoformat(),
        "board_count": board_count,
        "expected_board_count": expected_count,
        "coverage_ratio": coverage_ratio,
        "latest_board_count": latest_board_count,
        "latest_board_coverage_ratio": latest_board_coverage_ratio,
        "using_stale_cache": using_stale_cache,
        "missing_current_signal_ratio": missing_signal_ratio,
        "runnable": runnable,
        "errors": errors[:20],
        **derived_stats,
    }
    if valid_days < 250:
        raise SectorDataUnavailable(f"{sector_type} 有效历史仅 {valid_days} 个交易日")
    return SectorDataBundle(history, current, as_of, quality)
