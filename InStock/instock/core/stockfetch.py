#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
import datetime
import threading
import numpy as np
import pandas as pd
import talib as tl
import instock.core.tablestructure as tbs
import instock.lib.trade_time as trd
import instock.core.crawling.trade_date_hist as tdh
import instock.core.crawling.fund_etf_em as fee
import instock.core.crawling.stock_selection as sst
import instock.core.crawling.stock_lhb_em as sle
import instock.core.crawling.stock_lhb_sina as sls
import instock.core.crawling.stock_dzjy_em as sde
import instock.core.crawling.stock_hist_em as she
import instock.core.crawling.stock_fund_em as sff
import instock.core.crawling.stock_fhps_em as sfe
import instock.core.crawling.stock_chip_race as scr
import instock.core.crawling.stock_limitup_reason as slr

__author__ = 'myh '
__date__ = '2023/3/10 '

# 设置基础目录，每次加载使用。
cpath_current = os.path.dirname(os.path.dirname(__file__))
stock_hist_cache_path = os.path.join(cpath_current, 'cache', 'hist')
if not os.path.exists(stock_hist_cache_path):
    os.makedirs(stock_hist_cache_path)  # 创建多个文件夹结构。


# 600 601 603 605开头的股票是上证A股
# 600开头的股票是上证A股，属于大盘股，其中6006开头的股票是最早上市的股票，
# 6016开头的股票为大盘蓝筹股；900开头的股票是上证B股；
# 688开头的是上证科创板股票；
# 000开头的股票是深证A股，001、002开头的股票也都属于深证A股，
# 其中002开头的股票是深证A股中小企业股票；
# 200开头的股票是深证B股；
# 300、301开头的股票是创业板股票；400开头的股票是三板市场股票。
# 430、83、87开头的股票是北证A股
def is_a_stock(code):
    # 上证A股  # 深证A股
    return code.startswith(('600', '601', '603', '605', '000', '001', '002', '003', '300', '301'))


# 过滤掉 st 股票。
def is_not_st(name):
    return not name.startswith(('*ST', 'ST'))


# 过滤价格，如果没有基本上是退市了。
def is_open(price):
    return not np.isnan(price)


def is_open_with_line(price):
    return price != '-'


# 读取股票交易日历数据
def fetch_stocks_trade_date():
    try:
        data = tdh.tool_trade_date_hist_sina()
        if data is None or len(data.index) == 0:
            return None
        data_date = set(data['trade_date'].values.tolist())
        return data_date
    except Exception as e:
        logging.error(f"stockfetch.fetch_stocks_trade_date处理异常：{e}")
    return None


# 读取当天股票数据
def fetch_etfs(date):
    try:
        data = fee.fund_etf_spot_em()
        if data is None or len(data.index) == 0:
            raise RuntimeError("Eastmoney returned no ETF rows")
        if date is None:
            data.insert(0, 'date', datetime.datetime.now().strftime("%Y-%m-%d"))
        else:
            data.insert(0, 'date', date.strftime("%Y-%m-%d"))
        data.columns = list(tbs.TABLE_CN_ETF_SPOT['columns'])
        data = data.loc[data['new_price'].apply(is_open)]
        return data
    except Exception as e:
        logging.warning(f"ETF主数据源不可用，改用腾讯行情：{e}")
        try:
            from instock.core.fallback_quote import load_latest_snapshot, refresh_quotes
            columns = list(tbs.TABLE_CN_ETF_SPOT['columns'])
            data = refresh_quotes(load_latest_snapshot(tbs.TABLE_CN_ETF_SPOT['name'], columns))
            data.insert(0, 'date', date.strftime("%Y-%m-%d") if date else datetime.datetime.now().strftime("%Y-%m-%d"))
            logging.warning(f"ETF备用行情更新成功：{len(data.index)} 条")
            return data.loc[data['new_price'].apply(is_open)]
        except Exception as fallback_error:
            logging.error(f"ALL_SOURCES_FAILED stockfetch.fetch_etfs：主源={e}；备用源={fallback_error}")
    return None


# 读取当天股票数据
def fetch_stocks(date):
    try:
        data = she.stock_zh_a_spot_em()
        if data is None or len(data.index) == 0:
            raise RuntimeError("Eastmoney returned no stock rows")
        if date is None:
            data.insert(0, 'date', datetime.datetime.now().strftime("%Y-%m-%d"))
        else:
            data.insert(0, 'date', date.strftime("%Y-%m-%d"))
        data.columns = list(tbs.TABLE_CN_STOCK_SPOT['columns'])
        data = data.loc[data['code'].apply(is_a_stock)].loc[data['new_price'].apply(is_open)]
        from instock.core.fallback_quote import (
            load_latest_stock_universe,
            quote_coverage_is_sufficient,
        )
        universe = load_latest_stock_universe(list(tbs.TABLE_CN_STOCK_SPOT['columns']))
        if not quote_coverage_is_sufficient(len(data.index), len(universe.index)):
            raise RuntimeError(
                "Eastmoney stock snapshot coverage is incomplete: "
                f"rows={len(data.index)} universe={len(universe.index)}"
            )
        return data
    except Exception as e:
        logging.warning(f"股票主数据源不可用，改用腾讯行情：{e}")
        try:
            from instock.core.fallback_quote import (
                load_latest_snapshot,
                load_latest_stock_universe,
                merge_quote_universe,
                refresh_quotes,
            )
            columns = list(tbs.TABLE_CN_STOCK_SPOT['columns'])
            previous = load_latest_snapshot(tbs.TABLE_CN_STOCK_SPOT['name'], columns)
            universe = load_latest_stock_universe(columns)
            seeded = merge_quote_universe(previous, universe, columns)
            data = refresh_quotes(seeded)
            data.insert(0, 'date', date.strftime("%Y-%m-%d") if date else datetime.datetime.now().strftime("%Y-%m-%d"))
            data = data.loc[data['code'].apply(is_a_stock)].loc[data['new_price'].apply(is_open)]
            logging.warning(f"股票备用行情更新成功：{len(data.index)} 条")
            return data
        except Exception as fallback_error:
            logging.error(f"ALL_SOURCES_FAILED stockfetch.fetch_stocks：主源={e}；备用源={fallback_error}")
    return None


def fetch_stock_selection():
    try:
        data = sst.stock_selection()
        if data is None or len(data.index) == 0:
            return None
        data.columns = list(tbs.TABLE_CN_STOCK_SELECTION['columns'])
        data.drop_duplicates('code', keep='last', inplace=True)
        return data
    except Exception as e:
        logging.error(f"stockfetch.fetch_stocks_selection处理异常：{e}")
    return None


# 读取股票资金流向
def _deduplicate_fund_flow_rows(data, index):
    if data is None:
        return None
    if "code" not in data.columns:
        raise ValueError("individual fund-flow data is missing the code column")

    duplicate_rows = data.duplicated(subset="code", keep=False)
    if duplicate_rows.any():
        before = len(data.index)
        data = data.drop_duplicates(subset="code", keep="last")
        logging.warning(
            "individual fund-flow data contains duplicate codes: period=%s removed=%s",
            index,
            before - len(data.index),
        )
    return data.reset_index(drop=True)


def fetch_stocks_fund_flow(index):
    try:
        cn_flow = tbs.CN_STOCK_FUND_FLOW[index]
        data = sff.stock_individual_fund_flow_rank(indicator=cn_flow['cn'])
        if data is None or len(data.index) == 0:
            raise RuntimeError("Eastmoney returned no individual fund-flow rows")
        data.columns = list(cn_flow['columns'])
        data = data.loc[data['code'].apply(is_a_stock)].loc[data['new_price'].apply(is_open_with_line)]
        return _deduplicate_fund_flow_rows(data, index)
    except Exception as e:
        logging.warning(f"个股资金流主数据源不可用，改用同花顺：{e}")
        try:
            from instock.core.fund_flow_fallback import fetch_individual_rank
            data = fetch_individual_rank(index)
            data.columns = list(tbs.CN_STOCK_FUND_FLOW[index]['columns'])
            data = data.loc[data['code'].apply(is_a_stock)].loc[data['new_price'].notna()]
            logging.warning(f"个股资金流备用源更新成功：周期={index}，{len(data.index)} 条")
            return _deduplicate_fund_flow_rows(data, index)
        except Exception as fallback_error:
            logging.error(f"ALL_SOURCES_FAILED stockfetch.fetch_stocks_fund_flow：主源={e}；备用源={fallback_error}")
    return None


# 读取板块资金流向
def _deduplicate_sector_fund_flow_rows(data, index_sector, index_indicator):
    if data is None:
        return None
    if "name" not in data.columns:
        raise ValueError("sector fund-flow data is missing the name column")

    duplicate_rows = data.duplicated(subset="name", keep=False)
    if duplicate_rows.any():
        before = len(data.index)
        data = data.drop_duplicates(subset="name", keep="last")
        logging.warning(
            "sector fund-flow data contains duplicate names: sector=%s period=%s removed=%s",
            index_sector,
            index_indicator,
            before - len(data.index),
        )
    return data.reset_index(drop=True)


def fetch_stocks_sector_fund_flow(index_sector, index_indicator):
    try:
        cn_flow = tbs.CN_STOCK_SECTOR_FUND_FLOW[1][index_indicator]
        data = sff.stock_sector_fund_flow_rank(indicator=cn_flow['cn'], sector_type=tbs.CN_STOCK_SECTOR_FUND_FLOW[0][index_sector])
        if data is None or len(data.index) == 0:
            raise RuntimeError("Eastmoney returned no sector fund-flow rows")
        data.columns = list(cn_flow['columns'])
        return _deduplicate_sector_fund_flow_rows(data, index_sector, index_indicator)
    except Exception as e:
        logging.warning(f"板块资金流主数据源不可用，改用同花顺：{e}")
        try:
            from instock.core.fund_flow_fallback import fetch_sector_rank
            data = fetch_sector_rank(index_sector, index_indicator)
            data.columns = list(tbs.CN_STOCK_SECTOR_FUND_FLOW[1][index_indicator]['columns'])
            logging.warning(
                f"板块资金流备用源更新成功：板块={index_sector}，周期={index_indicator}，{len(data.index)} 条"
            )
            return _deduplicate_sector_fund_flow_rows(data, index_sector, index_indicator)
        except Exception as fallback_error:
            logging.error(f"ALL_SOURCES_FAILED stockfetch.fetch_stocks_sector_fund_flow：主源={e}；备用源={fallback_error}")
    return None


# 读取股票分红配送
def fetch_stocks_bonus(date):
    try:
        data = sfe.stock_fhps_em(date=trd.get_bonus_report_date())
        if data is None or len(data.index) == 0:
            return None
        if date is None:
            data.insert(0, 'date', datetime.datetime.now().strftime("%Y-%m-%d"))
        else:
            data.insert(0, 'date', date.strftime("%Y-%m-%d"))
        data.columns = list(tbs.TABLE_CN_STOCK_BONUS['columns'])
        data = data.loc[data['code'].apply(is_a_stock)]
        return data
    except Exception as e:
        logging.error(f"stockfetch.fetch_stocks_bonus处理异常：{e}")
    return None


# 股票近三月上龙虎榜且必须有2次以上机构参与的
def fetch_stock_top_entity_data(date):
    run_date = date + datetime.timedelta(days=-90)
    start_date = run_date.strftime("%Y%m%d")
    end_date = date.strftime("%Y%m%d")
    code_name = '代码'
    entity_amount_name = '买方机构数'
    try:
        data = sle.stock_lhb_jgmmtj_em(start_date, end_date)
        if data is None or len(data.index) == 0:
            return None

        # 机构买入次数大于1计算方法，首先：每次要有买方机构数(>0),然后：这段时间买方机构数求和大于1
        mask = (data[entity_amount_name] > 0)  # 首先：每次要有买方机构数(>0)
        data = data.loc[mask]

        if len(data.index) == 0:
            return None

        grouped = data.groupby(by=data[code_name])
        data_series = grouped[entity_amount_name].sum()
        data_code = set(data_series[data_series > 1].index.values)  # 然后：这段时间买方机构数求和大于1

        if not data_code:
            return None

        return data_code
    except Exception as e:
        logging.error(f"stockfetch.fetch_stock_top_entity_data处理异常：{e}")
    return None

# 描述: 获取东方财富-龙虎榜-个股上榜统计
def fetch_stock_lhb_data(date,count=12):
    try:
        start_date = trd.get_previous_trade_date(date,count).strftime("%Y%m%d")
        end_date = date.strftime("%Y%m%d")

        data = sle.stock_lhb_detail_em(start_date, end_date)
        if data is None or len(data.index) == 0:
            return None
        _columns = list(tbs.TABLE_CN_STOCK_lHB['columns'])
        _columns.pop(0)
        data.columns = _columns
        data = data.loc[data['code'].apply(is_a_stock)]
        data.drop_duplicates('code', keep='last', inplace=True)
        # data = data.sort_values(by='ranking_times', ascending=False)
        if date is None:
            data.insert(0, 'date', datetime.datetime.now().strftime("%Y-%m-%d"))
        else:
            data.insert(0, 'date', date.strftime("%Y-%m-%d"))
        return data
    except Exception as e:
        logging.error(f"stockfetch.fetch_stock_lhb_data处理异常：{e}")
    return None

# 描述: 获取新浪财经-龙虎榜-个股上榜统计
def fetch_stock_top_data(date):
    try:
        data = sls.stock_lhb_ggtj_sina()
        if data is None or len(data.index) == 0:
            return None
        _columns = list(tbs.TABLE_CN_STOCK_TOP['columns'])
        _columns.pop(0)
        data.columns = _columns
        data = data.loc[data['code'].apply(is_a_stock)]
        data.drop_duplicates('code', keep='last', inplace=True)
        if date is None:
            data.insert(0, 'date', datetime.datetime.now().strftime("%Y-%m-%d"))
        else:
            data.insert(0, 'date', date.strftime("%Y-%m-%d"))
        return data
    except Exception as e:
        logging.error(f"stockfetch.fetch_stock_top_data处理异常：{e}")
    return None


# 描述: 获取东方财富网-数据中心-大宗交易-每日统计
def fetch_stock_blocktrade_data(date):
    date_str = date.strftime("%Y%m%d")
    try:
        data = sde.stock_dzjy_mrtj(start_date=date_str, end_date=date_str)
        if data is None or len(data.index) == 0:
            return None

        columns = list(tbs.TABLE_CN_STOCK_BLOCKTRADE['columns'])
        columns.insert(0, 'index')
        data.columns = columns
        data = data.loc[data['code'].apply(is_a_stock)]
        data.drop('index', axis=1, inplace=True)
        return data
    except TypeError:
        logging.error("处理异常：目前还没有大宗交易数据，请17:00点后再获取！")
        return None
    except Exception as e:
        logging.error(f"stockfetch.fetch_stock_blocktrade_data处理异常：{e}")
    return None

# 读取早盘抢筹
def fetch_stock_chip_race_open(date):
    try:
        date_str =""
        if date != datetime.datetime.now().date():
            date_str = date.strftime("%Y%m%d")
        data = scr.stock_chip_race_open(date_str)
        if data is None or len(data.index) == 0:
            return None
        if date is None:
            data.insert(0, 'date', datetime.datetime.now().strftime("%Y-%m-%d"))
        else:
            data.insert(0, 'date', date.strftime("%Y-%m-%d"))
        data.columns = list(tbs.TABLE_CN_STOCK_CHIP_RACE_OPEN['columns'])
        return data
    except Exception as e:
        logging.error(f"stockfetch.fetch_stock_chip_race_open处理异常：{e}")
    return None

# 读取尾盘抢筹
def fetch_stock_chip_race_end(date):
    try:
        date_str =""
        if date != datetime.datetime.now().date():
            date_str = date.strftime("%Y%m%d")
        data = scr.stock_chip_race_end(date_str)
        if data is None or len(data.index) == 0:
            return None
        if date is None:
            data.insert(0, 'date', datetime.datetime.now().strftime("%Y-%m-%d"))
        else:
            data.insert(0, 'date', date.strftime("%Y-%m-%d"))
        data.columns = list(tbs.TABLE_CN_STOCK_CHIP_RACE_END['columns'])
        return data
    except Exception as e:
        logging.error(f"stockfetch.fetch_stock_chip_race_end处理异常：{e}")
    return None

# 读取涨停原因
def fetch_stock_limitup_reason(date):

    try:
        data = slr.stock_limitup_reason(date.strftime("%Y-%m-%d"))
        if data is None or len(data.index) == 0:
            return None
        data.columns = list(tbs.TABLE_CN_STOCK_LIMITUP_REASON['columns'])
        return data
    except Exception as e:
        logging.error(f"stockfetch.fetch_stock_limitup_reason处理异常：{e}")
    return None

# 读取股票历史数据
def fetch_etf_hist(data_base, date_start=None, date_end=None, adjust='qfq'):
    date = data_base[0]
    code = data_base[1]

    if date_start is None:
        date_start, is_cache = trd.get_trade_hist_interval(date)  # 提高运行效率，只运行一次
    try:
        if date_end is not None:
            data = fee.fund_etf_hist_em(symbol=code, period="daily", start_date=date_start, end_date=date_end,
                                        adjust=adjust)
        else:
            data = fee.fund_etf_hist_em(symbol=code, period="daily", start_date=date_start, adjust=adjust)

        if data is None or len(data.index) == 0:
            return None
        data.columns = tuple(tbs.CN_STOCK_HIST_DATA['columns'])
        data = data.sort_index()  # 将数据按照日期排序下。
        if data is not None:
            data.loc[:, 'p_change'] = tl.ROC(data['close'].values, 1)
            data['p_change'].values[np.isnan(data['p_change'].values)] = 0.0
            data["volume"] = data['volume'].values.astype('double') * 100  # 成交量单位从手变成股。
        return data
    except Exception as e:
        logging.error(f"stockfetch.fetch_etf_hist处理异常：{e}")
    return None


# 读取股票历史数据
def fetch_stock_hist(data_base, date_start=None, is_cache=True, latest_spot=None):
    date = data_base[0]
    code = data_base[1]

    if date_start is None:
        date_start, is_cache = trd.get_trade_hist_interval(date)  # 提高运行效率，只运行一次
        # date_end = date_end.strftime("%Y%m%d")
    try:
        data = stock_hist_cache(
            code, date_start, None, is_cache, 'qfq', latest_spot=latest_spot
        )
        if data is not None:
            data.loc[:, 'p_change'] = tl.ROC(data['close'].values, 1)
            data['p_change'].values[np.isnan(data['p_change'].values)] = 0.0
            data["volume"] = data['volume'].values.astype('double') * 100  # 成交量单位从手变成股。
        return data
    except Exception as e:
        logging.error(f"stockfetch.fetch_stock_hist处理异常：{e}")
    return None


_HISTORY_SPOT_COLUMNS = (
    "date",
    "open_price",
    "new_price",
    "high_price",
    "low_price",
    "volume",
    "deal_amount",
    "amplitude",
    "change_rate",
    "ups_downs",
    "turnoverrate",
)


def _roll_forward_previous_history_cache(
    code, date_start, cache_file, adjust, latest_spot=None
):
    """Build today's history cache from the last cache plus today's spot bar."""
    try:
        start = datetime.datetime.strptime(date_start, "%Y%m%d").date()
    except ValueError:
        return None

    previous_file = None
    for days_back in range(1, 32):
        previous_start = (start - datetime.timedelta(days=days_back)).strftime("%Y%m%d")
        candidate = os.path.join(
            stock_hist_cache_path,
            previous_start[0:6],
            previous_start,
            "%s%s.gzip.pickle" % (code, adjust),
        )
        if os.path.isfile(candidate):
            previous_file = candidate
            break
    if previous_file is None:
        return None

    if latest_spot is None:
        from instock.lib import database as mdb

        latest = mdb.executeSqlFetch(
            "SELECT `date`,`open_price`,`new_price`,`high_price`,`low_price`,`volume`,"
            "`deal_amount`,`amplitude`,`change_rate`,`ups_downs`,`turnoverrate` "
            "FROM `cn_stock_spot` WHERE `code`=%s ORDER BY `date` DESC LIMIT 1",
            (code,),
        )
        if not latest:
            return None
        spot = list(latest[0])
    else:
        try:
            spot = [latest_spot[column] for column in _HISTORY_SPOT_COLUMNS]
        except (KeyError, TypeError):
            spot = list(latest_spot)
        if len(spot) != len(_HISTORY_SPOT_COLUMNS):
            return None

    history = pd.read_pickle(previous_file, compression="gzip")
    spot[0] = spot[0].strftime("%Y-%m-%d") if hasattr(spot[0], "strftime") else str(spot[0])
    # In cn_stock_spot volume is stored in shares; historical source uses lots.
    spot[5] = float(spot[5]) / 100 if spot[5] is not None else None
    current = pd.DataFrame([spot], columns=list(tbs.CN_STOCK_HIST_DATA['columns']))
    history = pd.concat([history, current], ignore_index=True)
    history['_date_key'] = pd.to_datetime(history['date']).dt.strftime("%Y-%m-%d")
    history.drop_duplicates('_date_key', keep='last', inplace=True)
    history.drop(columns=['_date_key'], inplace=True)
    history.sort_values('date', inplace=True)
    history.reset_index(drop=True, inplace=True)
    history.to_pickle(cache_file, compression="gzip")
    return history


# 增加读取股票缓存方法。加快处理速度。多线程解决效率
def stock_hist_cache(
    code, date_start, date_end=None, is_cache=True, adjust='', latest_spot=None
):
    cache_dir = os.path.join(stock_hist_cache_path, date_start[0:6], date_start)
    # 如果没有文件夹创建一个。月文件夹和日文件夹。方便删除。
    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    except Exception:
        pass
    cache_file = os.path.join(cache_dir, "%s%s.gzip.pickle" % (code, adjust))
    # 如果缓存存在就直接返回缓存数据。压缩方式。
    try:
        if os.path.isfile(cache_file):
            return pd.read_pickle(cache_file, compression="gzip")
        else:
            if is_cache:
                rolled_forward = _roll_forward_previous_history_cache(
                    code, date_start, cache_file, adjust, latest_spot=latest_spot
                )
                if rolled_forward is not None:
                    return rolled_forward
            if date_end is not None:
                stock = she.stock_zh_a_hist(symbol=code, period="daily", start_date=date_start, end_date=date_end,
                                            adjust=adjust)
            else:
                stock = she.stock_zh_a_hist(symbol=code, period="daily", start_date=date_start, adjust=adjust)

            if stock is None or len(stock.index) == 0:
                return None
            stock.columns = tuple(tbs.CN_STOCK_HIST_DATA['columns'])
            stock = stock.sort_index()  # 将数据按照日期排序下。
            try:
                if is_cache:
                    stock.to_pickle(cache_file, compression="gzip")
            except Exception:
                pass
            # time.sleep(1)
            return stock
    except Exception as e:
        logging.error(f"stockfetch.stock_hist_cache处理异常：{code}代码{e}")
    return None


# Resilient source overrides: prefer stable quote refresh, keep push2 as a last resort.
_push2_fetch_etfs = fetch_etfs
_push2_fetch_stocks = fetch_stocks
_push2_fetch_stocks_fund_flow = fetch_stocks_fund_flow


def _date_string(date):
    return date.strftime("%Y-%m-%d") if date else datetime.datetime.now().strftime("%Y-%m-%d")


def _quote_snapshot(table, date):
    from instock.core.fallback_quote import (
        load_latest_snapshot,
        load_latest_stock_universe,
        merge_quote_universe,
        refresh_quotes,
    )

    columns = list(table["columns"])
    previous = load_latest_snapshot(table["name"], columns)
    if table["name"] == tbs.TABLE_CN_STOCK_SPOT["name"]:
        universe = load_latest_stock_universe(columns)
        previous = merge_quote_universe(previous, universe, columns)
    data = refresh_quotes(previous)
    data.insert(0, "date", _date_string(date))
    return data


def fetch_etfs(date):
    try:
        data = _quote_snapshot(tbs.TABLE_CN_ETF_SPOT, date)
        return data.loc[data["new_price"].apply(is_open)]
    except Exception as quote_error:
        logging.warning("ETF quote snapshot source unavailable, falling back to Eastmoney push2: %s", quote_error)
        return _push2_fetch_etfs(date)


def fetch_stocks(date):
    try:
        data = _quote_snapshot(tbs.TABLE_CN_STOCK_SPOT, date)
        data = data.loc[data["code"].apply(is_a_stock)].loc[data["new_price"].apply(is_open)]
        return data
    except Exception as quote_error:
        logging.warning("stock quote snapshot source unavailable, falling back to Eastmoney push2: %s", quote_error)
        return _push2_fetch_stocks(date)


def fetch_stocks_fund_flow(index):
    try:
        from instock.core.fund_flow_fallback import fetch_individual_rank

        data = fetch_individual_rank(index)
        data.columns = list(tbs.CN_STOCK_FUND_FLOW[index]["columns"])
        data = data.loc[data["code"].apply(is_a_stock)].loc[data["new_price"].notna()]
        return _deduplicate_fund_flow_rows(data, index)
    except Exception as flow_error:
        logging.warning("xuangu fund-flow source unavailable, falling back to Eastmoney push2: %s", flow_error)
        return _push2_fetch_stocks_fund_flow(index)


_hist_failure_log_counts = {}
_eastmoney_history_unavailable = threading.Event()


def _log_history_cache_failure(code, error):
    key = (type(error).__name__, str(error))
    count = _hist_failure_log_counts.get(key, 0) + 1
    _hist_failure_log_counts[key] = count
    if count <= 3:
        logging.error("stockfetch.stock_hist_cache failed: code=%s error=%s", code, error)
    elif count == 4:
        logging.error("further similar history failures suppressed: error=%s", error)


def stock_hist_cache(
    code, date_start, date_end=None, is_cache=True, adjust='', latest_spot=None
):
    cache_dir = os.path.join(stock_hist_cache_path, date_start[0:6], date_start)
    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    except Exception:
        pass
    cache_file = os.path.join(cache_dir, "%s%s.gzip.pickle" % (code, adjust))
    try:
        if os.path.isfile(cache_file):
            return pd.read_pickle(cache_file, compression="gzip")
        if is_cache:
            rolled_forward = _roll_forward_previous_history_cache(
                code, date_start, cache_file, adjust, latest_spot=latest_spot
            )
            if rolled_forward is not None:
                return rolled_forward
        stock = None
        if not _eastmoney_history_unavailable.is_set():
            try:
                if date_end is not None:
                    stock = she.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=date_start,
                        end_date=date_end,
                        adjust=adjust,
                    )
                else:
                    stock = she.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=date_start,
                        adjust=adjust,
                    )
            except Exception:
                _eastmoney_history_unavailable.set()

        if stock is None or len(stock.index) == 0:
            try:
                from instock.core.history_fallback import fetch_tencent_stock_history

                stock = fetch_tencent_stock_history(
                    code,
                    date_start,
                    date_end,
                    adjust=adjust,
                    latest_spot=latest_spot,
                )
            except Exception as fallback_error:
                _log_history_cache_failure(code, fallback_error)
                return None
        stock.columns = tuple(tbs.CN_STOCK_HIST_DATA['columns'])
        stock = stock.sort_index()
        try:
            if is_cache:
                stock.to_pickle(cache_file, compression="gzip")
        except Exception:
            pass
        return stock
    except Exception as error:
        _log_history_cache_failure(code, error)
    return None
