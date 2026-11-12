#!/usr/local/bin/python3
# -*- coding: utf-8 -*-


import logging
import concurrent.futures
import math
import pandas as pd
import os.path
import sys

cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
import instock.lib.run_template as runt
import instock.core.tablestructure as tbs
import instock.lib.database as mdb
import instock.lib.trade_time as trd
import instock.core.indicator.calculate_indicator as idr
from instock.core.singleton_stock import stock_hist_data

__author__ = 'myh '
__date__ = '2023/3/10 '


def require_indicator_partition(date, count_provider=None, minimum_ratio=0.95):
    if count_provider is None:
        def count_provider(table_name, date_value):
            result = mdb.executeSqlFetch(
                f"SELECT COUNT(*) FROM `{table_name}` WHERE `date`=%s",
                (date_value,),
            )
            return int(result[0][0]) if result else 0

    stock_table = tbs.TABLE_CN_STOCK_SPOT['name']
    indicator_table = tbs.TABLE_CN_STOCK_INDICATORS['name']
    stock_rows = int(count_provider(stock_table, date))
    indicator_rows = int(count_provider(indicator_table, date))
    required = math.ceil(stock_rows * float(minimum_ratio))
    if stock_rows <= 0 or indicator_rows < required:
        coverage = indicator_rows / stock_rows if stock_rows else 0.0
        raise RuntimeError(
            "cn_stock_indicators coverage is incomplete: "
            f"indicator_rows={indicator_rows} stock_rows={stock_rows} "
            f"coverage={coverage:.1%} required={minimum_ratio:.1%}"
        )


def prepare(date):
    try:
        stocks_data = stock_hist_data(date=date).get_data()
        if stocks_data is None:
            return
        results = run_check(stocks_data, date=date)
        if results is None:
            return

        table_name = tbs.TABLE_CN_STOCK_INDICATORS['name']
        dataKey = pd.DataFrame(results.keys())
        _columns = tuple(tbs.TABLE_CN_STOCK_FOREIGN_KEY['columns'])
        dataKey.columns = _columns

        dataVal = pd.DataFrame(results.values())
        dataVal.drop('date', axis=1, inplace=True)  # 删除日期字段，然后和原始数据合并。

        data = pd.merge(dataKey, dataVal, on=['code'], how='left')
        # data.set_index('code', inplace=True)
        # 单例，时间段循环必须改时间
        date_str = date.strftime("%Y-%m-%d")
        if date.strftime("%Y-%m-%d") != data.iloc[0]['date']:
            data['date'] = date_str
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_INDICATORS['columns']),
            primary_keys="`date`,`code`",
        )

    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED indicators_data_daily_job.prepare：{e}")


def run_check(stocks, date=None, workers=40):
    data = {}
    columns = list(tbs.STOCK_STATS_DATA['columns'])
    columns.insert(0, 'code')
    columns.insert(0, 'date')
    data_column = columns
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_data = {executor.submit(idr.get_indicator, k, stocks[k], data_column, date=date): k for k in stocks}
            for future in concurrent.futures.as_completed(future_to_data):
                stock = future_to_data[future]
                try:
                    _data_ = future.result()
                    if _data_ is not None:
                        data[stock] = _data_
                except Exception as e:
                    logging.error(f"indicators_data_daily_job.run_check处理异常：{stock[1]}代码{e}")
    except Exception as e:
        logging.error(f"indicators_data_daily_job.run_check处理异常：{e}")
    if not data:
        return None
    else:
        return data


# 对每日指标数据，进行筛选。将符合条件的。二次筛选出来。
# 只是做简单筛选
def guess_buy(date):
    try:
        _table_name = tbs.TABLE_CN_STOCK_INDICATORS['name']
        if not mdb.checkTableIsExist(_table_name):
            return

        _columns = tuple(tbs.TABLE_CN_STOCK_FOREIGN_KEY['columns'])
        _selcol = '`,`'.join(_columns)
        sql = f'''SELECT `{_selcol}` FROM `{_table_name}` WHERE `date` = '{date}' and 
                `kdjk` >= 80 and `kdjd` >= 70 and `kdjj` >= 100 and `rsi_6` >= 80 and 
                `cci` >= 100 and `cr` >= 300 and `wr_6` >= -20 and `vr` >= 160'''
        data = pd.read_sql(sql=sql, con=mdb.engine())
        data = data.drop_duplicates(subset="code", keep="last")
        # data.set_index('code', inplace=True)

        if len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_STOCK_INDICATORS_BUY['name']
        _columns_backtest = tuple(tbs.TABLE_CN_STOCK_BACKTEST_DATA['columns'])
        data = pd.concat([data, pd.DataFrame(columns=_columns_backtest)])
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_INDICATORS_BUY['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED indicators_data_daily_job.guess_buy：{e}")


# 设置卖出数据。
def guess_sell(date):
    try:
        _table_name = tbs.TABLE_CN_STOCK_INDICATORS['name']
        if not mdb.checkTableIsExist(_table_name):
            return

        _columns = tuple(tbs.TABLE_CN_STOCK_FOREIGN_KEY['columns'])
        _selcol = '`,`'.join(_columns)
        sql = f'''SELECT `{_selcol}` FROM `{_table_name}` WHERE `date` = '{date}' and 
                `kdjk` < 20 and `kdjd` < 30 and `kdjj` < 10 and `rsi_6` < 20 and 
                `cci` < -100 and `cr` < 40 and `wr_6` < -80 and `vr` < 40'''
        data = pd.read_sql(sql=sql, con=mdb.engine())
        data = data.drop_duplicates(subset="code", keep="last")
        # data.set_index('code', inplace=True)
        if len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_STOCK_INDICATORS_SELL['name']
        _columns_backtest = tuple(tbs.TABLE_CN_STOCK_BACKTEST_DATA['columns'])
        data = pd.concat([data, pd.DataFrame(columns=_columns_backtest)])
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_INDICATORS_SELL['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED indicators_data_daily_job.guess_sell：{e}")


def main():
    # 使用方法传递。
    runt.run_with_args(prepare)
    # 二次筛选数据。直接计算买卖股票数据。
    runt.run_with_args(guess_buy)
    runt.run_with_args(guess_sell)
    if len(sys.argv) == 1:
        _, latest_trade_date = trd.get_trade_date_last()
        require_indicator_partition(latest_trade_date)


# main函数入口
if __name__ == '__main__':
    main()
