#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import logging
import os.path
import sys

cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
import instock.lib.run_template as runt
import instock.core.tablestructure as tbs
import instock.lib.database as mdb
import instock.lib.trade_time as trd
import instock.core.stockfetch as stf
from instock.core.singleton_stock import stock_data

__author__ = 'myh '
__date__ = '2023/3/10 '


SNAPSHOT_TABLES = (
    tbs.TABLE_CN_STOCK_SPOT['name'],
    tbs.TABLE_CN_ETF_SPOT['name'],
)


def require_snapshot_partitions(date, count_provider=None):
    if count_provider is None:
        def count_provider(table_name, date_value):
            result = mdb.executeSqlFetch(
                f"SELECT COUNT(*) FROM `{table_name}` WHERE `date`=%s",
                (date_value,),
            )
            return int(result[0][0]) if result else 0

    for table_name in SNAPSHOT_TABLES:
        rows = int(count_provider(table_name, date))
        if rows <= 0:
            raise RuntimeError(f"{table_name} has no rows for {date}")


# 股票实时行情数据。
def save_nph_stock_spot_data(date, before=True):
    if before:
        return
    # 股票列表
    try:
        data = stock_data(date).get_data()
        if data is None or len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_STOCK_SPOT['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_SPOT['columns']),
            primary_keys="`date`,`code`",
        )

    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED basic_data_daily_job.save_stock_spot_data：{e}")


# 基金实时行情数据。
def save_nph_etf_spot_data(date, before=True):
    if before:
        return
    # 股票列表
    try:
        data = stf.fetch_etfs(date)
        if data is None or len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_ETF_SPOT['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_ETF_SPOT['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED basic_data_daily_job.save_nph_etf_spot_data：{e}")



def main():
    runt.run_with_args(save_nph_stock_spot_data)
    runt.run_with_args(save_nph_etf_spot_data)
    if len(sys.argv) == 1:
        _, latest_trade_date = trd.get_trade_date_last()
        require_snapshot_partitions(latest_trade_date)


# main函数入口
if __name__ == '__main__':
    main()
