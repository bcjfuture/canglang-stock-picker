#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import logging
import concurrent.futures
import os.path
import sys
import pandas as pd

cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
import instock.lib.run_template as runt
import instock.core.tablestructure as tbs
import instock.lib.database as mdb
import instock.core.stockfetch as stf

__author__ = 'myh '
__date__ = '2023/3/10 '

# 每日股票龙虎榜
def save_nph_stock_lhb_data(date, before=True):
    if before:
        return

    try:
        data = stf.fetch_stock_lhb_data(date)
        if data is None or len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_STOCK_lHB['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_lHB['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"basic_data_other_daily_job.save_stock_lhb_data处理异常：{e}")
    stock_spot_buy(date)

# 每日股票龙虎榜(新浪)
def save_nph_stock_top_data(date, before=True):
    if before:
        return

    try:
        data = stf.fetch_stock_top_data(date)
        if data is None or len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_STOCK_TOP['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_TOP['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED basic_data_other_daily_job.save_stock_top_data：{e}")
    stock_spot_buy(date)


# 每日股票资金流向
def save_nph_stock_fund_flow_data(date, before=True):
    if before:
        return

    try:
        times = tuple(range(4))
        results = run_check_stock_fund_flow(times)
        if results is None:
            return

        for t in times:
            if t == 0:
                data = results.get(t)
            else:
                r = results.get(t)
                if r is not None:
                    r.drop(columns=['name', 'new_price'], inplace=True)
                    data = pd.merge(data, r, on=['code'], how='left')

        if data is None or len(data.index) == 0:
            return

        duplicate_codes = data.loc[
            data.duplicated(subset="code", keep=False), "code"
        ].drop_duplicates().tolist()
        if duplicate_codes:
            logging.error(
                "DUPLICATE_FUND_FLOW_CODES merged fund-flow data is not unique: count=%s sample=%s",
                len(duplicate_codes),
                duplicate_codes[:10],
            )
            return

        data.insert(0, 'date', date.strftime("%Y-%m-%d"))

        table_name = tbs.TABLE_CN_STOCK_FUND_FLOW['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_FUND_FLOW['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED basic_data_other_daily_job.save_nph_stock_fund_flow_data：{e}")


def run_check_stock_fund_flow(times):
    data = {}
    try:
        for k in times :
            _data = stf.fetch_stocks_fund_flow(k)
            if _data is not None:
                data[k] = _data
    except Exception as e:
        logging.error(f"basic_data_other_daily_job.run_check_stock_fund_flow处理异常：{e}")
    # try:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=len(times)) as executor:
    #         future_to_data = {executor.submit(stf.fetch_stocks_fund_flow, k): k for k in times}
    #         for future in concurrent.futures.as_completed(future_to_data):
    #             _time = future_to_data[future]
    #             try:
    #                 _data_ = future.result()
    #                 if _data_ is not None:
    #                     data[_time] = _data_
    #             except Exception as e:
    #                 logging.error(f"basic_data_other_daily_job.run_check_stock_fund_flow处理异常：代码{e}")
    # except Exception as e:
    #     logging.error(f"basic_data_other_daily_job.run_check_stock_fund_flow处理异常：{e}")
    if not data:
        return None
    else:
        return data


# 每日行业资金流向
def save_nph_stock_sector_fund_flow_data(date, before=True):
    if before:
        return

    # times = tuple(range(2))
    # with concurrent.futures.ThreadPoolExecutor(max_workers=len(times)) as executor:
    #     {executor.submit(stock_sector_fund_flow_data, date, k): k for k in times}
    stock_sector_fund_flow_data(date, 0)
    stock_sector_fund_flow_data(date, 1)

def stock_sector_fund_flow_data(date, index_sector):
    try:
        times = tuple(range(3))
        results = run_check_stock_sector_fund_flow(index_sector, times)
        if results is None:
            return

        for t in times:
            if t == 0:
                data = results.get(t)
            else:
                r = results.get(t)
                if r is not None:
                    data = pd.merge(data, r, on=['name'], how='left')

        if data is None or len(data.index) == 0:
            return

        duplicate_names = data.loc[
            data.duplicated(subset="name", keep=False), "name"
        ].drop_duplicates().tolist()
        if duplicate_names:
            logging.error(
                "DUPLICATE_SECTOR_FUND_FLOW_NAMES merged sector fund-flow data is not unique: sector=%s count=%s sample=%s",
                index_sector,
                len(duplicate_names),
                duplicate_names[:10],
            )
            return

        data.insert(0, 'date', date.strftime("%Y-%m-%d"))

        if index_sector == 0:
            tbs_table = tbs.TABLE_CN_STOCK_FUND_FLOW_INDUSTRY
        else:
            tbs_table = tbs.TABLE_CN_STOCK_FUND_FLOW_CONCEPT
        table_name = tbs_table['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "name"],
            cols_type=tbs.get_field_types(tbs_table['columns']),
            primary_keys="`date`,`name`",
        )
    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED basic_data_other_daily_job.stock_sector_fund_flow_data：{e}")


def run_check_stock_sector_fund_flow(index_sector, times):
    data = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(times)) as executor:
            future_to_data = {executor.submit(stf.fetch_stocks_sector_fund_flow, index_sector, k): k for k in times}
            for future in concurrent.futures.as_completed(future_to_data):
                _time = future_to_data[future]
                try:
                    _data_ = future.result()
                    if _data_ is not None:
                        data[_time] = _data_
                except Exception as e:
                    logging.error(f"basic_data_other_daily_job.run_check_stock_sector_fund_flow处理异常：代码{e}")
    except Exception as e:
        logging.error(f"basic_data_other_daily_job.run_check_stock_sector_fund_flow处理异常：{e}")
    if not data:
        return None
    else:
        return data


# 每日股票分红配送
def save_nph_stock_bonus(date, before=True):
    if before:
        return

    try:
        data = stf.fetch_stocks_bonus(date)
        if data is None or len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_STOCK_BONUS['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_BONUS['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED basic_data_other_daily_job.save_nph_stock_bonus：{e}")


# 基本面选股
def stock_spot_buy(date):
    try:
        _table_name = tbs.TABLE_CN_STOCK_SPOT['name']
        if not mdb.checkTableIsExist(_table_name):
            return

        sql = f'''SELECT * FROM `{_table_name}` WHERE `date` = '{date}' and 
                `pe9` > 0 and `pe9` <= 20 and `pbnewmrq` <= 10 and `roe_weight` >= 15'''
        data = pd.read_sql(sql=sql, con=mdb.engine())
        data = data.drop_duplicates(subset="code", keep="last")
        if len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_STOCK_SPOT_BUY['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_SPOT_BUY['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED basic_data_other_daily_job.stock_spot_buy：{e}")


# 每日早盘抢筹
def stock_chip_race_open_data(date):
    try:
        data = stf.fetch_stock_chip_race_open(date)
        if data is None or len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_STOCK_CHIP_RACE_OPEN['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_CHIP_RACE_OPEN['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"basic_data_other_daily_job.stock_chip_race_open_data：{e}")


# 每日涨停原因
def stock_imitup_reason_data(date):
    try:
        data = stf.fetch_stock_limitup_reason(date)
        if data is None or len(data.index) == 0:
            return

        table_name = tbs.TABLE_CN_STOCK_LIMITUP_REASON['name']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_LIMITUP_REASON['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"basic_data_other_daily_job.stock_imitup_reason_data：{e}")

def main():
    runt.run_with_args(save_nph_stock_lhb_data)
    runt.run_with_args(save_nph_stock_bonus)
    runt.run_with_args(save_nph_stock_fund_flow_data)
    runt.run_with_args(save_nph_stock_sector_fund_flow_data)
    runt.run_with_args(stock_chip_race_open_data)
    runt.run_with_args(stock_imitup_reason_data)


# main函数入口
if __name__ == '__main__':
    main()
