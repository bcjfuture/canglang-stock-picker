#!/usr/local/bin/python3
# -*- coding: utf-8 -*-


import logging
import pandas as pd
import os.path
import sys

cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
import instock.lib.run_template as runt
import instock.core.tablestructure as tbs
import instock.lib.database as mdb
import instock.core.stockfetch as stf

__author__ = 'myh '
__date__ = '2023/5/5 '


def save_nph_stock_selection_data(date, before=True):
    if before:
        return

    try:
        data = stf.fetch_stock_selection()
        if data is None:
            return

        table_name = tbs.TABLE_CN_STOCK_SELECTION['name']
        partition_date = data.iloc[0]['date']
        mdb.replace_partition_from_df(
            data=data,
            table_name=table_name,
            partition_column="date",
            partition_value=partition_date,
            key_columns=["date", "code"],
            cols_type=tbs.get_field_types(tbs.TABLE_CN_STOCK_SELECTION['columns']),
            primary_keys="`date`,`code`",
        )
    except Exception as e:
        logging.error(f"DAILY_UPDATE_WRITE_FAILED selection_data_daily_job.save_nph_stock_selection_data：{e}")


def main():
    runt.run_with_args(save_nph_stock_selection_data)


# main函数入口
if __name__ == '__main__':
    main()
