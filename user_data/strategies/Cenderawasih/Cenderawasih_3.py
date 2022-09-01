import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter, stoploss_from_open)
from pandas import DataFrame, Series
from typing import Dict, List, Optional, Tuple
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date
import talib.abstract as ta
import math
import pandas_ta as pta
import logging
from logging import FATAL
import time
import requests
import threading

logger = logging.getLogger(__name__)

class Cenderawasih_3 (IStrategy):

    def version(self) -> str:
        return "v3"

    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 100.0
    }

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy_hma": 42,
        "low_offset_hma": 0.899,

    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell_ema": 31,
        "high_offset_ema": 1.0,

        "base_nb_candles_sell_ema2": 87,
        "high_offset_ema2": 0.84,

        "base_nb_candles_sell_ema3": 98,
        "high_offset_ema3": 0.969,

        "base_nb_candles_sell_ema4": 91,
        "high_offset_ema4": 1.113,

        "base_nb_candles_sell_zema": 93,
        "high_offset_zema": 1.089,
        
        "base_nb_candles_sell_zema2": 57,
        "high_offset_zema2": 0.879,
    }
    
    optimize_buy_hma = False
    base_nb_candles_buy_hma = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_hma)
    low_offset_hma = DecimalParameter(0.6, 0.99, default=0.95, space='buy', optimize=optimize_buy_hma)

    optimize_sell_zema = False
    base_nb_candles_sell_zema = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_zema)
    high_offset_zema = DecimalParameter(1, 1.2, default=1, space='sell', optimize=optimize_sell_zema)

    optimize_sell_zema2 = False
    base_nb_candles_sell_zema2 = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_zema2)
    high_offset_zema2 = DecimalParameter(0.6, 0.99, default=0.95, space='sell', optimize=optimize_sell_zema2)

    optimize_sell_ema = False
    base_nb_candles_sell_ema = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema)
    high_offset_ema = DecimalParameter(1, 1.2, default=1, space='sell', optimize=optimize_sell_ema)

    optimize_sell_ema2 = False
    base_nb_candles_sell_ema2 = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema2)
    high_offset_ema2 = DecimalParameter(0.7, 0.99, default=0.95, space='sell', optimize=optimize_sell_ema2)

    optimize_sell_ema3 = False
    base_nb_candles_sell_ema3 = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema3)
    high_offset_ema3 = DecimalParameter(0.7, 0.99, default=0.95, space='sell', optimize=optimize_sell_ema3)

    optimize_sell_ema4 = False
    base_nb_candles_sell_ema4 = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema4)
    high_offset_ema4 = DecimalParameter(1, 1.2, default=1, space='sell', optimize=optimize_sell_ema4)

    # Stoploss:
    stoploss = -0.1

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 200


    age_filter = 30

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['age_filter_ok'] = (dataframe['volume'].rolling(window=self.age_filter, min_periods=self.age_filter).min() > 0)

        if not self.config['runmode'].value in ('dry_run', 'live'):
            drop_columns = ['open', 'high', 'low', 'close', 'volume']
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

        if not self.optimize_buy_hma:
            dataframe['hma_offset_buy'] = tv_hma(dataframe, int(self.base_nb_candles_buy_hma.value)) *self.low_offset_hma.value

        if not self.optimize_sell_zema:
            dataframe['zema_offset_sell'] = zema(dataframe, int(self.base_nb_candles_sell_zema.value)) *self.high_offset_zema.value

        if not self.optimize_sell_zema2:
            dataframe['zema_offset_sell2'] = zema(dataframe, int(self.base_nb_candles_sell_zema2.value)) *self.high_offset_zema2.value

        if not self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema.value)) *self.high_offset_ema.value

        if not self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema2.value)) *self.high_offset_ema2.value

        if not self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema3.value)) *self.high_offset_ema3.value

        if not self.optimize_sell_ema4:
            dataframe['ema_offset_sell4'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema4.value)) *self.high_offset_ema4.value

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []

        if self.optimize_buy_hma:
            dataframe['hma_offset_buy'] = tv_hma(dataframe, int(self.base_nb_candles_buy_hma.value)) *self.low_offset_hma.value

        dataframe.loc[:, 'enter_tag'] = ''
        dataframe.loc[:, 'enter_long'] = 0

        add_check = (
            dataframe['live_data_ok']
            &
            dataframe['age_filter_ok_1d']
        )

        buy_offset_hma = (
            (dataframe['close'] < dataframe['hma_offset_buy'])
        )
        dataframe.loc[buy_offset_hma, 'enter_tag'] += 'hma '
        conditions.append(buy_offset_hma)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions)
                &
                add_check,
                'enter_long',
            ]= 1


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.optimize_sell_zema:
            dataframe['zema_offset_sell'] = zema(dataframe, int(self.base_nb_candles_sell_zema.value)) *self.high_offset_zema.value

        if self.optimize_sell_zema2:
            dataframe['zema_offset_sell2'] = zema(dataframe, int(self.base_nb_candles_sell_zema2.value)) *self.high_offset_zema2.value

        if self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema.value)) *self.high_offset_ema.value

        if self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema2.value)) *self.high_offset_ema2.value

        if self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema3.value)) *self.high_offset_ema3.value

        if self.optimize_sell_ema4:
            dataframe['ema_offset_sell4'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema4.value)) *self.high_offset_ema4.value

        dataframe.loc[:, 'exit_tag'] = ''
        conditions = []
        
        sell_zema_1 = (
            (dataframe['close'] > dataframe['zema_offset_sell'])
        )
        conditions.append(sell_zema_1)
        dataframe.loc[sell_zema_1, 'exit_tag'] += 'ZEMA_1 '

        sell_zema_2 = (
            (dataframe['close'] < dataframe['zema_offset_sell2'])
        )
        conditions.append(sell_zema_2)
        dataframe.loc[sell_zema_2, 'exit_tag'] += 'ZEMA_2 '

        sell_ema_1 = (
            (dataframe['close'] > dataframe['ema_offset_sell'])
        )
        conditions.append(sell_ema_1)
        dataframe.loc[sell_ema_1, 'exit_tag'] += 'EMA_1 '

        sell_ema_2 = (
            (dataframe['close'] < dataframe['ema_offset_sell2']).rolling(2).min() > 0
        )
        conditions.append(sell_ema_2)
        dataframe.loc[sell_ema_2, 'exit_tag'] += 'EMA_2 '

        sell_ema_3 = (
            (dataframe['close'] < dataframe['ema_offset_sell3'])
        )
        conditions.append(sell_ema_3)
        dataframe.loc[sell_ema_3, 'exit_tag'] += 'EMA_3 '

        sell_ema_4 = (
            (dataframe['close'] > dataframe['ema_offset_sell4']).rolling(2).min() > 0
        )
        conditions.append(sell_ema_4)
        dataframe.loc[sell_ema_4, 'exit_tag'] += 'EMA_4 '

        add_check = (
            (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions) & add_check,
                'exit_long'
            ] = 1

        return dataframe

def tv_wma(df, length = 9) -> DataFrame:
    """
    Source: Tradingview "Moving Average Weighted"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : WMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_wma'
    """

    norm = 0
    sum = 0

    for i in range(1, length - 1):
        weight = (length - i) * length
        norm = norm + weight
        sum = sum + df.shift(i) * weight

    tv_wma = (sum / norm) if norm > 0 else 0
    return tv_wma

def tv_hma(dataframe, length = 9, field = 'close') -> DataFrame:
    """
    Source: Tradingview "Hull Moving Average"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : HMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_hma'
    """

    h = 2 * tv_wma(dataframe[field], math.floor(length / 2)) - tv_wma(dataframe[field], length)

    tv_hma = tv_wma(h, math.floor(math.sqrt(length)))
    # dataframe.drop("h", inplace=True, axis=1)

    return tv_hma

def zema(dataframe, period, field='close'):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/overlap_studies.py#L79
    Modified slightly to use ta.EMA instead of technical ema
    """
    df = dataframe.copy()

    df['ema1'] = ta.EMA(df[field], timeperiod=period)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=period)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']

    return df['zema']
