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
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
import talib.abstract as ta
import math
import pandas_ta as pta
import logging
from logging import FATAL

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

logger = logging.getLogger(__name__)

class Cenderawasih_30m (IStrategy):

    def version(self) -> str:
        return "Cenderawasih-v1-30m"

    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    # Buy hyperspace params:
    buy_params = {
        "buy_length_hma": 130,
        "buy_offset_hma": 0.83,

        "buy_length_hma2": 63,
        "buy_offset_hma2": 0.85,

        "buy_length_hma3": 30,
        "buy_offset_hma3": 0.84,

        "buy_length_hma4": 39,
        "buy_offset_hma4": 0.9,

        "buy_rsi_1": 42,
        "buy_rsi_2": 48,

        "buy_min_red_2h": 6,
        "buy_min_red_2h_2": 18,
        "buy_min_red_2h_3": 17,
        "buy_min_red_2h_4": 11,

        "buy_max_red_2h": -4,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_length_ema": 5,
        "sell_offset_ema": 1.0,

        "sell_length_ema2": 102,
        "sell_offset_ema2": 0.87,

        "sell_length_ema3": 71,
        "sell_offset_ema3": 0.89,

        "sell_length_ema4": 133,
        "sell_offset_ema4": 1.17,

        "sell_long_green3": 6,
    }

    buy_rsi_1 = IntParameter(20, 50, default=50, optimize=False)
    buy_rsi_2 = IntParameter(20, 50, default=50, optimize=False)
    
    buy_min_red_2h = IntParameter(1, 30, default=1, optimize=False)
    buy_min_red_2h_2 = IntParameter(1, 30, default=1, optimize=False)
    buy_min_red_2h_3 = IntParameter(1, 30, default=1, optimize=False)
    buy_min_red_2h_4 = IntParameter(1, 30, default=1, optimize=False)
    buy_max_red_2h = IntParameter(-20, 20, default=1, optimize=False)

    optimize_buy_hma = False
    buy_length_hma = IntParameter(5, 150, default=6, optimize=optimize_buy_hma)
    buy_offset_hma = DecimalParameter(0.8, 1, default=1, decimals=2, optimize=optimize_buy_hma)

    optimize_buy_hma2 = False
    buy_length_hma2 = IntParameter(5, 150, default=6, optimize=optimize_buy_hma2)
    buy_offset_hma2 = DecimalParameter(0.8, 1, default=1, decimals=2, optimize=optimize_buy_hma2)

    optimize_buy_hma3 = False
    buy_length_hma3 = IntParameter(5, 150, default=6, optimize=optimize_buy_hma3)
    buy_offset_hma3 = DecimalParameter(0.8, 1, default=1, decimals=2, optimize=optimize_buy_hma3)

    optimize_buy_hma4 = False
    buy_length_hma4 = IntParameter(5, 150, default=6, optimize=optimize_buy_hma4)
    buy_offset_hma4 = DecimalParameter(0.8, 1, default=1, decimals=2, optimize=optimize_buy_hma4)

    optimize_sell_ema = False
    sell_length_ema = IntParameter(5, 150, default=6, optimize=optimize_sell_ema)
    sell_offset_ema = DecimalParameter(1, 1.2, default=1.02, decimals=2, optimize=optimize_sell_ema)

    optimize_sell_ema2 = False
    sell_length_ema2 = IntParameter(5, 150, default=6, optimize=optimize_sell_ema2)
    sell_offset_ema2 = DecimalParameter(0.8, 1, default=0.98, decimals=2, optimize=optimize_sell_ema2)

    optimize_sell_ema3 = False
    sell_length_ema3 = IntParameter(5, 150, default=6, optimize=optimize_sell_ema3)
    sell_offset_ema3 = DecimalParameter(0.8, 1, default=0.95, decimals=2, optimize=optimize_sell_ema3)

    optimize_sell_ema4 = False
    sell_length_ema4 = IntParameter(5, 150, default=6, optimize=optimize_sell_ema4)
    sell_offset_ema4 = DecimalParameter(1, 1.2, default=1, decimals=2, optimize=optimize_sell_ema4)

    sell_long_green3 = IntParameter(5, 40, default=10, optimize=False)

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.15
    trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    timeframe = '30m'

    process_only_new_candles = True
    startup_candle_count = 999

    timeframe_minutes = timeframe_to_minutes(timeframe)
    timeframe_minutes_string = f"{timeframe_minutes}m"
    if int(timeframe_minutes) >= 60:
        timeframe_minutes_string = f"{timeframe_minutes//60}h"

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['age_filter_ok'] = (dataframe['volume'].rolling(window=30, min_periods=30).min() > 0)

        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    @informative('2h')
    def populate_indicators_2h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['pct_change'] = dataframe['close'].pct_change()

        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    @informative(timeframe, 'BTC/{stake}', '{base}_{column}_{timeframe}')
    def populate_indicators_btc_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['pct_change'] = dataframe['close'].pct_change()
        dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

        if not self.optimize_buy_hma:
            dataframe['hma_offset_buy'] = tv_hma(dataframe, int(self.buy_length_hma.value)) *self.buy_offset_hma.value

        if not self.optimize_buy_hma2:
            dataframe['hma_offset_buy2'] = tv_hma(dataframe, int(self.buy_length_hma2.value)) *self.buy_offset_hma2.value

        if not self.optimize_buy_hma3:
            dataframe['hma_offset_buy3'] = tv_hma(dataframe, int(self.buy_length_hma3.value)) *self.buy_offset_hma3.value

        if not self.optimize_buy_hma4:
            dataframe['hma_offset_buy4'] = tv_hma(dataframe, int(self.buy_length_hma4.value)) *self.buy_offset_hma4.value

        if not self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.sell_length_ema.value)) *self.sell_offset_ema.value

        if not self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(self.sell_length_ema2.value)) *self.sell_offset_ema2.value

        if not self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(self.sell_length_ema3.value)) *self.sell_offset_ema3.value

        if not self.optimize_sell_ema4:
            dataframe['ema_offset_sell4'] = ta.EMA(dataframe, int(self.sell_length_ema4.value)) *self.sell_offset_ema4.value

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []

        if self.optimize_buy_hma:
            dataframe['hma_offset_buy'] = tv_hma(dataframe, int(self.buy_length_hma.value)) *self.buy_offset_hma.value

        if self.optimize_buy_hma2:
            dataframe['hma_offset_buy2'] = tv_hma(dataframe, int(self.buy_length_hma2.value)) *self.buy_offset_hma2.value

        if self.optimize_buy_hma3:
            dataframe['hma_offset_buy3'] = tv_hma(dataframe, int(self.buy_length_hma3.value)) *self.buy_offset_hma3.value

        if self.optimize_buy_hma4:
            dataframe['hma_offset_buy4'] = tv_hma(dataframe, int(self.buy_length_hma4.value)) *self.buy_offset_hma4.value

        dataframe['enter_tag'] = ''

        add_check = (
            dataframe['live_data_ok']
            &
            dataframe['age_filter_ok_1d']
            &
            (dataframe['close'] < dataframe['open'])
        )

        buy_offset_hma = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe['pct_change_2h'] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe['pct_change_2h'] < (-0.01 * self.buy_max_red_2h.value))
            )
            |
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe['pct_change_2h'] > (-0.01 * self.buy_min_red_2h_2.value))
            )
            |
            (
                (dataframe['close'] < dataframe['hma_offset_buy3'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 50)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 70)
                &
                (dataframe['pct_change_2h'] > (-0.01 * self.buy_min_red_2h_3.value))
            )
            |
            (
                (dataframe['close'] < dataframe['hma_offset_buy4'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 70)
                &
                (dataframe['pct_change_2h'] > (-0.01 * self.buy_min_red_2h_4.value))
            )
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

        if self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.sell_length_ema.value)) *self.sell_offset_ema.value

        if self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(self.sell_length_ema2.value)) *self.sell_offset_ema2.value

        if self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(self.sell_length_ema3.value)) *self.sell_offset_ema3.value

        if self.optimize_sell_ema4:
            dataframe['ema_offset_sell4'] = ta.EMA(dataframe, int(self.sell_length_ema4.value)) *self.sell_offset_ema4.value

        dataframe['exit_tag'] = ''
        conditions = []
        
        sell_ema_1 = (
            (dataframe['close'] > dataframe['ema_offset_sell'])
        )
        conditions.append(sell_ema_1)
        dataframe.loc[sell_ema_1, 'exit_tag'] += 'EMA_up '

        sell_ema_2 = (
            (dataframe['close'] < dataframe['ema_offset_sell2'])
        )
        conditions.append(sell_ema_2)
        dataframe.loc[sell_ema_2, 'exit_tag'] += 'EMA_down '

        sell_ema_3 = (
            (dataframe['close'] < dataframe['ema_offset_sell3']).rolling(2).min() > 0
        )
        conditions.append(sell_ema_3)
        dataframe.loc[sell_ema_3, 'exit_tag'] += 'EMA_down_2 '

        sell_ema_4 = (
            (dataframe['close'] > dataframe['ema_offset_sell4']).rolling(2).min() > 0
        )
        conditions.append(sell_ema_4)
        dataframe.loc[sell_ema_4, 'exit_tag'] += 'EMA_up_2 '

        sell_long_green3 = (
            (dataframe['pct_change'].rolling(3).sum() > (0.01 * self.sell_long_green3.value))
        )
        conditions.append(sell_long_green3)
        dataframe.loc[sell_long_green3, 'exit_tag'] += 'green_3 '

        add_check = (
            (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions) & add_check,
                'exit_long'
            ] = 1

        return dataframe
