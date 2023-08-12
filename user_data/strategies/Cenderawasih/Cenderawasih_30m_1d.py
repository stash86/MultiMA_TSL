import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter, stoploss_from_open)
from pandas import DataFrame, Series
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
import talib.abstract as ta
import math
import pandas_ta as pta
# from finta import TA as fta
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

class Cenderawasih_30m_1d (IStrategy):

    def version(self) -> str:
        return "Cenderawasih-v1d-30m"

    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    # Buy hyperspace params:
    buy_params = {
        "buy_length_hma": 130,
        "buy_offset_hma": 0.83,

        "buy_length_hma1a": 23,
        "buy_offset_hma1a": 16,

        "buy_length_hma1b": 6,
        "buy_offset_hma1b": 17,

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
        "sell_length_ema": 1,
        "sell_offset_ema": 20,

        "sell_length_ema2": 102,
        "sell_offset_ema2": 0.87,

        "sell_length_ema2a": 2,
        "sell_offset_ema2a": 16,

        "sell_length_ema2b": 1,
        "sell_offset_ema2b": 19,

        "sell_length_ema3": 71,
        "sell_offset_ema3": 0.89,

        "sell_length_ema3a": 6,
        "sell_offset_ema3a": 17,

        "sell_length_ema4": 133,
        "sell_offset_ema4": 1.17,

        "sell_long_green3": 6,
        "sell_long_green3a": 17,
        "sell_long_green3b": 0,
    }

    buy_dummy = IntParameter(20, 70, default=61, optimize=True)

    buy_rsi_1 = IntParameter(20, 50, default=50, optimize=False)
    buy_rsi_2 = IntParameter(20, 50, default=50, optimize=False)
    # buy_rsi_3 = IntParameter(20, 50, default=50, optimize=True)
    # buy_rsi_4 = IntParameter(20, 50, default=50, optimize=False)

    # buy_long_red2_2 = IntParameter(1, 20, default=1, optimize=True)
    # buy_long_red2_3 = IntParameter(1, 20, default=1, optimize=False)
    # buy_long_red2_4 = IntParameter(1, 10, default=1, optimize=False)
    
    buy_min_red_2h = IntParameter(1, 30, default=1, optimize=False)
    buy_min_red_2h_2 = IntParameter(1, 30, default=1, optimize=False)
    buy_min_red_2h_3 = IntParameter(1, 30, default=1, optimize=False)
    buy_min_red_2h_4 = IntParameter(1, 30, default=1, optimize=False)
    buy_max_red_2h = IntParameter(-20, 20, default=1, optimize=False)
    # buy_max_red_2h_2 = IntParameter(-20, 20, default=1, optimize=True)
    # buy_max_red_2h_3 = IntParameter(-20, 20, default=1, optimize=True)
    # buy_max_red_2h_4 = IntParameter(-20, 20, default=1, optimize=True)
    # buy_min_red_3_2h = IntParameter(1, 50, default=5, optimize=True)
    # buy_min_red_4_2h = IntParameter(11, 60, default=15, optimize=True)

    # buy_lowest_open_rolling = IntParameter(1, 10, default=1, optimize=False)
    # buy_lowest_close_rolling = IntParameter(1, 10, default=1, optimize=False)

    optimize_buy_hma = False
    buy_length_hma = IntParameter(5, 150, default=6, optimize=optimize_buy_hma)
    buy_offset_hma = DecimalParameter(0.8, 1, default=1, decimals=2, optimize=optimize_buy_hma)

    optimize_buy_hma1a = False
    buy_length_hma1a = IntParameter(1, 30, default=6, optimize=optimize_buy_hma1a)
    buy_offset_hma1a = IntParameter(16, 20, default=20, optimize=optimize_buy_hma1a)

    optimize_buy_hma1b = False
    buy_length_hma1b = IntParameter(1, 30, default=6, optimize=optimize_buy_hma1b)
    buy_offset_hma1b = IntParameter(16, 20, default=20, optimize=optimize_buy_hma1b)

    # buy_red_candle_rolling = IntParameter(1, 5, default=2, optimize=False)

    optimize_buy_hma2 = False
    buy_length_hma2 = IntParameter(5, 150, default=6, optimize=optimize_buy_hma2)
    buy_offset_hma2 = DecimalParameter(0.8, 1, default=1, decimals=2, optimize=optimize_buy_hma2)

    # optimize_buy_hma2a = True
    # buy_length_hma2a = IntParameter(1, 30, default=6, optimize=optimize_buy_hma2a)
    # buy_offset_hma2a = IntParameter(16, 20, default=20, optimize=optimize_buy_hma2a)

    optimize_buy_hma3 = False
    buy_length_hma3 = IntParameter(5, 150, default=6, optimize=optimize_buy_hma3)
    buy_offset_hma3 = DecimalParameter(0.8, 1, default=1, decimals=2, optimize=optimize_buy_hma3)

    # optimize_buy_hma3b = True
    # buy_length_hma3b = IntParameter(1, 30, default=6, optimize=optimize_buy_hma3b)
    # buy_offset_hma3b = IntParameter(16, 20, default=20, optimize=optimize_buy_hma3b)

    optimize_buy_hma4 = False
    buy_length_hma4 = IntParameter(5, 150, default=6, optimize=optimize_buy_hma4)
    buy_offset_hma4 = DecimalParameter(0.8, 1, default=1, decimals=2, optimize=optimize_buy_hma4)

    optimize_sell_ema = False
    sell_length_ema = IntParameter(1, 30, default=6, optimize=optimize_sell_ema)
    sell_offset_ema = IntParameter(20, 24, default=20, optimize=optimize_sell_ema)

    optimize_sell_ema2 = False
    sell_length_ema2 = IntParameter(5, 150, default=6, optimize=optimize_sell_ema2)
    sell_offset_ema2 = DecimalParameter(0.8, 1, default=0.98, decimals=2, optimize=optimize_sell_ema2)

    optimize_sell_ema2a = False
    sell_length_ema2a = IntParameter(1, 30, default=6, optimize=optimize_sell_ema2a)
    sell_offset_ema2a = IntParameter(16, 20, default=20, optimize=optimize_sell_ema2a)

    optimize_sell_ema2b = False
    sell_length_ema2b = IntParameter(1, 50, default=6, optimize=optimize_sell_ema2b)
    sell_offset_ema2b = IntParameter(16, 20, default=20, optimize=optimize_sell_ema2b)

    optimize_sell_ema3 = False
    sell_length_ema3 = IntParameter(5, 150, default=6, optimize=optimize_sell_ema3)
    sell_offset_ema3 = DecimalParameter(0.8, 1, default=0.95, decimals=2, optimize=optimize_sell_ema3)

    optimize_sell_ema3a = False
    sell_length_ema3a = IntParameter(1, 30, default=6, optimize=optimize_sell_ema3a)
    sell_offset_ema3a = IntParameter(16, 20, default=20, optimize=optimize_sell_ema3a)

    optimize_sell_ema4 = False
    sell_length_ema4 = IntParameter(5, 150, default=6, optimize=optimize_sell_ema4)
    sell_offset_ema4 = DecimalParameter(1, 1.2, default=1, decimals=2, optimize=optimize_sell_ema4)

    # optimize_sell_rsi = False
    # sell_rsi_overbought = IntParameter(40, 90, default=50, optimize=True)

    # sell_long_green = IntParameter(1, 50, default=1, optimize=False)
    sell_long_green3 = IntParameter(5, 40, default=10, optimize=False)
    # sell_long_green6 = IntParameter(-5, 40, default=10, optimize=True)
    sell_long_green3b = IntParameter(-5, 40, default=10, optimize=False)
    # sell_long_red3 = IntParameter(0, 50, default=1, optimize=True)

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
    inf_timeframe1 = '2h'

    process_only_new_candles = True
    startup_candle_count = 999

    timeframe_minutes = timeframe_to_minutes(timeframe)
    timeframe_minutes_string = f"{timeframe_minutes}m"
    if int(timeframe_minutes) >= 60:
        timeframe_minutes_string = f"{timeframe_minutes//60}h"

    inf_timeframe1_minutes = timeframe_to_minutes(inf_timeframe1)
    inf_timeframe1_minutes_string = f"{inf_timeframe1_minutes}m"
    if int(inf_timeframe1_minutes) >= 60:
        inf_timeframe1_minutes_string = f"{inf_timeframe1_minutes//60}h"

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['age_filter_ok'] = (dataframe['volume'].rolling(window=30, min_periods=30).min() > 0)

        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    @informative(inf_timeframe1)
    def populate_indicators_inf(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['pct_change'] = dataframe['close'].pct_change()

        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    @informative(timeframe, 'BTC/{stake}', '{base}_{column}_{timeframe}')
    # @informative('15m', 'BTC/USDT', '{base}_{column}_{timeframe}')
    def populate_indicators_btc_inf(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)

        dataframe['pct_change'] = dataframe['close'].pct_change()

        dataframe['vrsi'] = ta.RSI(dataframe['volume'], timeperiod=15)
        dataframe['vrsi_45'] = ta.RSI(dataframe['volume'], timeperiod=45)

        dataframe['close_mean_75'] = dataframe['close'].rolling(75).mean()
        dataframe['close_median_75'] = dataframe['close'].rolling(75).median()
        dataframe['close_mean_150'] = dataframe['close'].rolling(150).mean()
        dataframe['close_median_150'] = dataframe['close'].rolling(150).median()
        dataframe['close_mean_300'] = dataframe['close'].rolling(300).mean()
        dataframe['close_median_300'] = dataframe['close'].rolling(300).median()

        dataframe['mfi'] = ta.MFI(dataframe, 15)
        dataframe['mfi_45'] = ta.MFI(dataframe, 45)

        dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

        if not self.optimize_buy_hma:
            dataframe['hma_offset_buy1'] = tv_hma(dataframe, int(self.buy_length_hma.value)) *self.buy_offset_hma.value

        if not self.optimize_buy_hma1a:
            dataframe['hma_offset_buy1a'] = tv_hma(dataframe, int(5 * self.buy_length_hma1a.value)) * 0.05 * self.buy_offset_hma1a.value

        if not self.optimize_buy_hma1b:
            dataframe['hma_offset_buy1b'] = tv_hma(dataframe, int(5 * self.buy_length_hma1b.value)) * 0.05 * self.buy_offset_hma1b.value

        if not self.optimize_buy_hma2:
            dataframe['hma_offset_buy2'] = tv_hma(dataframe, int(self.buy_length_hma2.value)) *self.buy_offset_hma2.value

        # if not self.optimize_buy_hma2a:
        #     dataframe['hma_offset_buy2a'] = tv_hma(dataframe, int(5 * self.buy_length_hma2a.value)) * 0.05 * self.buy_offset_hma2a.value

        if not self.optimize_buy_hma3:
            dataframe['hma_offset_buy3'] = tv_hma(dataframe, int(self.buy_length_hma3.value)) *self.buy_offset_hma3.value

        # if not self.optimize_buy_hma3b:
        #     dataframe['hma_offset_buy3b'] = tv_hma(dataframe, int(5 * self.buy_length_hma3b.value)) * 0.05 * self.buy_offset_hma3b.value

        if not self.optimize_buy_hma4:
            dataframe['hma_offset_buy4'] = tv_hma(dataframe, int(self.buy_length_hma4.value)) *self.buy_offset_hma4.value

        if not self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(5 * self.sell_length_ema.value)) * 0.05 * self.sell_offset_ema.value

        if not self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(self.sell_length_ema2.value)) *self.sell_offset_ema2.value

        if not self.optimize_sell_ema2a:
            dataframe['ema_offset_sell2a'] = ta.EMA(dataframe, int(5 * self.sell_length_ema2a.value)) * 0.05 * self.sell_offset_ema2a.value

        if not self.optimize_sell_ema2b:
            dataframe['ema_offset_sell2b'] = ta.EMA(dataframe, int(5 * self.sell_length_ema2b.value)) * 0.05 * self.sell_offset_ema2b.value

        if not self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(self.sell_length_ema3.value)) *self.sell_offset_ema3.value

        if not self.optimize_sell_ema3a:
            dataframe['ema_offset_sell3a'] = ta.EMA(dataframe, int(5 * self.sell_length_ema3a.value)) * 0.05 * self.sell_offset_ema3a.value

        if not self.optimize_sell_ema4:
            dataframe['ema_offset_sell4'] = ta.EMA(dataframe, int(self.sell_length_ema4.value)) *self.sell_offset_ema4.value

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []

        if self.optimize_buy_hma:
            dataframe['hma_offset_buy1'] = tv_hma(dataframe, int(self.buy_length_hma.value)) *self.buy_offset_hma.value

        if self.optimize_buy_hma1a:
            dataframe['hma_offset_buy1a'] = tv_hma(dataframe, int(5 * self.buy_length_hma1a.value)) * 0.05 * self.buy_offset_hma1a.value

        if self.optimize_buy_hma1b:
            dataframe['hma_offset_buy1b'] = tv_hma(dataframe, int(5 * self.buy_length_hma1b.value)) * 0.05 * self.buy_offset_hma1b.value

        if self.optimize_buy_hma2:
            dataframe['hma_offset_buy2'] = tv_hma(dataframe, int(self.buy_length_hma2.value)) *self.buy_offset_hma2.value

        # if self.optimize_buy_hma2a:
        #     dataframe['hma_offset_buy2a'] = tv_hma(dataframe, int(5 * self.buy_length_hma2a.value)) * 0.05 * self.buy_offset_hma2a.value

        if self.optimize_buy_hma3:
            dataframe['hma_offset_buy3'] = tv_hma(dataframe, int(self.buy_length_hma3.value)) *self.buy_offset_hma3.value

        # if self.optimize_buy_hma3b:
        #     dataframe['hma_offset_buy3b'] = tv_hma(dataframe, int(5 * self.buy_length_hma3b.value)) * 0.05 * self.buy_offset_hma3b.value

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

        mean_above_median_75 = np.where((dataframe['close_mean_75'] >= dataframe['close_median_75']), True, False)

        mean_above_median_150 = np.where((dataframe['close_mean_150'] >= dataframe['close_median_150']), True, False)
        close_above_median_150 = np.where((dataframe['close'] >= dataframe['close_median_150']), True, False)
        close_above_mean_150 = np.where((dataframe['close'] >= dataframe['close_mean_150']), True, False)
        
        mean_above_median_300 = np.where((dataframe['close_mean_300'] >= dataframe['close_median_300']), True, False)
        close_above_median_300 = np.where((dataframe['close'] >= dataframe['close_median_300']), True, False)
        close_above_mean_300 = np.where((dataframe['close'] >= dataframe['close_mean_300']), True, False)

        mean_75_above_mean_150 = np.where((dataframe['close_mean_75'] >= dataframe['close_mean_150']), True, False)
        mean_75_above_mean_300 = np.where((dataframe['close_mean_75'] >= dataframe['close_mean_300']), True, False)
        mean_150_above_mean_300 = np.where((dataframe['close_mean_150'] >= dataframe['close_mean_300']), True, False)

        median_75_above_median_150 = np.where((dataframe['close_median_75'] >= dataframe['close_median_150']), True, False)
        median_75_above_median_300 = np.where((dataframe['close_median_75'] >= dataframe['close_median_300']), True, False)
        median_150_above_median_300 = np.where((dataframe['close_median_150'] >= dataframe['close_median_300']), True, False)

        buy_offset_hma1a1a1a1a = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1b'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                mean_above_median_150
                &
                close_above_median_150
                &
                close_above_mean_150
                &
                median_150_above_median_300
            )
        )
        dataframe.loc[buy_offset_hma1a1a1a1a, 'enter_tag'] += 'hma1a1a1a1a '
        conditions.append(buy_offset_hma1a1a1a1a)

        buy_offset_hma1a1a1a1b1 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1a'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                mean_above_median_150
                &
                close_above_median_150
                &
                close_above_mean_150
                &
                ~median_150_above_median_300
                &
                mean_150_above_mean_300
            )
        )
        dataframe.loc[buy_offset_hma1a1a1a1b1, 'enter_tag'] += 'hma1a1a1a1b1 '
        conditions.append(buy_offset_hma1a1a1a1b1)

        buy_offset_hma1a1a1a2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                mean_above_median_150
                &
                close_above_median_150
                &
                ~close_above_mean_150
            )
        )
        dataframe.loc[buy_offset_hma1a1a1a2, 'enter_tag'] += 'hma1a1a1a2 '
        conditions.append(buy_offset_hma1a1a1a2)

        buy_offset_hma1a1a1b = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                mean_above_median_150
                &
                ~close_above_median_150
            )
        )
        dataframe.loc[buy_offset_hma1a1a1b, 'enter_tag'] += 'hma1a1a1b '
        conditions.append(buy_offset_hma1a1a1b)

        buy_offset_hma1a1a2a = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                ~mean_above_median_150
                &
                close_above_median_150
            )
        )
        dataframe.loc[buy_offset_hma1a1a2a, 'enter_tag'] += 'hma1a1a2a '
        conditions.append(buy_offset_hma1a1a2a)

        buy_offset_hma1a1a2b1 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                ~mean_above_median_150
                &
                ~close_above_median_150
                &
                mean_150_above_mean_300
            )
        )
        dataframe.loc[buy_offset_hma1a1a2b1, 'enter_tag'] += 'hma1a1a2b1 '
        conditions.append(buy_offset_hma1a1a2b1)

        buy_offset_hma1a1a2b2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                ~mean_above_median_150
                &
                ~close_above_median_150
                &
                ~mean_150_above_mean_300
            )
        )
        dataframe.loc[buy_offset_hma1a1a2b2, 'enter_tag'] += 'hma1a1a2b2 '
        conditions.append(buy_offset_hma1a1a2b2)

        buy_offset_hma1a1b = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                ~close_above_mean_300
            )
        )
        dataframe.loc[buy_offset_hma1a1b, 'enter_tag'] += 'hma1a1b '
        conditions.append(buy_offset_hma1a1b)

        buy_offset_hma1a2a = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
                &
                close_above_mean_300
            )
        )
        dataframe.loc[buy_offset_hma1a2a, 'enter_tag'] += 'hma1a2a '
        conditions.append(buy_offset_hma1a2a)

        buy_offset_hma1a2b1 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
                &
                ~close_above_mean_300
                &
                mean_above_median_150
            )
        )
        dataframe.loc[buy_offset_hma1a2b1, 'enter_tag'] += 'hma1a2b1 '
        conditions.append(buy_offset_hma1a2b1)

        buy_offset_hma1a2b2a1 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
                &
                ~close_above_mean_300
                &
                ~mean_above_median_150
                &
                mean_150_above_mean_300
                &
                mean_75_above_mean_150
            )
        )
        dataframe.loc[buy_offset_hma1a2b2a1, 'enter_tag'] += 'hma1a2b2a1 '
        conditions.append(buy_offset_hma1a2b2a1)

        buy_offset_hma1a2b2a2b = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
                &
                ~close_above_mean_300
                &
                ~mean_above_median_150
                &
                mean_150_above_mean_300
                &
                ~mean_75_above_mean_150
                &
                ~mean_75_above_mean_300
            )
        )
        dataframe.loc[buy_offset_hma1a2b2a2b, 'enter_tag'] += 'hma1a2b2a2b '
        conditions.append(buy_offset_hma1a2b2a2b)

        buy_offset_hma1a2b2b = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
                &
                ~close_above_mean_300
                &
                ~mean_above_median_150
                &
                ~mean_150_above_mean_300
            )
        )
        dataframe.loc[buy_offset_hma1a2b2b, 'enter_tag'] += 'hma1a2b2b '
        conditions.append(buy_offset_hma1a2b2b)

        buy_offset_hma1b = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy1'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 30)
                &
                (dataframe['rsi'] < self.buy_rsi_1.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h.value))
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] < (-0.01 * self.buy_max_red_2h.value))
                &
                ~mean_above_median_300
            )
        )
        dataframe.loc[buy_offset_hma1b, 'enter_tag'] += 'hma1b '
        conditions.append(buy_offset_hma1b)

        buy_offset_hma2a1a1b = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                mean_above_median_150
                &
                ~close_above_median_150
            )
        )
        dataframe.loc[buy_offset_hma2a1a1b, 'enter_tag'] += 'hma2a1a1b '
        conditions.append(buy_offset_hma2a1a1b)

        buy_offset_hma2a1a2a = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                ~mean_above_median_150
                &
                close_above_median_150
            )
        )
        dataframe.loc[buy_offset_hma2a1a2a, 'enter_tag'] += 'hma2a1a2a '
        conditions.append(buy_offset_hma2a1a2a)

        buy_offset_hma2a2a1 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
                &
                close_above_mean_300
                &
                mean_above_median_150
            )
        )
        dataframe.loc[buy_offset_hma2a2a1, 'enter_tag'] += 'hma2a2a1 '
        conditions.append(buy_offset_hma2a2a1)

        buy_offset_hma2a2a2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
                &
                close_above_mean_300
                &
                ~mean_above_median_150
            )
        )
        dataframe.loc[buy_offset_hma2a2a2, 'enter_tag'] += 'hma2a2a2 '
        conditions.append(buy_offset_hma2a2a2)

        buy_offset_hma2a2b1a2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
                &
                ~close_above_mean_300
                &
                mean_above_median_150
                &
                close_above_median_150
                &
                ~close_above_mean_150
            )
        )
        dataframe.loc[buy_offset_hma2a2b1a2, 'enter_tag'] += 'hma2a2b1a2 '
        conditions.append(buy_offset_hma2a2b1a2)

        buy_offset_hma2a2b2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
                &
                ~close_above_mean_300
                &
                ~mean_above_median_150
            )
        )
        dataframe.loc[buy_offset_hma2a2b2, 'enter_tag'] += 'hma2a2b2 '
        conditions.append(buy_offset_hma2a2b2)

        buy_offset_hma2b1a1a2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                ~mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                mean_above_median_150
                &
                close_above_median_150
                &
                ~close_above_mean_150
            )
        )
        dataframe.loc[buy_offset_hma2b1a1a2, 'enter_tag'] += 'hma2b1a1a2 '
        conditions.append(buy_offset_hma2b1a1a2)

        buy_offset_hma2b1b = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                ~mean_above_median_300
                &
                close_above_median_300
                &
                ~close_above_mean_300
            )
        )
        dataframe.loc[buy_offset_hma2b1b, 'enter_tag'] += 'hma2b1b '
        conditions.append(buy_offset_hma2b1b)

        buy_offset_hma2b2a = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                ~mean_above_median_300
                &
                ~close_above_median_300
                &
                close_above_mean_300
            )
        )
        dataframe.loc[buy_offset_hma2b2a, 'enter_tag'] += 'hma2b2a '
        conditions.append(buy_offset_hma2b2a)

        buy_offset_hma2b2b1b = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                ~mean_above_median_300
                &
                ~close_above_median_300
                &
                ~close_above_mean_300
                &
                mean_above_median_150
                &
                ~mean_75_above_mean_150
            )
        )
        dataframe.loc[buy_offset_hma2b2b1b, 'enter_tag'] += 'hma2b2b1b '
        conditions.append(buy_offset_hma2b2b1b)

        buy_offset_hma2b2b2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy2'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 30)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 50)
                &
                (dataframe['rsi'] < self.buy_rsi_2.value)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_2.value))
                &
                ~mean_above_median_300
                &
                ~close_above_median_300
                &
                ~close_above_mean_300
                &
                ~mean_above_median_150
            )
        )
        dataframe.loc[buy_offset_hma2b2b2, 'enter_tag'] += 'hma2b2b2 '
        conditions.append(buy_offset_hma2b2b2)

        buy_offset_hma3a1 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy3'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 50)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 70)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_3.value))
                &
                mean_above_median_300
                &
                close_above_median_300
            )
        )
        dataframe.loc[buy_offset_hma3a1, 'enter_tag'] += 'hma3a1 '
        conditions.append(buy_offset_hma3a1)

        buy_offset_hma3a2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy3'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 50)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 70)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_3.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
            )
        )
        dataframe.loc[buy_offset_hma3a2, 'enter_tag'] += 'hma3a2 '
        conditions.append(buy_offset_hma3a2)

        # buy_offset_hma3b1 = (
        #     (
        #         (dataframe['close'] < dataframe['hma_offset_buy3'])
        #         &
        #         (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 50)
        #         &
        #         (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 70)
        #         &
        #         (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_3.value))
        #         &
        #         ~mean_above_median_300
        #         &
        #         (dataframe['close'] > dataframe['close_median_300'])
        #     )
        # )
        # dataframe.loc[buy_offset_hma3b1, 'enter_tag'] += 'hma3b1 '
        # conditions.append(buy_offset_hma3b1)

        buy_offset_hma3b2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy3'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 50)
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] < 70)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_3.value))
                &
                ~mean_above_median_300
                &
                ~close_above_median_300
            )
        )
        dataframe.loc[buy_offset_hma3b2, 'enter_tag'] += 'hma3b2 '
        conditions.append(buy_offset_hma3b2)

        buy_offset_hma4a1a1a = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy4'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 70)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_4.value))
                &
                mean_above_median_300
                &
                close_above_median_300
                &
                close_above_mean_300
                &
                mean_150_above_mean_300
                &
                mean_75_above_mean_150
            )
        )
        dataframe.loc[buy_offset_hma4a1a1a, 'enter_tag'] += 'hma4a1a1a '
        conditions.append(buy_offset_hma4a1a1a)

        # buy_offset_hma4a1a1b = (
        #     (
        #         (dataframe['close'] < dataframe['hma_offset_buy4'])
        #         &
        #         (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 70)
        #         &
        #         (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_4.value))
        #         &
        #         mean_above_median_300
        #         &
        #         close_above_median_300
        #         &
        #         close_above_mean_300
        #         &
        #         mean_150_above_mean_300
        #         &
        #         ~mean_75_above_mean_150
        #     )
        # )
        # dataframe.loc[buy_offset_hma4a1a1b, 'enter_tag'] += 'hma4a1a1b '
        # conditions.append(buy_offset_hma4a1a1b)

        # buy_offset_hma4a1a2 = (
        #     (
        #         (dataframe['close'] < dataframe['hma_offset_buy4'])
        #         &
        #         (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 70)
        #         &
        #         (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_4.value))
        #         &
        #         mean_above_median_300
        #         &
        #         close_above_median_300
        #         &
        #         close_above_mean_300
        #         &
        #         ~mean_150_above_mean_300
        #     )
        # )
        # dataframe.loc[buy_offset_hma4a1a2, 'enter_tag'] += 'hma4a1a2 '
        # conditions.append(buy_offset_hma4a1a2)

        buy_offset_hma4a2 = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy4'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 70)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_4.value))
                &
                mean_above_median_300
                &
                ~close_above_median_300
            )
        )
        dataframe.loc[buy_offset_hma4a2, 'enter_tag'] += 'hma4a2 '
        conditions.append(buy_offset_hma4a2)

        buy_offset_hma4b = (
            (
                (dataframe['close'] < dataframe['hma_offset_buy4'])
                &
                (dataframe[f"btc_rsi_{self.timeframe_minutes_string}"] >= 70)
                &
                (dataframe[f"pct_change_{self.inf_timeframe1_minutes_string}"] > (-0.01 * self.buy_min_red_2h_4.value))
                &
                ~mean_above_median_300
            )
        )
        dataframe.loc[buy_offset_hma4b, 'enter_tag'] += 'hma4b '
        conditions.append(buy_offset_hma4b)

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
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(5 * self.sell_length_ema.value)) * 0.05 * self.sell_offset_ema.value

        if self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(self.sell_length_ema2.value)) *self.sell_offset_ema2.value

        if self.optimize_sell_ema2a:
            dataframe['ema_offset_sell2a'] = ta.EMA(dataframe, int(5 * self.sell_length_ema2a.value)) * 0.05 * self.sell_offset_ema2a.value

        if self.optimize_sell_ema2b:
            dataframe['ema_offset_sell2b'] = ta.EMA(dataframe, int(5 * self.sell_length_ema2b.value)) * 0.05 * self.sell_offset_ema2b.value

        if self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(self.sell_length_ema3.value)) *self.sell_offset_ema3.value

        if self.optimize_sell_ema3a:
            dataframe['ema_offset_sell3a'] = ta.EMA(dataframe, int(5 * self.sell_length_ema3a.value)) * 0.05 * self.sell_offset_ema3a.value

        if self.optimize_sell_ema4:
            dataframe['ema_offset_sell4'] = ta.EMA(dataframe, int(self.sell_length_ema4.value)) *self.sell_offset_ema4.value

        dataframe['exit_tag'] = ''
        conditions = []
        
        mean_above_median_75 = np.where((dataframe['close_mean_75'] >= dataframe['close_median_75']), True, False)

        mean_above_median_150 = np.where((dataframe['close_mean_150'] >= dataframe['close_median_150']), True, False)
        close_above_median_150 = np.where((dataframe['close'] >= dataframe['close_median_150']), True, False)
        close_above_mean_150 = np.where((dataframe['close'] >= dataframe['close_mean_150']), True, False)
        
        mean_above_median_300 = np.where((dataframe['close_mean_300'] >= dataframe['close_median_300']), True, False)
        close_above_median_300 = np.where((dataframe['close'] >= dataframe['close_median_300']), True, False)
        close_above_mean_300 = np.where((dataframe['close'] >= dataframe['close_mean_300']), True, False)

        mean_75_above_mean_150 = np.where((dataframe['close_mean_75'] >= dataframe['close_mean_150']), True, False)
        mean_75_above_mean_300 = np.where((dataframe['close_mean_75'] >= dataframe['close_mean_300']), True, False)
        mean_150_above_mean_300 = np.where((dataframe['close_mean_150'] >= dataframe['close_mean_300']), True, False)

        median_75_above_median_150 = np.where((dataframe['close_median_75'] >= dataframe['close_median_150']), True, False)
        median_75_above_median_300 = np.where((dataframe['close_median_75'] >= dataframe['close_median_300']), True, False)
        median_150_above_median_300 = np.where((dataframe['close_median_150'] >= dataframe['close_median_300']), True, False)

        strong_uptrend = (
            mean_75_above_mean_150
            &
            mean_75_above_mean_300
            &
            mean_150_above_mean_300
        )

        medium_uptrend = (
            mean_75_above_mean_150
            &
            mean_75_above_mean_300
            &
            ~mean_150_above_mean_300
        )

        weak_uptrend = (
            mean_75_above_mean_150
            &
            ~mean_75_above_mean_300
            &
            ~mean_150_above_mean_300
        )

        strong_downtrend = (
            ~mean_75_above_mean_150
            &
            ~mean_75_above_mean_300
            &
            ~mean_150_above_mean_300
        )

        medium_downtrend = (
            ~mean_75_above_mean_150
            &
            ~mean_75_above_mean_300
            &
            mean_150_above_mean_300
        )

        weak_downtrend = (
            ~mean_75_above_mean_150
            &
            mean_75_above_mean_300
            &
            mean_150_above_mean_300
        )

        sell_ema_1a1 = (
            (dataframe['close'] > dataframe['ema_offset_sell'])
            # &
            # mean_above_median_300
            # &
            # mean_above_median_150
        )
        conditions.append(sell_ema_1a1)
        dataframe.loc[sell_ema_1a1, 'exit_tag'] += 'EMA_up '

        # sell_ema_1a2 = (
        #     (dataframe['close'] > dataframe['ema_offset_sell'])
        #     &
        #     mean_above_median_300
        #     &
        #     ~mean_above_median_150
        # )
        # conditions.append(sell_ema_1a2)
        # dataframe.loc[sell_ema_1a2, 'exit_tag'] += 'EMA_up_a2 '

        # sell_ema_1b1 = (
        #     (dataframe['close'] > dataframe['ema_offset_sell'])
        #     &
        #     ~mean_above_median_300
        #     &
        #     mean_above_median_150
        # )
        # conditions.append(sell_ema_1b1)
        # dataframe.loc[sell_ema_1b1, 'exit_tag'] += 'EMA_up_b1 '

        # sell_ema_1b2 = (
        #     (dataframe['close'] > dataframe['ema_offset_sell'])
        #     &
        #     ~mean_above_median_300
        #     &
        #     ~mean_above_median_150
        # )
        # conditions.append(sell_ema_1b2)
        # dataframe.loc[sell_ema_1b2, 'exit_tag'] += 'EMA_up_b2 '

        sell_ema_2a1 = (
            (dataframe['close'] < dataframe['ema_offset_sell2'])
            &
            mean_above_median_300
            &
            mean_above_median_150
        )
        conditions.append(sell_ema_2a1)
        dataframe.loc[sell_ema_2a1, 'exit_tag'] += 'EMA_down_a1 '

        sell_ema_2a2 = (
            (dataframe['close'] < dataframe['ema_offset_sell2a'])
            &
            mean_above_median_300
            &
            ~mean_above_median_150
        )
        conditions.append(sell_ema_2a2)
        dataframe.loc[sell_ema_2a2, 'exit_tag'] += 'EMA_down_a2 '

        sell_ema_2b = (
            (dataframe['close'] < dataframe['ema_offset_sell2'])
            &
            ~mean_above_median_300
        )
        conditions.append(sell_ema_2b)
        dataframe.loc[sell_ema_2b, 'exit_tag'] += 'EMA_down_b '

        sell_ema_2bb = (
            (dataframe['close'] < dataframe['ema_offset_sell2b'])
            &
            strong_uptrend
        )
        conditions.append(sell_ema_2bb)
        dataframe.loc[sell_ema_2bb, 'exit_tag'] += 'EMA_down_b2 '

        sell_ema_3a1 = (
            ((dataframe['close'] < dataframe['ema_offset_sell3a']).rolling(2).min() > 0)
            &
            mean_above_median_300
            &
            mean_above_median_150
        )
        conditions.append(sell_ema_3a1)
        dataframe.loc[sell_ema_3a1, 'exit_tag'] += 'EMA_down_2a1 '

        sell_ema_3a2 = (
            ((dataframe['close'] < dataframe['ema_offset_sell3']).rolling(2).min() > 0)
            &
            mean_above_median_300
            &
            ~mean_above_median_150
        )
        conditions.append(sell_ema_3a2)
        dataframe.loc[sell_ema_3a2, 'exit_tag'] += 'EMA_down_2a2 '

        sell_ema_3b = (
            ((dataframe['close'] < dataframe['ema_offset_sell3']).rolling(2).min() > 0)
            &
            ~mean_above_median_300
        )
        conditions.append(sell_ema_3b)
        dataframe.loc[sell_ema_3b, 'exit_tag'] += 'EMA_down_2b '

        sell_ema_4 = (
            (dataframe['close'] > dataframe['ema_offset_sell4']).rolling(2).min() > 0
        )
        conditions.append(sell_ema_4)
        dataframe.loc[sell_ema_4, 'exit_tag'] += 'EMA_up_2 '

        sell_long_green3a = (
            (dataframe['pct_change'].rolling(3).sum() > (0.01 * self.sell_long_green3.value))
            &
            ~strong_uptrend
            &
            ~strong_downtrend
            &
            ~medium_downtrend
        )
        conditions.append(sell_long_green3a)
        dataframe.loc[sell_long_green3a, 'exit_tag'] += 'green_3 '

        sell_long_green3c = (
            (dataframe['pct_change'].rolling(3).sum() > (0.01 * self.sell_long_green3b.value))
            &
            strong_downtrend
        )
        conditions.append(sell_long_green3c)
        dataframe.loc[sell_long_green3c, 'exit_tag'] += 'green_3_strong_down '

        # dataframe.loc[strong_uptrend, 'exit_tag'] += 'strong_up '
        # dataframe.loc[medium_uptrend, 'exit_tag'] += 'medium_up '
        # dataframe.loc[weak_uptrend, 'exit_tag'] += 'weak_up '
        # dataframe.loc[strong_downtrend, 'exit_tag'] += 'strong_down '
        # dataframe.loc[medium_downtrend, 'exit_tag'] += 'medium_down '
        # dataframe.loc[weak_downtrend, 'exit_tag'] += 'weak_down '

        add_check = (
            (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions) & add_check,
                'exit_long'
            ] = 1

        return dataframe
