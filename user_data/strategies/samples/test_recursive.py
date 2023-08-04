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
import logging
from logging import FATAL
import pandas as pd

def smi_momentum(dataframe: DataFrame, k_length=9, d_length=3):
    """     
    The Stochastic Momentum Index (SMI) Indicator was developed by 
    William Blau in 1993 and is considered to be a momentum indicator 
    that can help identify trend reversal points
        
    :return: DataFrame with smi column populated
    """
    df = dataframe.copy()
    ll = df['low'].rolling(window=k_length).min()
    hh = df['high'].rolling(window=k_length).max()

    diff = hh - ll
    rdiff = df['close'] - (hh + ll) / 2

    avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
    avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()

    df['smi'] = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)
    
    return df['smi']

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

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

# Range midpoint acts as Support
def is_support(row_data) -> bool:
    conditions = []
    for row in range(len(row_data)-1):
        if row < len(row_data)//2:
            conditions.append(row_data[row] > row_data[row+1])
        else:
            conditions.append(row_data[row] < row_data[row+1])
    result = reduce(lambda x, y: x & y, conditions)
    return result

# Range midpoint acts as Resistance
def is_resistance(row_data) -> bool:
    conditions = []
    for row in range(len(row_data)-1):
        if row < len(row_data)//2:
            conditions.append(row_data[row] < row_data[row+1])
        else:
            conditions.append(row_data[row] > row_data[row+1])
    result = reduce(lambda x, y: x & y, conditions)
    return result

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

def williams_fractals(dataframe: pd.DataFrame, period: int = 2) -> tuple:
    """Williams Fractals implementation

    :param dataframe: OHLC data
    :param period: number of lower (or higher) points on each side of a high (or low)
    :return: tuple of boolean Series (bearish, bullish) where True marks a fractal pattern
    """

    window = 2 * period + 1

    bears = dataframe['high'].rolling(window, center=True).apply(lambda x: x[period] == max(x), raw=True)
    bulls = dataframe['low'].rolling(window, center=True).apply(lambda x: x[period] == min(x), raw=True)

    return bears, bulls

# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    vwma = vwma.fillna(0, inplace=True)
    return vwma

# Exponential moving average of a volume weighted simple moving average
def ema_vwma_osc(dataframe, len_slow_ma):
    slow_ema = Series(ta.EMA(vwma(dataframe, len_slow_ma), len_slow_ma))
    return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

def t3_average(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe1'].fillna(0, inplace=True)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe2'].fillna(0, inplace=True)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe3'].fillna(0, inplace=True)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe4'].fillna(0, inplace=True)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe5'].fillna(0, inplace=True)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    df['xe6'].fillna(0, inplace=True)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']

# Pivot Points - 3 variants - daily recommended
def pivot_points(dataframe: DataFrame, mode = 'fibonacci') -> Series:
    if mode == 'simple':
        hlc3_pivot = (dataframe['high'] + dataframe['low'] + dataframe['close']).shift(1) / 3
        res1 = hlc3_pivot * 2 - dataframe['low'].shift(1)
        sup1 = hlc3_pivot * 2 - dataframe['high'].shift(1)
        res2 = hlc3_pivot + (dataframe['high'] - dataframe['low']).shift()
        sup2 = hlc3_pivot - (dataframe['high'] - dataframe['low']).shift()
        res3 = hlc3_pivot * 2 + (dataframe['high'] - 2 * dataframe['low']).shift()
        sup3 = hlc3_pivot * 2 - (2 * dataframe['high'] - dataframe['low']).shift()
        return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
    elif mode == 'fibonacci':
        hlc3_pivot = (dataframe['high'] + dataframe['low'] + dataframe['close']).shift(1) / 3
        hl_range = (dataframe['high'] - dataframe['low']).shift(1)
        res1 = hlc3_pivot + 0.382 * hl_range
        sup1 = hlc3_pivot - 0.382 * hl_range
        res2 = hlc3_pivot + 0.618 * hl_range
        sup2 = hlc3_pivot - 0.618 * hl_range
        res3 = hlc3_pivot + 1 * hl_range
        sup3 = hlc3_pivot - 1 * hl_range
        return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
    elif mode == 'DeMark':
        demark_pivot_lt = (dataframe['low'] * 2 + dataframe['high'] + dataframe['close'])
        demark_pivot_eq = (dataframe['close'] * 2 + dataframe['low'] + dataframe['high'])
        demark_pivot_gt = (dataframe['high'] * 2 + dataframe['low'] + dataframe['close'])
        demark_pivot = np.where((dataframe['close'] < dataframe['open']), demark_pivot_lt, np.where((dataframe['close'] > dataframe['open']), demark_pivot_gt, demark_pivot_eq))
        dm_pivot = demark_pivot / 4
        dm_res = demark_pivot / 2 - dataframe['low']
        dm_sup = demark_pivot / 2 - dataframe['high']
        return dm_pivot, dm_res, dm_sup

# Heikin Ashi candles
def heikin_ashi(dataframe, smooth_inputs = False, smooth_outputs = False, length = 10):
    df = dataframe[['open','close','high','low']].copy().fillna(0)
    if smooth_inputs:
        df['open_s']  = ta.EMA(df['open'], timeframe = length)
        df['high_s']  = ta.EMA(df['high'], timeframe = length)
        df['low_s']   = ta.EMA(df['low'],  timeframe = length)
        df['close_s'] = ta.EMA(df['close'],timeframe = length)

        open_ha  = (df['open_s'].shift(1) + df['close_s'].shift(1)) / 2
        high_ha  = df.loc[:, ['high_s', 'open_s', 'close_s']].max(axis=1)
        low_ha   = df.loc[:, ['low_s', 'open_s', 'close_s']].min(axis=1)
        close_ha = (df['open_s'] + df['high_s'] + df['low_s'] + df['close_s'])/4
    else:
        open_ha  = (df['open'].shift(1) + df['close'].shift(1)) / 2
        high_ha  = df.loc[:, ['high', 'open', 'close']].max(axis=1)
        low_ha   = df.loc[:, ['low', 'open', 'close']].min(axis=1)
        close_ha = (df['open'] + df['high'] + df['low'] + df['close'])/4

    open_ha = open_ha.fillna(0)
    high_ha = high_ha.fillna(0)
    low_ha  = low_ha.fillna(0)
    close_ha = close_ha.fillna(0)

    if smooth_outputs:
        open_sha  = ta.EMA(open_ha, timeframe = length)
        high_sha  = ta.EMA(high_ha, timeframe = length)
        low_sha   = ta.EMA(low_ha, timeframe = length)
        close_sha = ta.EMA(close_ha, timeframe = length)

        return open_sha, close_sha, low_sha
    else:
        return open_ha, close_ha, low_ha

# Peak Percentage Change
def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
    """
    Rolling Percentage Change Maximum across interval.

    :param dataframe: DataFrame The original OHLC dataframe
    :param method: High to Low / Open to Close
    :param length: int The length to look back
    """
    if method == 'HL':
        return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
    elif method == 'OC':
        return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
    else:
        raise ValueError(f"Method {method} not defined!")

# Percentage distance to top peak
def top_percent_change(self, dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price

    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The length to look back
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def smma(s: Series, length):
    smma = s.copy()
    smma[:length - 1] = np.nan
    smma.iloc[length - 1] = ta.SMA(s, length)[length]
    for i in range(length, len(s)):
        smma.iloc[i] = ((length - 1) * smma.iloc[i - 1] + smma.iloc[i]) / length
    return smma

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

# exclusive vwap indicator for zond by @rk
def vwap_fast(dataframe: DataFrame):
    split_indices = list(dataframe.loc[
        (
            (dataframe['date'].dt.second == 0)
            &
            (dataframe['date'].dt.minute == 0)
            &
            (dataframe['date'].dt.hour == 0)
        )].index)
    split_indices.insert(0, 0)
    split_indices.append(len(dataframe))
    vwap_slices = []
    for i in range(1, len(split_indices)):
        start_idx = split_indices[i - 1]
        end_idx = split_indices[i]
        slice = dataframe[start_idx:end_idx]
        hlc3 = (slice['high'] + slice['low'] + slice['close']) / 3
        wp = hlc3 * slice['volume']
        vwap = wp.cumsum() / slice['volume'].cumsum()
        vwap_slices.append(vwap)
    vwap = pd.concat(vwap_slices)
    return vwap

def chaikin_mf(dataframe, periods=20):
    close = dataframe['close']
    low = dataframe['low']
    high = dataframe['high']
    volume = dataframe['volume']

    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)# float division by zero
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()

    return Series(cmf, name='cmf')

logger = logging.getLogger(__name__)

class test_recursive (IStrategy):

    def version(self) -> str:
        return "test_recursive"

    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 0.01
    }

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    timeframe = '30m'

    process_only_new_candles = True
    startup_candle_count = 500

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema_25'] = ta.EMA(dataframe, 25)
        dataframe['ema_50'] = ta.EMA(dataframe, 50)
        dataframe['ema_100'] = ta.EMA(dataframe, 100)
        dataframe['ema_200'] = ta.EMA(dataframe, 200)

        dataframe['tema_25'] = ta.TEMA(dataframe, 25)
        dataframe['dema_25'] = ta.DEMA(dataframe, 25)
        
        dataframe['tema_50'] = ta.TEMA(dataframe, 50)
        dataframe['dema_50'] = ta.DEMA(dataframe, 50)
        
        dataframe['tema_100'] = ta.TEMA(dataframe, 100)
        dataframe['dema_100'] = ta.DEMA(dataframe, 100)
        
        dataframe['tema_200'] = ta.TEMA(dataframe, 200)
        dataframe['dema_200'] = ta.DEMA(dataframe, 200)
        dataframe['hma_100'] = tv_hma(dataframe, 100)

        dataframe['ewo_50_200'] = EWO(dataframe, 50, 200)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma_100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['close_9_mean'] = dataframe['close'].rolling(9).mean()

        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_20'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_45'] = ta.RSI(dataframe, timeperiod=45)

        dataframe['mfi_14'] = ta.MFI(dataframe, 14)
        dataframe['mfi_45'] = ta.MFI(dataframe, 45)

        dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

        # BB 20 - STD2
        bb_20_std2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb20_2_low'] = bb_20_std2['lower']
        dataframe['bb20_2_mid'] = bb_20_std2['mid']
        dataframe['bb20_2_upp'] = bb_20_std2['upper']

        # BB 40 - STD2
        bb_40_std2 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['bb40_2_low'] = bb_40_std2['lower']
        dataframe['bb40_2_mid'] = bb_40_std2['mid']
        dataframe['bb40_2_delta'] = (bb_40_std2['mid'] - dataframe['bb40_2_low']).abs()
        dataframe['bb40_2_tail'] = (dataframe['close'] - dataframe['bb40_2_low']).abs()

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_480'] = williams_r(dataframe, period=480)

        # CTI
        dataframe['cti_20'] = pta.cti(dataframe["close"], length=20)

        # SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # CCI
        dataframe['cci_20'] = ta.CCI(dataframe, source='hlc3', timeperiod=20)

        # TSI
        tsi = pta.tsi(dataframe["close"])
        dataframe['tsi'] = tsi.iloc[:, 0]
        dataframe['tsi_signal'] = tsi.iloc[:, 1]

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Dip protection
        dataframe['tpct_change_0'] = top_percent_change(self, dataframe, 0)
        dataframe['tpct_change_2'] = top_percent_change(self, dataframe, 2)

        # Close max
        dataframe['close_max_12'] = dataframe['close'].rolling(12).max()
        dataframe['close_max_24'] = dataframe['close'].rolling(24).max()
        dataframe['close_max_48'] = dataframe['close'].rolling(48).max()

        # Close min
        dataframe['close_min_12'] = dataframe['close'].rolling(12).min()


        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe
