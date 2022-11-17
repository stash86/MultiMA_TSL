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
from technical.indicators import zema
import talib.abstract as ta
import math
import pandas_ta as pta
import logging
from logging import FATAL
import time

logger = logging.getLogger(__name__)

###########################################################################################################
##                                                                                                       ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##
##                                                                                                       ##
##                                                                                                       ##
###########################################################################################################
##               DONATIONS for stash86                                                                   ##
##                                                                                                       ##
##   Real-life money : https://patreon.com/stash86                                                       ##
##   BTC: 1FghqtgGLpD9F21BNDMje4iyj4cSzVPZPb                                                             ##
##   ETH (ERC20): 0x689c16451889824d3d3a79ad6fc867909dc8874d                                             ##
##   BEP20/BSC (USDT): 0x689c16451889824d3d3a79ad6fc867909dc8874d                                        ##
##   TRC20/TRON (USDT): TKMuRHJppPok3ik2siZp2SYRdBdfdSWxrt                                               ##
##                                                                                                       ##
##               REFERRAL LINKS                                                                          ##
##                                                                                                       ##
##  Binance: https://accounts.binance.com/en/register?ref=143744527                                      ##
##  Kucoin: https://www.kucoin.com/ucenter/signup?rcode=r3BWY2T                                          ##
##  Vultr (you get $100 credit that expires in 14 days) : https://www.vultr.com/?ref=8944192-8H          ##
###########################################################################################################

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

def tv_hma(dataframe, length = 9) -> DataFrame:
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

    h = 2 * tv_wma(dataframe['close'], math.floor(length / 2)) - tv_wma(dataframe['close'], length)

    tv_hma = tv_wma(h, math.floor(math.sqrt(length)))
    # dataframe.drop("h", inplace=True, axis=1)

    return tv_hma

def rvol(dataframe, window=24):
    av = ta.SMA(dataframe['volume'], timeperiod=int(window))
    rvol = dataframe['volume'] / av
    return rvol

# patreon
class Cenderawasih_2_kucoin (IStrategy):

    def version(self) -> str:
        return "v2_kucoin"

    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 100.0
    }

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy_vwma": 44,
        "low_offset_vwma": 0.931,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell_ema": 58,
        "high_offset_ema": 0.951,

        "base_nb_candles_sell_ema2": 5,
        "high_offset_ema2": 0.908,

        "base_nb_candles_sell_ema3": 49,
        "high_offset_ema3": 0.914,
    }
   
    # Protection hyperspace params:
    protection_params = {
        "cooldown_lookback": 2,  # value loaded from strategy
    }


    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=False)

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        return prot
    
    dummy = IntParameter(20, 70, default=61, space='buy', optimize=False)

    # rsi_buy_ema = IntParameter(20, 70, default=61, space='buy', optimize=False)
    # buy_rsi_1 = IntParameter(0, 70, default=50, optimize=False)
    # buy_rsi_fast_1 = IntParameter(0, 70, default=50, optimize=False)

    # optimize_buy_hma = False
    # base_nb_candles_buy_hma = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_hma)
    # low_offset_hma = DecimalParameter(0.9, 0.99, default=0.95, space='buy', optimize=optimize_buy_hma)

    # optimize_buy_hma2 = False
    # base_nb_candles_buy_hma2 = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_hma2)
    # low_offset_hma2 = DecimalParameter(0.9, 0.99, default=0.95, space='buy', optimize=optimize_buy_hma2)

    # optimize_buy_ema = False
    # base_nb_candles_buy_ema = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_ema)
    # low_offset_ema = DecimalParameter(0.9, 1.1, default=1, space='buy', optimize=optimize_buy_ema)

    # optimize_buy_ema2 = False
    # base_nb_candles_buy_ema2 = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_ema2)
    # low_offset_ema2 = DecimalParameter(0.9, 1.1, default=1, space='buy', optimize=optimize_buy_ema2)

    optimize_buy_vwma = False
    base_nb_candles_buy_vwma = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_vwma)
    low_offset_vwma = DecimalParameter(0.9, 0.99, default=0.9, space='buy', optimize=optimize_buy_vwma)

    # optimize_buy_vwma_2 = False
    # base_nb_candles_buy_vwma_2 = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_vwma_2)
    # low_offset_vwma_2 = DecimalParameter(0.9, 0.99, default=0.9, space='buy', optimize=optimize_buy_vwma_2)

    # optimize_buy_vwma_3 = False
    # base_nb_candles_buy_vwma_3 = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_vwma_3)
    # low_offset_vwma_3 = DecimalParameter(0.9, 0.99, default=0.9, space='buy', optimize=optimize_buy_vwma_3)

    # optimize_buy_vwma_4 = False
    # base_nb_candles_buy_vwma_4 = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_vwma_4)
    # low_offset_vwma_4 = DecimalParameter(0.9, 0.99, default=0.9, space='buy', optimize=optimize_buy_vwma_4)

    # optimize_buy_vwma2 = False
    # base_nb_candles_buy_vwma2 = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_vwma2)
    # low_offset_vwma2 = DecimalParameter(0.9, 0.99, default=0.9, space='buy', optimize=optimize_buy_vwma2)

    # optimize_buy_volatility = False
    # buy_length_volatility = IntParameter(10, 200, default=72, space='buy', optimize=optimize_buy_volatility)
    # buy_min_volatility = DecimalParameter(0, 0.5, default=0, decimals = 2, space='buy', optimize=optimize_buy_volatility)
    # buy_max_volatility = DecimalParameter(0.5, 2, default=1, decimals = 2, space='buy', optimize=optimize_buy_volatility)

    # optimize_buy_volatility_2 = True
    # buy_length_volatility_2 = IntParameter(10, 200, default=72, space='buy', optimize=optimize_buy_volatility_2)
    # buy_min_volatility_2 = DecimalParameter(0, 0.2, default=0, decimals = 2, space='buy', optimize=optimize_buy_volatility_2)
    # buy_max_volatility_2 = DecimalParameter(0.5, 2, default=1, decimals = 1, space='buy', optimize=optimize_buy_volatility_2)

    # optimize_buy_volatility_hma = False
    # buy_length_volatility_hma = IntParameter(10, 200, default=72, space='buy', optimize=optimize_buy_volatility_hma)
    # buy_min_volatility_hma = DecimalParameter(0, 0.5, default=0, decimals = 2, space='buy', optimize=optimize_buy_volatility_hma)
    # buy_max_volatility_hma = DecimalParameter(0.5, 2, default=1, decimals = 2, space='buy', optimize=optimize_buy_volatility_hma)

    # optimize_buy_volatility2 = False
    # buy_length_volatility2 = IntParameter(10, 200, default=72, space='buy', optimize=optimize_buy_volatility2)
    # buy_min_volatility2 = DecimalParameter(0, 0.5, default=0, decimals = 2, space='buy', optimize=False)
    # buy_max_volatility2 = DecimalParameter(0.5, 2, default=1, decimals = 2, space='buy', optimize=optimize_buy_volatility2)

    # optimize_buy_volume = False
    # buy_length_volume = IntParameter(5, 100, default=6, optimize=optimize_buy_volume)
    # buy_volume_volatility = DecimalParameter(0.5, 3, default=1, decimals=2, optimize=optimize_buy_volume)

    # buy_rsi_vwma = IntParameter(10, 70, default=50, optimize=False)
    # buy_rsi4_vwma = IntParameter(10, 70, default=50, optimize=False)
    # buy_rsx_vwma = IntParameter(20, 70, default=61, optimize=False)
    # buy_rsx4_vwma = IntParameter(20, 70, default=61, optimize=False)

    # optimize_rsi_rsx_vwma_2 = False
    # buy_rsi_vwma_2 = IntParameter(10, 70, default=50, optimize=optimize_rsi_rsx_vwma_2)
    # buy_rsi4_vwma_2 = IntParameter(10, 70, default=50, optimize=optimize_rsi_rsx_vwma_2)
    # buy_rsx_vwma_2 = IntParameter(20, 70, default=61, optimize=optimize_rsi_rsx_vwma_2)
    # buy_rsx4_vwma_2 = IntParameter(20, 70, default=61, optimize=optimize_rsi_rsx_vwma_2)

    # optimize_rsi_rsx_hma = False
    # buy_rsi_hma = IntParameter(10, 70, default=50, optimize=optimize_rsi_rsx_hma)
    # buy_rsi4_hma = IntParameter(10, 70, default=50, optimize=optimize_rsi_rsx_hma)
    # buy_rsx_hma = IntParameter(20, 70, default=61, optimize=optimize_rsi_rsx_hma)
    # buy_rsx4_hma = IntParameter(20, 70, default=61, optimize=optimize_rsi_rsx_hma)

    # optimize_2_stars = False
    # buy_rsx_2_stars = IntParameter(10, 70, default=61, optimize=optimize_2_stars)

    # optimize_3_stars = False
    # buy_rsx_3_stars = IntParameter(10, 70, default=61, optimize=optimize_3_stars)

    # optimize_4_stars = False
    # buy_rsx_4_stars = IntParameter(10, 70, default=61, optimize=optimize_4_stars)

    # optimize_5_stars = False
    # buy_rsx_5_stars = IntParameter(10, 70, default=61, optimize=optimize_5_stars)

    # buy_rsx_hma = IntParameter(20, 70, default=61, optimize=False)
    # buy_rsx4_hma = IntParameter(20, 70, default=61, optimize=False)

    # Sell
    # optimize_sell_hma = False
    # base_nb_candles_sell_hma = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_hma)
    # high_offset_hma = DecimalParameter(0.9, 1.1, default=0.95, space='sell', optimize=optimize_sell_hma)

    optimize_sell_ema = False
    base_nb_candles_sell_ema = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema)
    high_offset_ema = DecimalParameter(0.9, 1.1, default=0.95, space='sell', optimize=optimize_sell_ema)

    optimize_sell_ema2 = False
    base_nb_candles_sell_ema2 = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema2)
    high_offset_ema2 = DecimalParameter(0.9, 1.1, default=0.95, space='sell', optimize=optimize_sell_ema2)

    optimize_sell_ema3 = False
    base_nb_candles_sell_ema3 = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema3)
    high_offset_ema3 = DecimalParameter(0.9, 1.1, default=0.95, space='sell', optimize=optimize_sell_ema3)

    # Stoploss:
    stoploss = -0.99

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

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if (current_profit > 0.2):
            sl_new = 0.05
        elif (current_profit > 0.1):
            sl_new = 0.03
        elif (current_profit > 0.06):
            sl_new = 0.02
        elif (current_profit > 0.03):
            sl_new = 0.01

        return sl_new

    age_filter = 30

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['age_filter_ok'] = (dataframe['volume'].rolling(window=self.age_filter, min_periods=self.age_filter).min() > 0)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

        if not self.optimize_buy_vwma:
            dataframe['vwma_offset_buy'] = pta.vwma(dataframe["close"], dataframe["volume"], int(self.base_nb_candles_buy_vwma.value)) *self.low_offset_vwma.value

        if not self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema.value)) *self.high_offset_ema.value

        if not self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema2.value)) *self.high_offset_ema2.value

        if not self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema3.value)) *self.high_offset_ema3.value

        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []

        if self.optimize_buy_vwma:
            dataframe['vwma_offset_buy'] = pta.vwma(dataframe["close"], dataframe["volume"], int(self.base_nb_candles_buy_vwma.value)) *self.low_offset_vwma.value

        dataframe.loc[:, 'enter_tag'] = ''
        dataframe.loc[:, 'buy'] = 0

        add_check = (
            dataframe['live_data_ok']
            &
            dataframe['age_filter_ok_1d']
        )

        buy_offset_vwma = (
            ((dataframe['close'] < dataframe['vwma_offset_buy']))
        )
        dataframe.loc[buy_offset_vwma, 'enter_tag'] += 'vwma '
        conditions.append(buy_offset_vwma)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions)
                &
                add_check,
                'buy',
            ]= 1


        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # if self.optimize_sell_hma:
        #     dataframe['hma_offset_sell'] = tv_hma(dataframe, int(self.base_nb_candles_sell_hma.value)) *self.high_offset_hma.value

        if self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema.value)) *self.high_offset_ema.value

        if self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema2.value)) *self.high_offset_ema2.value

        if self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema3.value)) *self.high_offset_ema3.value

        dataframe.loc[:, 'exit_tag'] = ''
        conditions = []

        sell_cond_2 = (                   
            (dataframe['close'] > dataframe['ema_offset_sell'])
        )
        conditions.append(sell_cond_2)
        dataframe.loc[sell_cond_2, 'exit_tag'] += 'EMA_1 '

        sell_cond_4 = (                   
            (dataframe['close'] < dataframe['ema_offset_sell2'])
        )
        conditions.append(sell_cond_4)
        dataframe.loc[sell_cond_4, 'exit_tag'] += 'EMA_2 '

        sell_cond_3 = (                   
            ((dataframe['close'] < dataframe['ema_offset_sell3']).rolling(2).min() > 0)
        )
        conditions.append(sell_cond_3)
        dataframe.loc[sell_cond_3, 'exit_tag'] += 'EMA_3 '

        add_check = (
            (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions) & add_check,
                'sell'
            ] = 1

        return dataframe
