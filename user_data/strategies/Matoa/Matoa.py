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
import math
import talib.abstract as ta
import logging
from logging import FATAL

logger = logging.getLogger(__name__)

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

class Matoa (IStrategy):

    def version(self) -> str:
        return "Matoa-v1"

    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy_hma": 62,
        "low_offset_hma": 0.88,

        "base_nb_candles_buy_hma2": 118,
        "low_offset_hma2": 0.86,

        "base_nb_candles_buy_hma3": 46,
        "low_offset_hma3": 0.883,

        "buy_length_volatility": 95,
        "buy_max_volatility": 1.19,

        "buy_length_volatility2": 110,
        "buy_max_volatility2": 1.55,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell_ema": 49,
        "high_offset_ema": 1.07,

        "base_nb_candles_sell_ema2": 36,
        "high_offset_ema2": 0.95,

        "base_nb_candles_sell_ema3": 104,
        "high_offset_ema3": 0.94,

        "base_nb_candles_sell_ema4": 5,
        "high_offset_ema4": 1.01,
    }

    # Protection hyperspace params:
    protection_params = {
        "cooldown_lookback": 1,  # value loaded from strategy
    }


    cooldown_lookback = IntParameter(1, 48, default=2, space="protection", optimize=False)

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        return prot
    
    dummy = IntParameter(20, 70, default=61, space='buy', optimize=True)

    optimize_buy_hma = False
    base_nb_candles_buy_hma = IntParameter(5, 150, default=6, space='buy', optimize=optimize_buy_hma)
    low_offset_hma = DecimalParameter(0.7, 0.99, default=0.98, decimals=2, space='buy', optimize=optimize_buy_hma)

    optimize_buy_hma2 = False
    base_nb_candles_buy_hma2 = IntParameter(5, 150, default=6, space='buy', optimize=optimize_buy_hma2)
    low_offset_hma2 = DecimalParameter(0.7, 0.99, default=0.95, decimals=2, space='buy', optimize=optimize_buy_hma2)

    optimize_buy_hma3 = False
    base_nb_candles_buy_hma3 = IntParameter(5, 150, default=6, space='buy', optimize=optimize_buy_hma3)
    low_offset_hma3 = DecimalParameter(0.7, 0.99, default=0.95, space='buy', optimize=optimize_buy_hma3)

    optimize_buy_volatility = False
    buy_length_volatility = IntParameter(10, 200, default=72, space='buy', optimize=optimize_buy_volatility)
    buy_min_volatility = DecimalParameter(0, 0.5, default=0, decimals = 2, space='buy', optimize=False)
    buy_max_volatility = DecimalParameter(0.5, 2, default=1, decimals = 2, space='buy', optimize=optimize_buy_volatility)

    optimize_buy_volatility2 = False
    buy_length_volatility2 = IntParameter(10, 200, default=72, space='buy', optimize=optimize_buy_volatility2)
    buy_max_volatility2 = DecimalParameter(0.5, 2, default=1, decimals = 2, space='buy', optimize=optimize_buy_volatility2)

    optimize_sell_ema = False
    base_nb_candles_sell_ema = IntParameter(5, 150, default=6, space='sell', optimize=optimize_sell_ema)
    high_offset_ema = DecimalParameter(1, 1.2, default=1.02, decimals=2, space='sell', optimize=optimize_sell_ema)

    optimize_sell_ema2 = False
    base_nb_candles_sell_ema2 = IntParameter(5, 150, default=6, space='sell', optimize=optimize_sell_ema2)
    high_offset_ema2 = DecimalParameter(0.7, 0.99, default=0.98, decimals=2, space='sell', optimize=optimize_sell_ema2)

    optimize_sell_ema3 = False
    base_nb_candles_sell_ema3 = IntParameter(5, 150, default=6, space='sell', optimize=optimize_sell_ema3)
    high_offset_ema3 = DecimalParameter(0.7, 0.99, default=0.95, decimals=2, space='sell', optimize=optimize_sell_ema3)

    optimize_sell_ema4 = False
    base_nb_candles_sell_ema4 = IntParameter(5, 150, default=6, space='sell', optimize=optimize_sell_ema4)
    high_offset_ema4 = DecimalParameter(1, 1.2, default=1, decimals=2, space='sell', optimize=optimize_sell_ema4)

    # Stoploss:
    stoploss = -0.08

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.08
    trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 300

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

        if not self.optimize_buy_hma2:
            dataframe['hma_offset_buy2'] = tv_hma(dataframe, int(self.base_nb_candles_buy_hma2.value)) *self.low_offset_hma2.value

        if not self.optimize_buy_hma3:
            dataframe['hma_offset_buy3'] = tv_hma(dataframe, int(self.base_nb_candles_buy_hma3.value)) *self.low_offset_hma3.value

        if not self.optimize_buy_volatility:
            df_std = dataframe['close'].rolling(int(self.buy_length_volatility.value)).std()
            dataframe["volatility"] = (df_std > self.buy_min_volatility.value) & (df_std < self.buy_max_volatility.value)

        if not self.optimize_buy_volatility2:
            df_std = dataframe['close'].rolling(int(self.buy_length_volatility2.value)).std()
            dataframe["volatility2"] = (df_std > self.buy_min_volatility.value) & (df_std < self.buy_max_volatility2.value)

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

        if self.optimize_buy_hma2:
            dataframe['hma_offset_buy2'] = tv_hma(dataframe, int(self.base_nb_candles_buy_hma2.value)) *self.low_offset_hma2.value

        if self.optimize_buy_hma3:
            dataframe['hma_offset_buy3'] = tv_hma(dataframe, int(self.base_nb_candles_buy_hma3.value)) *self.low_offset_hma3.value

        if self.optimize_buy_volatility:
            df_std = dataframe['close'].rolling(int(self.buy_length_volatility.value)).std()
            dataframe["volatility"] = (df_std > self.buy_min_volatility.value) & (df_std < self.buy_max_volatility.value)

        if self.optimize_buy_volatility2:
            df_std = dataframe['close'].rolling(int(self.buy_length_volatility2.value)).std()
            dataframe["volatility2"] = (df_std > self.buy_min_volatility.value) & (df_std < self.buy_max_volatility2.value)
        
        dataframe['enter_tag'] = ''

        add_check = (
            dataframe['live_data_ok']
            &
            dataframe['age_filter_ok_1d']
            &
            (dataframe['close'] < dataframe['open'])
        )

        buy_offset_hma = (
            (dataframe['close'] < dataframe['hma_offset_buy'])
            &
            (dataframe['volatility'].shift() == True)
        )
        dataframe.loc[buy_offset_hma, 'enter_tag'] += 'hma '
        conditions.append(buy_offset_hma)

        buy_offset_hma2 = (
            ((dataframe['close'] < dataframe['hma_offset_buy2']).rolling(2).min() > 0)
            &
            (dataframe['volatility2'].shift() == True)
        )
        dataframe.loc[buy_offset_hma2, 'enter_tag'] += 'hma_2 '
        conditions.append(buy_offset_hma2)

        buy_offset_hma3 = (
            (dataframe['close'] < dataframe['hma_offset_buy3']).rolling(3).min() > 0
        )
        dataframe.loc[buy_offset_hma3, 'enter_tag'] += 'hma_3 '
        conditions.append(buy_offset_hma3)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions)
                &
                add_check,
                'enter_long',
            ]= 1

            dataframe.loc[
                buy_offset_hma 
                &
                buy_offset_hma2
                &
                np.invert(buy_offset_hma3),
                'enter_long'
            ]= 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

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
            (dataframe['close'] > dataframe['ema_offset_sell4']).rolling(3).min() > 0
        )
        conditions.append(sell_ema_4)
        dataframe.loc[sell_ema_4, 'exit_tag'] += 'EMA_up_3 '

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
