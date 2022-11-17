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

class Cenderawasih_3_kucoin (IStrategy):

    def version(self) -> str:
        return "cend_3_kucoin"

    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 100.0
    }

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy_vwma": 31,
        "low_offset_vwma": 0.989,
        "buy_rsi_vwma": 52,

        "base_nb_candles_buy_ema": 19,
        "low_offset_ema": 0.912,

        "buy_ema_length_15m": 30,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell_ema": 61,
        "high_offset_ema": 0.942,

        "base_nb_candles_sell_ema2": 5,
        "high_offset_ema2": 0.908,

        "base_nb_candles_sell_ema3": 7,
        "high_offset_ema3": 0.947,

        "base_nb_candles_sell_ema4": 14,
        "high_offset_ema4": 1.088,
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
    
    optimize_buy_ema = False
    base_nb_candles_buy_ema = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_ema)
    low_offset_ema = DecimalParameter(0.9, 0.99, default=0.9, space='buy', optimize=optimize_buy_ema)

    optimize_buy_vwma = False
    base_nb_candles_buy_vwma = IntParameter(5, 100, default=6, space='buy', optimize=optimize_buy_vwma)
    low_offset_vwma = DecimalParameter(0.9, 0.99, default=0.9, space='buy', optimize=optimize_buy_vwma)

    optimize_buy_ema_length_15m = False
    buy_ema_length_15m = CategoricalParameter([5, 10, 15, 20, 25, 30, 35, 40], default=10, optimize=optimize_buy_ema_length_15m)
    
    buy_rsi_vwma = IntParameter(10, 70, default=50, optimize=False)

    optimize_sell_ema = False
    base_nb_candles_sell_ema = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema)
    high_offset_ema = DecimalParameter(0.9, 1.1, default=0.95, space='sell', optimize=optimize_sell_ema)

    optimize_sell_ema2 = False
    base_nb_candles_sell_ema2 = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema2)
    high_offset_ema2 = DecimalParameter(0.9, 1.1, default=0.95, space='sell', optimize=optimize_sell_ema2)

    optimize_sell_ema3 = False
    base_nb_candles_sell_ema3 = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema3)
    high_offset_ema3 = DecimalParameter(0.9, 1.1, default=0.95, space='sell', optimize=optimize_sell_ema3)

    optimize_sell_ema4 = False
    base_nb_candles_sell_ema4 = IntParameter(5, 100, default=6, space='sell', optimize=optimize_sell_ema4)
    high_offset_ema4 = DecimalParameter(0.9, 1.1, default=0.95, space='sell', optimize=optimize_sell_ema4)

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
        elif (current_profit > 0.015):
            sl_new = 0.005

        return sl_new

    age_filter = 30

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['age_filter_ok'] = (dataframe['volume'].rolling(window=self.age_filter, min_periods=self.age_filter).min() > 0)

        if not self.config['runmode'].value in ('dry_run', 'live'):
            drop_columns = ['open', 'high', 'low', 'close', 'volume']
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    @informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        if (self.config['runmode'].value in ('hyperopt')) and self.optimize_buy_ema_length_15m:
            for val in self.buy_ema_length_15m.range:
                dataframe[f'ema_{val}'] = ta.EMA(dataframe, timeperiod=int(val))
        else:
            dataframe[f'ema_{self.buy_ema_length_15m.value}'] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_length_15m.value))
            
        if not self.config['runmode'].value in ('dry_run', 'live'):
            drop_columns = ['open', 'high', 'low', 'close', 'volume']
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

        if not self.optimize_buy_ema:
            dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value)) *self.low_offset_ema.value

        if not self.optimize_buy_vwma:
            dataframe['vwma_offset_buy'] = pta.vwma(dataframe["close"], dataframe["volume"], int(self.base_nb_candles_buy_vwma.value)) *self.low_offset_vwma.value

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

        if self.optimize_buy_ema:
            dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value)) *self.low_offset_ema.value

        if self.optimize_buy_vwma:
            dataframe['vwma_offset_buy'] = pta.vwma(dataframe["close"], dataframe["volume"], int(self.base_nb_candles_buy_vwma.value)) *self.low_offset_vwma.value

        dataframe.loc[:, 'enter_tag'] = ''
        dataframe.loc[:, 'enter_long'] = 0

        add_check = (
            dataframe['live_data_ok']
            &
            dataframe['age_filter_ok_1d']
        )

        buy_offset_vwma = (
            ((dataframe['close'] < dataframe['vwma_offset_buy']))
            &
            (dataframe['close'] < dataframe['ema_offset_buy'])
            &
            (dataframe['rsi'] < self.buy_rsi_vwma.value)
            &
            (dataframe['close'] < dataframe[f'ema_{self.buy_ema_length_15m.value}_15m'])
        )
        dataframe.loc[buy_offset_vwma, 'enter_tag'] += 'vwma '
        conditions.append(buy_offset_vwma)

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
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema.value)) *self.high_offset_ema.value

        if self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema2.value)) *self.high_offset_ema2.value

        if self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema3.value)) *self.high_offset_ema3.value

        if self.optimize_sell_ema4:
            dataframe['ema_offset_sell4'] = ta.EMA(dataframe, int(self.base_nb_candles_sell_ema4.value)) *self.high_offset_ema4.value

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

        sell_cond_1 = (
            (dataframe['close'] > dataframe['ema_offset_sell4']).rolling(2).min() > 0
        )
        conditions.append(sell_cond_1)
        dataframe.loc[sell_cond_1, 'exit_tag'] += 'EMA_4 '

        add_check = (
            (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions) & add_check,
                'exit_long'
            ] = 1

        return dataframe
    