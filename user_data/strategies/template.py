import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter, stoploss_from_open)
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
import time

logger = logging.getLogger(__name__)

class template (IStrategy):

    def version(self) -> str:
        return "template-v1"

    INTERFACE_VERSION = 3

    # ROI table:
    # ROI is used for backtest/hyperopt to not overestimating the effectiveness of trailing stoploss
    # Remember to change this to 100000 for dry/live and turn on trailing stoploss below
    minimal_roi = {
        "0": 0.03
    }

    optimize_buy_ema = False
    buy_length_ema = IntParameter(1, 15, default=6, optimize=optimize_buy_ema)

    optimize_buy_ema2 = False
    buy_length_ema2 = IntParameter(1, 15, default=6, optimize=optimize_buy_ema2)

    optimize_sell_ema = False
    sell_length_ema = IntParameter(1, 15, default=6, optimize=optimize_sell_ema)

    optimize_sell_ema2 = False
    sell_length_ema2 = IntParameter(1, 15, default=6, optimize=optimize_sell_ema2)

    optimize_sell_ema3 = False
    sell_length_ema3 = IntParameter(1, 15, default=6, optimize=optimize_sell_ema3)

    sell_min_profit = DecimalParameter(0, 0.03, default=0.01, decimals=2, optimize=False)

    sell_clear_old_trade = IntParameter(6, 15, default=10, optimize=False)
    sell_clear_old_trade_profit = IntParameter(-2, 2, default=1, optimize=False)

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    # Turned off for backtest/hyperopt to not gaming the backtest.
    # Turn this on for dry/live
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
    startup_candle_count = 150

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # In-strat age filter
        dataframe['age_filter_ok'] = (dataframe['volume'].rolling(window=30, min_periods=30).min() > 0)

        # Drop unused columns to save memory
        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    # Use BTC indicators as informative for other pairs
    @informative('30m', 'BTC/{stake}', '{base}_{column}_{timeframe}')
    def populate_indicators_btc_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # DOn't trade coins that have 0 volume candle on the past 72 candles
        dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Calculate EMA30 of RSI
        dataframe['ema_rsi_30'] = ta.EMA(dataframe['rsi'], 30)

        if not self.optimize_buy_ema:
            # Have the period of EMA on increment of 5 without having to use CategoricalParameter
            dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(5 * self.buy_length_ema.value)) * 0.9

        if not self.optimize_buy_ema2:
            dataframe['ema_offset_buy2'] = ta.EMA(dataframe, int(5 * self.buy_length_ema2.value)) * 0.9

        if not self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(5 * self.sell_length_ema.value))

        if not self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(5 * self.sell_length_ema2.value))

        if not self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(5 * self.sell_length_ema3.value))

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []

        if self.optimize_buy_ema:
            dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(5 * self.buy_length_ema.value)) * 0.9

        if self.optimize_buy_ema2:
            dataframe['ema_offset_buy2'] = ta.EMA(dataframe, int(5 * self.buy_length_ema2.value)) * 0.9

        if self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(5 * self.sell_length_ema.value))

        if self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(5 * self.sell_length_ema2.value))

        if self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(5 * self.sell_length_ema3.value))

        dataframe['enter_tag'] = ''

        add_check = (
            dataframe['live_data_ok']
            &
            dataframe['age_filter_ok_1d']
            &
            (dataframe['close'] < dataframe['open'])
        )

        # Imitate exit signal colliding, where entry shouldn't happen when the exit signal is triggered.
        # So this check make sure no exit logics are triggered
        ema_check = (
            (dataframe['close'] > dataframe['ema_offset_sell'])
            &
            ((dataframe['close'] < dataframe['ema_offset_sell2']).rolling(2).min() == 0)
        )

        buy_offset_ema = (
            (dataframe['close'] < dataframe['ema_offset_buy'])
            &
            (dataframe['btc_rsi_30m'] >= 50)
            &
            ema_check
        )
        dataframe.loc[buy_offset_ema, 'enter_tag'] += 'ema_strong '
        conditions.append(buy_offset_ema)

        buy_offset_ema_2 = (
            (dataframe['close'] < dataframe['ema_offset_buy'])
            &
            (dataframe['btc_rsi_30m'] < 50)
            &
            ema_check
        )
        dataframe.loc[buy_offset_ema_2, 'enter_tag'] += 'ema_weak '
        conditions.append(buy_offset_ema_2)

        ema2_check = (
            ((dataframe['close'] < dataframe['ema_offset_sell3']).rolling(2).min() == 0)
        )

        buy_offset_ema2 = (
            ((dataframe['close'] < dataframe['ema_offset_buy2']).rolling(2).min() > 0)
            &
            (dataframe['btc_rsi_30m'] >= 50)
            &
            ema2_check
        )
        dataframe.loc[buy_offset_ema2, 'enter_tag'] += 'ema_2_strong '
        conditions.append(buy_offset_ema2)

        buy_offset_ema2_2 = (
            ((dataframe['close'] < dataframe['ema_offset_buy2']).rolling(2).min() > 0)
            &
            (dataframe['btc_rsi_30m'] < 50)
            &
            ema2_check
        )
        dataframe.loc[buy_offset_ema2_2, 'enter_tag'] += 'ema_2_weak '
        conditions.append(buy_offset_ema2_2)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions)
                &
                add_check,
                'enter_long',
            ]= 1


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # No exit logic here because we want to use custom exit instead

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        if (len(dataframe) > 1):
            previous_candle_1 = dataframe.iloc[-2].squeeze()

        enter_tag = 'empty'
        if hasattr(trade, 'enter_tag') and trade.enter_tag is not None:
            enter_tag = trade.enter_tag
        enter_tags = enter_tag.split()

        # Specific exit logics for trades that have either ema_strong or ema_weak enter tag
        if any(c in ['ema_strong', 'ema_weak'] for c in enter_tags):

            # Checks to mitate colliding signals. Don't exit if the entry signal is triggered
            buy_offset_ema = (
                (current_candle['close'] < current_candle['ema_offset_buy'])
                &
                (current_candle['btc_rsi_30m'] >= 50)
            )

            buy_offset_ema2 = (
                (current_candle['close'] < current_candle['ema_offset_buy'])
                &
                (current_candle['btc_rsi_30m'] < 50)
            )

            if (current_candle['close'] < current_candle['ema_offset_sell']) & (buy_offset_ema == False) & (buy_offset_ema2 == False):
                return f"ema_down ({enter_tag})"

            if (len(dataframe) > 1):
                if (current_candle['close'] < current_candle['ema_offset_sell2']) & (previous_candle_1['close'] < previous_candle_1['ema_offset_sell2']) & (buy_offset_ema == False) & (buy_offset_ema2 == False):
                    return f"ema_down_2 ({enter_tag})"

        # Specific exit logic for ema_2_strong and ema_2_weak enter tags
        if (len(dataframe) > 1):
            if any(c in ['ema_2_strong', 'ema_2_weak'] for c in enter_tags):
                buy_offset_ema = (
                    (current_candle['close'] < current_candle['ema_offset_buy2'])
                    &
                    (previous_candle_1['close'] < previous_candle_1['ema_offset_buy2'])
                    &
                    (current_candle['btc_rsi_30m'] >= 50)
                )

                buy_offset_ema2 = (
                    (current_candle['close'] < current_candle['ema_offset_buy2'])
                    &
                    (previous_candle_1['close'] < previous_candle_1['ema_offset_buy2'])
                    &
                    (current_candle['btc_rsi_30m'] < 50)
                )

                if (current_candle['close'] < current_candle['ema_offset_sell3']) & (previous_candle_1['close'] < previous_candle_1['ema_offset_sell3']) & (buy_offset_ema == False) & (buy_offset_ema2 == False):
                    return f"ema_down_2 ({enter_tag})"

        # Change current profit value to be tied to latest candle's close value, so that backtest and dry/live behavior is the same
        current_profit = trade.calc_profit_ratio(current_candle['close'])

        timeframe_minutes = timeframe_to_minutes(self.timeframe)
        
        if current_time - timedelta(minutes=int(timeframe_minutes * self.sell_clear_old_trade.value)) > trade.open_date_utc:
            if (current_profit >= (-0.01 * self.sell_clear_old_trade_profit.value)):
                return f"sell_old_trade ({enter_tag})"

        if ((current_time - timedelta(minutes=timeframe_minutes)) > trade.open_date_utc):
            if (current_profit > self.sell_min_profit.value):
                return f"take_profit ({enter_tag})"
