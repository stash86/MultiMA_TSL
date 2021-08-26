import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, CategoricalParameter)
from pandas import DataFrame
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_prev_date


###########################################################################################################
##    MultiMA_TSL, modded by stash86, based on SMAOffsetProtectOptV1 (modded by Perkmeister)             ##
##    Based on @Lamborghini Store's SMAOffsetProtect strat, heavily based on @tirail's original SMAOffset##
##                                                                                                       ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##
##                                                                                                       ##
###########################################################################################################

# I hope you do enough testing before proceeding, either backtesting and/or dry run.
# Any profits and losses are all your responsibility

class MultiMA_TSL(IStrategy):
    INTERFACE_VERSION = 2

    buy_params = {
        "base_nb_candles_buy_ema": 6,
        "low_offset_ema": 0.985,
        "rsi_buy_ema": 61,

        "base_nb_candles_buy_trima": 6,
        "low_offset_trima": 0.981,
        "rsi_buy_trima": 59,
    }

    sell_params = {
        "base_nb_candles_sell": 30,
        "high_offset_ema": 1.004,
    }

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    stoploss = -0.15

    # Multi Offset
    base_nb_candles_sell = IntParameter(5, 80, default=20, load=True, space='sell', optimize=False)

    base_nb_candles_buy_ema = IntParameter(5, 80, default=20, load=True, space='buy', optimize=False)
    low_offset_ema = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=False)
    high_offset_ema = DecimalParameter(0.99, 1.1, default=1.012, load=True, space='sell', optimize=False)
    rsi_buy_ema = IntParameter(30, 70, default=61, space='buy', optimize=False)

    base_nb_candles_buy_trima = IntParameter(5, 80, default=20, load=True, space='buy', optimize=True)
    low_offset_trima = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    rsi_buy_trima = IntParameter(30, 70, default=61, space='buy', optimize=True)

    # Protection
    ewo_low = DecimalParameter(
        -20.0, -8.0, default=-20.0, load=True, space='buy', optimize=False)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=6.0, load=True, space='buy', optimize=False)
    fast_ewo = IntParameter(
        10, 50, default=50, load=True, space='buy', optimize=False)
    slow_ewo = IntParameter(
        100, 200, default=200, load=True, space='buy', optimize=False)

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.018

    use_custom_stoploss = True

    protections = [
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 20,
            "trade_limit": 1,
            "stop_duration": 20,
            "required_profit": -0.05
        },
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2
        }
    ]

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if (current_profit > 0.3):
            return 0.05
        elif (current_profit > 0.2):
            return 0.04
        elif (current_profit > 0.1):
            return 0.03
        elif (current_profit > 0.06):
            return 0.02
        elif (current_profit > 0.03):
            return 0.01
        elif (current_profit > 0.018):
            return 0.005

        return 0.99

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        else:
            trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            buy_signal = dataframe.loc[dataframe['date'] < trade_open_date]
            if not buy_signal.empty:
                buy_signal_candle = buy_signal.iloc[-1]
                buy_tag = buy_signal_candle['buy_tag'] if buy_signal_candle['buy_tag'] != '' else 'empty'
        buy_tags = buy_tag.split()

        last_candle = dataframe.iloc[-1].squeeze()

        if (last_candle['close'] > (last_candle['ema_offset_sell'])) :
            return 'sell signal (' + buy_tag +')'

        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if ((rate > last_candle['close'])) : return False

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        current_profit = trade.calc_profit_ratio(rate)
        if (sell_reason.startswith('sell signal (') and (current_profit > 0.018)):
            # Reject sell signal when trailing stoplosses
            return False
        return True
        
    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EWO
        dataframe['ewo'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value)) *self.low_offset_ema.value
        dataframe['trima_offset_buy'] = ta.TRIMA(dataframe, int(self.base_nb_candles_buy_trima.value)) *self.low_offset_trima.value

        dataframe.loc[:, 'buy_tag'] = ''

        buy_offset_ema = (
            (dataframe['close'] < dataframe['ema_offset_buy']) &
            (
                (dataframe['ewo'] < self.ewo_low.value)
                |
                (
                    (dataframe['ewo'] > self.ewo_high.value)
                    &
                    (dataframe['rsi'] < self.rsi_buy_ema.value)
                )
            )
        )
        dataframe.loc[buy_offset_ema, 'buy_tag'] += 'ema '
        conditions.append(buy_offset_ema)

        buy_offset_trima = (
            (dataframe['close'] < dataframe['trima_offset_buy']) &
            (
                (dataframe['ewo'] < self.ewo_low.value)
                |
                (
                    (dataframe['ewo'] > self.ewo_high.value)
                    &
                    (dataframe['rsi'] < self.rsi_buy_trima.value)
                )
            )
        )
        dataframe.loc[buy_offset_trima, 'buy_tag'] += 'trima '
        conditions.append(buy_offset_trima)

        add_check = (
            (dataframe['volume'] > 0)
        )
        
        if conditions:
            dataframe.loc[:, 'buy'] = reduce(lambda x, y: (x | y) & add_check, conditions)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell.value)) *self.high_offset_ema.value

        dataframe.loc[:,'sell'] = 0

        return dataframe


# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif
