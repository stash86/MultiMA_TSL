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
        "base_nb_candles_buy_sma": 76,
        "low_offset_sma": 0.959,
        "rsi_buy_sma": 55,
        
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
        "sl_filter_candles": 3,
        "sl_filter_offset": 0.992,
    }

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    stoploss = -0.15

    # Multi Offset
    base_nb_candles_sell = IntParameter(5, 80, default=20, load=True, space='sell', optimize=False)

    base_nb_candles_buy_sma = IntParameter(5, 80, default=20, load=True, space='buy', optimize=False)
    low_offset_sma = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=False)
    rsi_buy_sma = IntParameter(30, 70, default=61, space='buy', optimize=False)

    base_nb_candles_buy_ema = IntParameter(5, 80, default=20, load=True, space='buy', optimize=False)
    low_offset_ema = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=False)
    high_offset_ema = DecimalParameter(0.99, 1.1, default=1.012, load=True, space='sell', optimize=False)
    rsi_buy_ema = IntParameter(30, 70, default=61, space='buy', optimize=False)

    base_nb_candles_buy_trima = IntParameter(5, 80, default=20, load=True, space='buy', optimize=False)
    low_offset_trima = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=False)
    rsi_buy_trima = IntParameter(30, 70, default=61, space='buy', optimize=False)

    # Protection
    ewo_low = DecimalParameter(-20.0, -8.0, default=-20.0, load=True, space='buy', optimize=False)
    ewo_high = DecimalParameter(2.0, 12.0, default=6.0, load=True, space='buy', optimize=False)
    fast_ewo = IntParameter(10, 50, default=50, load=True, space='buy', optimize=False)
    slow_ewo = IntParameter(100, 200, default=200, load=True, space='buy', optimize=False)

    # stoploss sharp dip filter
    sl_filter_candles = IntParameter(2, 10, default=5, space='sell', optimize=False, load=True)
    sl_filter_offset = DecimalParameter(0.960, 0.999, default=0.989, decimals=3, space='sell', optimize=False, load=True)

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.018

    use_custom_stoploss = False

    # Protection hyperspace params:
    protection_params = {
        "low_profit_lookback": 60,
        "low_profit_min_req": 0.03,
        "low_profit_stop_duration": 29,
        "cooldown_lookback": 2,  # value loaded from strategy
    }

    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=False)

    low_profit_lookback = IntParameter(2, 60, default=20, space="protection", optimize=False)
    low_profit_stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=False)
    low_profit_min_req = DecimalParameter(-0.05, 0.05, default=-0.05, space="protection", decimals=2, optimize=False)

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        prot.append({
            "method": "LowProfitPairs",
            "lookback_period_candles": self.low_profit_lookback.value,
            "trade_limit": 1,
            "stop_duration": int(self.low_profit_stop_duration.value),
            "required_profit": self.low_profit_min_req.value
        })

        return prot

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if ((rate > last_candle['close'])) : return False

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        # Code from Perkmeister, to check for a sudden dip
        if(sell_reason == 'stop_loss'):
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if(len(dataframe) < 1):
                return True
            last_candle = dataframe.iloc[-1]
            current_profit = trade.calc_profit_ratio(rate)
                
            if(
                (trade.initial_stop_loss == trade.stop_loss) &
                (last_candle['ma_sl_filter_offset'] > rate)
            ):
                # Reject hard stoploss on large dip
                return False

        return True
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.config['runmode'].value == 'hyperopt':
            dataframe['ma_sl_filter_offset'] = ta.EMA(dataframe, timeperiod=int(self.sl_filter_candles.value)) * self.sl_filter_offset.value
        
        # EWO
        dataframe['ewo'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        dataframe['sma_offset_buy'] = ta.SMA(dataframe, int(self.base_nb_candles_buy_sma.value)) *self.low_offset_sma.value
        dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value)) *self.low_offset_ema.value
        dataframe['trima_offset_buy'] = ta.TRIMA(dataframe, int(self.base_nb_candles_buy_trima.value)) *self.low_offset_trima.value
        dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell.value)) *self.high_offset_ema.value

        dataframe.loc[:, 'buy_tag'] = ''

        buy_offset_sma = (
            (dataframe['close'] < dataframe['sma_offset_buy']) &
            (
                (dataframe['ewo'] < self.ewo_low.value)
                |
                (
                    (dataframe['ewo'] > self.ewo_high.value)
                    &
                    (dataframe['rsi'] < self.rsi_buy_sma.value)
                )
            )
        )
        dataframe.loc[buy_offset_sma, 'buy_tag'] += 'sma '
        conditions.append(buy_offset_sma)

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
            (dataframe['rsi_fast'] < 35)
            &
            (dataframe['close'] < dataframe['ema_offset_sell'])
            &
            (dataframe['volume'] > 0)
        )
        
        if conditions:
            dataframe.loc[:, 'buy'] = add_check & reduce(lambda x, y: x | y, conditions)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell.value)) *self.high_offset_ema.value
            dataframe['ma_sl_filter_offset'] = ta.EMA(dataframe, timeperiod=int(self.sl_filter_candles.value)) * self.sl_filter_offset.value

        conditions = []

        conditions.append(
            (
                (dataframe['close'] > dataframe['ema_offset_sell']) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe


# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif
