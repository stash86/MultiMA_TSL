# MultiMA_TSL

## Important note!!! As of today, seems like there is a bug in Freqtrade's backtesting where a trailing stoploss triggered below 5 minutes duration will be treated as a loss. So the backtest result would be bad. A workaround for now is using the "normal" trailing stoploss option below and disabling `use_custom_stoploss `
```
trailing_stop = True
trailing_only_offset_is_reached = True
trailing_stop_positive = 0.001
trailing_stop_positive_offset = 0.018
```

Strategy for Freqtrade https://github.com/freqtrade/freqtrade

This strategy based on SMAOffsetProtectOptV1 (modded by Perkmeister), based on @Lamborghini Store's SMAOffsetProtect strat, heavily based on @tirail's original SMAOffset

Config files that I'm using on my live bot also can be found in the user_data folder.
