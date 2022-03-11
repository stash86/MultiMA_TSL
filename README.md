# MultiMA_TSL

## Update 2022-3-11
Thanks for all who are/were using MultiMA_TSL. Hopefully you would get a strategy that works for you. And for those who want to follow my updates on future strats, you can folow me on https://twitter.com/Crypto_Pratama

## Update 2022-2-23
The last free version is available as paid early access on the patreon https://www.patreon.com/stash86
It will be live here in 2 weeks from now, so your choice whether you want to pay for it or waiting for the free version.

## Update 2022-2-22
There will be one last free update that should be uploaded in 3 weeks. After that, I'll have it behind a paywall. The strategy getting heavy considerably that using free 3vCPU ARM Oracle VPS isn't enough anymore. For 1 run of hyperopt of 500 epochs, I would need to wait for at least 20 hours. So I'm thinking of using paid VPS. 4 CPU 8GB Memory on vultr would cost $40/month today. That's why I'm putting it behind paywall. The link to my patreon is
https://www.patreon.com/stash86

Another important update is that I change the EWO calculation to use SMA instead of SMA. It would affect all 3 versions of the strat.

## Important note!!! This strategy is developed and tested using BUSD pairs' data. So it might give better result if it's being used when BUSD is used as the stake, compared to other stake.

## Also in this release, I include 2 classes in the strategy file. The latest version should give you higher average profit, higher total profit, but will give you longer average duration and lower win ratio. Which one you want to use, it's up to you. That's why I include them both

Strategy for Freqtrade https://github.com/freqtrade/freqtrade

This strategy based on SMAOffsetProtectOptV1 (modded by Perkmeister), based on @Lamborghini Store's SMAOffsetProtect strat, heavily based on @tirail's original SMAOffset

Config files that I'm using on my live bot also can be found in the user_data folder.

## Affiliate links
Binance (you and I will both receive 10% commisions) - https://accounts.binance.com/en/register?ref=QIC2NUO9
Kucoin - https://www.kucoin.com/ucenter/signup?rcode=r3BWY2T
FTX - https://ftx.com/profile#a=stash86