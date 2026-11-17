# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/4Quaranz/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/4Quaranz/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts, hour by hour, how many people will be watching the Grand Theft Auto VI category live on Twitch over the 24 UTC hours of its release day (2026-11-19), from the first hour's print alone: five quantiles (5/25/50/75/95%) for each of hours 2–24.

Method: a launch day's shape depends strongly on how awake its category already is in hour 1. For each hour, the program regresses the comparables' log ratio ln(viewers_h / viewers_1) on ln(viewers_1) and takes the fitted value at this case's hour-1 print as the median log-ratio; the spread comes from the residuals of the 12 comparables nearest in ln(viewers_1), so a huge hour 1 gets a tighter band than a small one. It falls back to pooled empirical quantiles, then to the house persistence baseline, if the case carries too few comparables. Python 3.13, standard library only, offline, deterministic.

Practice validation (four past launches' real SullyGnome curves, training only on the 37 comparables the practice cases carry): mean pinball loss on the log scale 0.213, against the house baseline's 0.387.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=scveefga&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
