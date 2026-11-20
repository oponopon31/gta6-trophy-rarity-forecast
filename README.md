# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oponopon31/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oponopon31/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A program that reads the earn rate of every Grand Theft Auto VI trophy on Exophase and forecasts each trophy's rate a week after launch, as five quantiles per trophy. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## How it works

`main.ts` (Node 22, standard library only) reads every `*.in` case in the input directory and writes one `*.out` forecast per case. For each trophy it fits a recency-weighted quadratic of `logit(rate)` against `ln(1 + hours since 2026-11-19T00:00:00Z)` over all distinct reads (consecutive identical reads are one observation, since Exophase's page only changes when it refreshes), with the curvature shrunk towards zero and the quadratic prediction blended 75/25 with a linear trend whose slope is shrunk towards the tier mean, and reads the blend at the target minus half a typical refresh gap. A line would extrapolate the recent decline straight into the floor; the quadratic lets the trend bend toward flat as dilution eases and earners catch up. Two priors cover what the history cannot show: an unseen-refresh median shift (the first launch-night refresh cut the average rate to ~3/4 of its old value as players tracked grew 6x) that decays as refreshes are observed, and an honest spread calibrated from the history's own volatility around its trends plus refresh uncertainty — wide enough that one more refresh does not kick every trophy out of its band. A trophy last seen at 90%+ never has its median raised; the one trophy last seen at 99%+ (everyone tracked holds it) has its median pinned at its last rate with a tight band instead, since trophies are permanent and it cannot fall. Any case that fails a step still gets a valid persistence forecast, so no case is ever invalid.

Local validation: on the kit's practice cases p00/p01 the output is valid (55 rows, quantiles non-decreasing, 0–100), two runs are byte-identical, and the run takes ~0.4 s for both cases. The practice has no answers in the kit, so the model was tuned on synthetic backtests: 24 final-like worlds anchored to the two real snapshots (the launch-night rates at 638 tracked players and the live Exophase rates at ~26 h, fetched 20 Nov), spanning fast/slow tracked-player growth and fast/slow mass adoption. On those worlds the new model scores mean pinball loss 0.22 (worst world 0.34) against 0.35 (worst 0.55) for the previous linear model and ~0.57 for persistence. The practice cases cannot distinguish the variants (their histories are a single flat page), so the backtest, not the practice score, is what the model choice rests on.

Run it locally with:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```
