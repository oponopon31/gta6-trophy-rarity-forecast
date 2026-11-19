# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oponopon31/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oponopon31/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A program that reads the earn rate of every Grand Theft Auto VI trophy on Exophase and forecasts each trophy's rate a week after launch, as five quantiles per trophy. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## How it works

`main.ts` (Node 22, standard library only) reads every `*.in` case in the input directory and writes one `*.out` forecast per case. For each trophy it fits a recency-weighted line of `logit(rate)` against `ln(1 + hours since 2026-11-19T00:00:00Z)` over all distinct reads (consecutive identical reads are one observation, since Exophase's page only changes when it refreshes), with the slope shrunk towards the tier mean, and reads the line at the target minus half a typical refresh gap. Two priors cover what the history cannot show: an unseen-refresh median shift (the first launch-night refresh cut the average rate to ~3/4 of its old value as players tracked grew 6x) that decays as refreshes are observed, and an honest spread calibrated from the history's own volatility around its trends plus refresh uncertainty — wide enough that one more refresh does not kick every trophy out of its band. A trophy last seen at 90%+ never has its median raised. Any case that fails a step still gets a valid persistence forecast, so no case is ever invalid.

Local validation: on the kit's practice cases p00/p01 the output is valid (55 rows, quantiles non-decreasing, 0–100), two runs are byte-identical, and the run takes ~0.2 s for both cases. The practice has no answers in the kit, so the model was tuned on synthetic backtests of the same generative shape (step refreshes, denominator dilution, tiered saturating adoption, per-refresh noise): there it beats both house baselines on final-like cases (mean pinball loss ~0.27 vs ~0.57 persistence and ~0.49 house curve) and on practice-like cases (~0.41 vs ~0.62).

Run it locally with:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```
