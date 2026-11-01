# Trend and carry across futures

Do simple trend and carry rules earn a premium after trading costs across
futures? Over 2005–2026, the combined, trend-only and carry-only net Sharpe ratios
are 0.82 (0.72 with next-session execution), 0.77 and 0.51. Modeled costs reduce
the combined annual return by 2.3 percentage points, and its 95% bootstrap
Sharpe interval is 0.40–1.23.
Carry covers half the universe, and energy carry ends in 2024.

## Data

The panel contains 44 US-listed contracts across eight sectors, starting in
2003. Yahoo Finance supplies front-month prices,
[FRED](https://fred.stlouisfed.org/series/DTB3) supplies rates and yield curves,
and the [EIA](https://www.eia.gov/dnav/pet/pet_pri_fut_s1_d.htm) supplies energy
settlements. Energy returns use nearby contracts; other roll jumps are modeled
or their switch-day returns are discarded when curves are unavailable.
Downloaded histories stay local under `data/real/` and are excluded from Git.

Carry estimates long-position roll yield from dividends less financing,
Treasury yield and roll-down, FX rate differentials, or nearby energy prices.
Synthetic data checks that the backtester and reports run reproducibly.

## Method

Trend scales trailing price changes by volatility. Carry combines its level
scaled by cross-contract dispersion with a centered rank; missing carry gets
zero carry weight. Contracts use inverse-volatility sizing within sectors,
with equal forecast sector risk budgets. A trailing covariance estimate targets
12% portfolio volatility. Gross notional is capped at eight times equity.

Walk-forward selection compares trend lookbacks of 63, 126 and 252 sessions
and carry weights of 0, 0.35 and 0.65. It selects the highest net training Sharpe
every 126 sessions using the preceding 504 sessions, excluding initial signal
warm-up. The separate sleeves use a fixed 252-session trend or carry alone.

The baseline assumes fills at the signal close and earns the next session's
return. Volatility sizing requires 63 daily returns. Rolls are charged on held
positions; commissions and 0.75-basis-point slippage apply to turnover, including
entries and exits.

## Results

The [saved results](reports/real/summary.csv) cover January 3, 2005 through
October 30, 2026.

| Strategy | Net annual return | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Walk-forward trend and carry | 11.3% | 13.7% | 0.82 | −22.8% |
| Trend only | 10.2% | 13.3% | 0.77 | −24.1% |
| Carry only | 6.5% | 12.8% | 0.51 | −26.8% |

![Net futures strategy equity curves](reports/real/equity_curve.png)

The trend sleeve gained in 2008, 2014 and 2022 and lost money in 2012, 2016 and
2018 ([yearly returns](reports/real/periodic_returns.csv)).

The [robustness checks](reports/real/robustness.csv) use the same dates and risk
settings. Delay and cost scenarios freeze the baseline decisions and targets;
cost multipliers affect commission and slippage, leaving roll costs unchanged.
Next-session execution trades the signal from close t at close t+1.

| Check | Net annual return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Next-session close execution | 9.9% | 0.72 | −25.2% |
| Trading costs ×2 | 9.5% | 0.69 | −30.4% |
| Trading costs ×3 | 7.7% | 0.56 | −37.3% |
| Fixed 252-session trend, carry weight 0.35 | 11.2% | 0.83 | −26.5% |

Selection does not improve full-sample Sharpe over the fixed blend. Held gross
leverage averages 4.4×, with a 95th percentile and maximum of 8.0×; the cap binds
on 5.8% of days. Realized sector contributions to gross portfolio variance range
from 9.0% for rates to 15.6% for FX; the
[sector breakdown](reports/real/robustness_sectors.csv) records every scenario.

## Limitations

Front-month histories and modeled rolls cannot establish exact dated-contract
returns. Macro series can be revised, and the fixed US-listed universe may have
survivorship bias. Bad-print filtering and data-quality start dates also affect
the reconstructed history.

Realized volatility can exceed its target. The model excludes market impact,
margin calls, price limits, collateral interest and funding constraints.
Bootstrap intervals use 500 resamples of 21-session blocks and condition on the
observed strategy path without repeating parameter selection.

## Reproduce

Create an environment with `python -m venv .venv`; on Windows, activate it with
`.venv\Scripts\Activate.ps1`.

```sh
pip install -e ".[dev,data]"
futures-backtest build-data --out data/real --end 2026-10-30
futures-backtest run --data-dir data/real --end 2026-10-30
futures-backtest robustness
```

For synthetic data, run `futures-backtest run --output-dir reports/synthetic`.
Validate with `ruff check .` and `pytest`; CI uses synthetic data only.
