# Trend & Carry Backtester for Global Futures

Vectorized Python research engine for trend-following and carry strategies
across a diversified global futures universe. The project is designed as a
reviewer-ready research engineering sample: deterministic synthetic data,
transparent assumptions, reproducible CLI outputs, and no dependency on private
data feeds.

## Target Resume Bullets

- Built a vectorized futures backtester for trend-following and carry signals
  across roughly 50 liquid contracts, with volatility targeting, risk parity,
  and dynamic correlation-based portfolio scaling.
- Implemented walk-forward parameter selection, regime-conditioned analytics,
  and bootstrap confidence bands for Sharpe and tail metrics.
- Modeled contract rolls, slippage, commissions, execution costs, and turnover
  explicitly so strategy returns are reported net of realistic frictions.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Run The Sample Backtest

```bash
futures-backtest run --output-dir reports/generated
```

The command builds deterministic synthetic futures data in memory, estimates
trend and carry signals, chooses parameters with walk-forward validation, sizes
contracts with volatility targeting and risk parity, applies correlation-based
portfolio scaling, subtracts roll and execution costs, and writes:

- `reports/generated/summary.csv`
- `reports/generated/rolling_metrics.csv`
- `reports/generated/daily_returns.csv`
- `reports/generated/weights.csv`
- `reports/generated/parameter_grid.csv`
- `reports/generated/parameter_grid_summary.csv`
- `reports/generated/regime_metrics.csv`
- `reports/generated/sector_exposure.csv`
- `reports/generated/concentration.csv`
- `reports/generated/drawdowns.csv`
- `reports/generated/execution_costs.csv`
- `reports/generated/execution_costs_by_contract.csv`
- `reports/generated/signal_contribution.csv`
- `reports/generated/walk_forward_summary.csv`
- `reports/generated/bootstrap_bands.csv`
- `reports/generated/bootstrap_reliability.csv`
- `reports/generated/equity_curve.png`
- `reports/generated/research_report.md`
- `reports/generated/output_manifest.csv`

## Sample Results

The chart below is produced with `futures-backtest run --output-dir reports/generated`
from deterministic synthetic futures data. It is a reproducibility artifact for
reviewing signal construction, walk-forward selection, sizing, costs, and
reporting, not a live-market performance claim.

![Synthetic trend and carry futures equity curve](reports/generated/equity_curve.png)

For a faster smoke run:

```bash
futures-backtest run --contracts 24 --days 900 --bootstrap-samples 100
```

To write only the deterministic sample panel:

```bash
futures-backtest sample-data --output data/cache/sample_futures.parquet
```

## Implemented Scope

- Deterministic synthetic futures universe spanning equity index, rates,
  currencies, energy, metals, grains, softs, and livestock.
- Continuous front-adjusted price series, carry estimates, roll events,
  sector labels, realized volatility, and liquidity/cost metadata.
- Trend and carry signals with rolling lookback windows and signal clipping.
- Vectorized weight construction with volatility targeting, inverse-volatility
  risk parity, turnover-aware execution costs, roll costs, and slippage.
- Walk-forward parameter selection over trend/carry blends and lookbacks.
- Portfolio returns recomputed from the stitched walk-forward weight path so
  parameter switches use actual prior holdings and incur consistent transition
  costs.
- Walk-forward stability diagnostics covering selected parameter diversity,
  parameter switching, train/test Sharpe decay, positive test-window rate, and
  selected-window drawdown severity.
- Full walk-forward parameter-grid diagnostics showing train and test outcomes
  for selected and unselected candidates.
- Regime-conditioned performance analytics and bootstrap confidence bands for
  Sharpe, Sortino, Calmar, maximum drawdown, CVaR, and annualized return.
- Bootstrap reliability diagnostics covering positive-return probability,
  positive-Sharpe probability, drawdown breach probability, and median
  resampled tail risk.
- Rolling out-of-sample risk diagnostics for time-varying annualized return,
  volatility, Sharpe, and drawdown review.
- Sector exposure summaries to make concentration and diversification review
  reproducible from each generated run.
- Contract concentration diagnostics, including top-5 gross exposure share,
  HHI, and effective contract count.
- Drawdown episode diagnostics covering depth, length, trough, and recovery.
- Turnover and execution cost diagnostics split across roll, commission, and
  slippage drag.
- Signal contribution diagnostics attributing selected walk-forward gross
  return and allocation exposure to trend and carry components.
- Deterministic output manifests with byte sizes, SHA-256 checksums, and CSV
  dimensions for generated research artifacts.

The generated outputs use synthetic data and should not be interpreted as
evidence of live-market profitability.

## Project Layout

```text
src/futures_backtester/
  data/          Synthetic futures universe and panel serialization
  signals/       Trend, carry, volatility, and blended signal logic
  portfolio/     Risk parity, volatility targeting, and cost-aware returns
  backtesting/   Walk-forward engine and analytics summaries
  reporting/     Reserved package boundary for report-specific expansion
  cli.py         Typer command line interface
tests/           Unit and CLI coverage for each layer
reports/         Reproducible generated research artifacts
```
