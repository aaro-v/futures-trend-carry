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
- `reports/generated/daily_returns.csv`
- `reports/generated/weights.csv`
- `reports/generated/regime_metrics.csv`
- `reports/generated/bootstrap_bands.csv`
- `reports/generated/equity_curve.png`
- `reports/generated/research_report.md`

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
- Regime-conditioned performance analytics and bootstrap confidence bands for
  Sharpe, maximum drawdown, CVaR, and annualized return.

The generated outputs use synthetic data and should not be interpreted as
evidence of live-market profitability.
