from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from futures_backtester.analytics import (
    bootstrap_bands,
    performance_metrics,
    regime_metrics,
    sector_exposure_summary,
)
from futures_backtester.data import FuturesPanel, make_sample_panel
from futures_backtester.portfolio import apply_vol_target, net_strategy_returns, risk_parity_weights
from futures_backtester.signals import blended_signal, daily_returns, realized_vol


@dataclass(frozen=True)
class BacktestSettings:
    trend_lookbacks: tuple[int, ...] = (63, 126, 252)
    carry_weights: tuple[float, ...] = (0.0, 0.35, 0.65)
    train_days: int = 504
    test_days: int = 126
    target_vol: float = 0.12
    slippage_bps: float = 0.75
    bootstrap_samples: int = 500


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame
    selections: pd.DataFrame
    summary: pd.DataFrame
    regime_metrics: pd.DataFrame
    sector_exposure: pd.DataFrame
    bootstrap_bands: pd.DataFrame


def _candidate_returns(
    panel: FuturesPanel,
    lookback: int,
    carry_weight: float,
    settings: BacktestSettings,
) -> tuple[pd.Series, pd.DataFrame]:
    returns = daily_returns(panel.prices)
    signal = blended_signal(panel.prices, panel.carry, lookback, carry_weight)
    base = risk_parity_weights(signal, realized_vol(returns))
    weights = apply_vol_target(base, returns, target_vol=settings.target_vol)
    net = net_strategy_returns(
        weights,
        returns,
        panel.roll_cost_bps,
        panel.commission_bps,
        slippage_bps=settings.slippage_bps,
    )
    return net, weights


def run_backtest(
    panel: FuturesPanel | None = None,
    settings: BacktestSettings | None = None,
) -> BacktestResult:
    settings = settings or BacktestSettings()
    panel = panel or make_sample_panel()
    dates = panel.prices.index

    candidates: dict[tuple[int, float], tuple[pd.Series, pd.DataFrame]] = {}
    for lookback in settings.trend_lookbacks:
        for carry_weight in settings.carry_weights:
            candidates[(lookback, carry_weight)] = _candidate_returns(
                panel,
                lookback,
                carry_weight,
                settings,
            )

    stitched_returns = pd.Series(0.0, index=dates, name="net_return")
    stitched_weights = pd.DataFrame(0.0, index=dates, columns=panel.prices.columns)
    selections = []
    start = settings.train_days
    while start < len(dates):
        train_slice = slice(start - settings.train_days, start)
        test_end = min(start + settings.test_days, len(dates))
        test_slice = slice(start, test_end)

        scores = {
            params: performance_metrics(result[0].iloc[train_slice])["sharpe"]
            for params, result in candidates.items()
        }
        best = max(scores, key=scores.get)
        best_returns, best_weights = candidates[best]
        stitched_returns.iloc[test_slice] = best_returns.iloc[test_slice]
        stitched_weights.iloc[test_slice, :] = best_weights.iloc[test_slice, :]
        selections.append(
            {
                "start": dates[start],
                "end": dates[test_end - 1],
                "trend_lookback": best[0],
                "carry_weight": best[1],
                "train_sharpe": scores[best],
            }
        )
        start = test_end

    live = stitched_returns.iloc[settings.train_days :].rename("net_return")
    summary = pd.DataFrame([performance_metrics(live)])
    return BacktestResult(
        returns=live,
        weights=stitched_weights.loc[live.index],
        selections=pd.DataFrame(selections),
        summary=summary,
        regime_metrics=regime_metrics(live, panel.prices),
        sector_exposure=sector_exposure_summary(stitched_weights.loc[live.index], panel.sectors),
        bootstrap_bands=bootstrap_bands(live, samples=settings.bootstrap_samples),
    )
