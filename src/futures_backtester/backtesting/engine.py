from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from futures_backtester.backtesting.analytics import (
    bootstrap_bands,
    bootstrap_reliability_summary,
    concentration_summary,
    drawdown_summary,
    execution_cost_by_contract,
    execution_cost_summary,
    parameter_grid_summary,
    path_stress_summary,
    performance_metrics,
    rebalance_summary,
    regime_metrics,
    return_distribution_summary,
    rolling_metrics,
    sector_exposure_summary,
    selection_quality_summary,
    signal_contribution_summary,
    walk_forward_summary,
)
from futures_backtester.data import FuturesPanel, make_sample_panel
from futures_backtester.portfolio.construction import (
    apply_vol_target,
    net_strategy_returns,
    risk_parity_weights,
)
from futures_backtester.signals.factors import (
    blended_signal,
    carry_signal,
    daily_returns,
    realized_vol,
    trend_signal,
)


@dataclass(frozen=True)
class BacktestSettings:
    trend_lookbacks: tuple[int, ...] = (63, 126, 252)
    carry_weights: tuple[float, ...] = (0.0, 0.35, 0.65)
    train_days: int = 504
    test_days: int = 126
    target_vol: float = 0.12
    slippage_bps: float = 0.75
    bootstrap_samples: int = 500
    rolling_window: int = 252
    rolling_min_periods: int = 63


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame
    selections: pd.DataFrame
    parameter_grid: pd.DataFrame
    summary: pd.DataFrame
    return_distribution: pd.DataFrame
    rolling_metrics: pd.DataFrame
    regime_metrics: pd.DataFrame
    sector_exposure: pd.DataFrame
    concentration: pd.DataFrame
    drawdowns: pd.DataFrame
    path_stress: pd.DataFrame
    execution_costs: pd.DataFrame
    execution_costs_by_contract: pd.DataFrame
    rebalance: pd.DataFrame
    signal_contribution: pd.DataFrame
    walk_forward_summary: pd.DataFrame
    parameter_grid_summary: pd.DataFrame
    selection_quality: pd.DataFrame
    bootstrap_bands: pd.DataFrame
    bootstrap_reliability: pd.DataFrame


def _candidate_returns(
    panel: FuturesPanel,
    lookback: int,
    carry_weight: float,
    settings: BacktestSettings,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = daily_returns(panel.prices)
    trend = (1.0 - carry_weight) * trend_signal(panel.prices, lookback)
    carry = carry_weight * carry_signal(panel.carry)
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
    return net, weights, trend, carry


def run_backtest(
    panel: FuturesPanel | None = None,
    settings: BacktestSettings | None = None,
) -> BacktestResult:
    settings = settings or BacktestSettings()
    panel = panel or make_sample_panel()
    dates = panel.prices.index

    CandidateResult = tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    candidates: dict[tuple[int, float], CandidateResult] = {}
    for lookback in settings.trend_lookbacks:
        for carry_weight in settings.carry_weights:
            candidates[(lookback, carry_weight)] = _candidate_returns(
                panel,
                lookback,
                carry_weight,
                settings,
            )

    stitched_weights = pd.DataFrame(0.0, index=dates, columns=panel.prices.columns)
    stitched_trend_component = pd.DataFrame(0.0, index=dates, columns=panel.prices.columns)
    stitched_carry_component = pd.DataFrame(0.0, index=dates, columns=panel.prices.columns)
    selections = []
    parameter_grid = []
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
        for params, result in candidates.items():
            test_metrics = performance_metrics(result[0].iloc[test_slice])
            parameter_grid.append(
                {
                    "start": dates[start],
                    "end": dates[test_end - 1],
                    "trend_lookback": params[0],
                    "carry_weight": params[1],
                    "is_selected": params == best,
                    "train_sharpe": scores[params],
                    "test_ann_return": test_metrics["ann_return"],
                    "test_sharpe": test_metrics["sharpe"],
                    "test_max_drawdown": test_metrics["max_drawdown"],
                }
            )
        _, best_weights, best_trend, best_carry = candidates[best]
        stitched_weights.iloc[test_slice, :] = best_weights.iloc[test_slice, :]
        stitched_trend_component.iloc[test_slice, :] = best_trend.iloc[test_slice, :]
        stitched_carry_component.iloc[test_slice, :] = best_carry.iloc[test_slice, :]
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

    stitched_returns = net_strategy_returns(
        stitched_weights,
        daily_returns(panel.prices),
        panel.roll_cost_bps,
        panel.commission_bps,
        slippage_bps=settings.slippage_bps,
    )
    for selection in selections:
        test_metrics = performance_metrics(
            stitched_returns.loc[selection["start"] : selection["end"]]
        )
        selection.update(
            {
                "test_ann_return": test_metrics["ann_return"],
                "test_sharpe": test_metrics["sharpe"],
                "test_max_drawdown": test_metrics["max_drawdown"],
            }
        )

    live = stitched_returns.iloc[settings.train_days :].rename("net_return")
    summary = pd.DataFrame([performance_metrics(live)])
    parameter_grid_frame = pd.DataFrame(parameter_grid)
    return BacktestResult(
        returns=live,
        weights=stitched_weights.loc[live.index],
        selections=pd.DataFrame(selections),
        parameter_grid=parameter_grid_frame,
        summary=summary,
        return_distribution=return_distribution_summary(live),
        rolling_metrics=rolling_metrics(
            live,
            window=settings.rolling_window,
            min_periods=settings.rolling_min_periods,
        ),
        regime_metrics=regime_metrics(live, panel.prices),
        sector_exposure=sector_exposure_summary(stitched_weights.loc[live.index], panel.sectors),
        concentration=concentration_summary(stitched_weights.loc[live.index]),
        drawdowns=drawdown_summary(live),
        path_stress=path_stress_summary(live),
        execution_costs=execution_cost_summary(
            stitched_weights.loc[live.index],
            panel.roll_cost_bps,
            panel.commission_bps,
            settings.slippage_bps,
        ),
        execution_costs_by_contract=execution_cost_by_contract(
            stitched_weights.loc[live.index],
            panel.roll_cost_bps,
            panel.commission_bps,
            settings.slippage_bps,
        ),
        rebalance=rebalance_summary(stitched_weights.loc[live.index]),
        signal_contribution=signal_contribution_summary(
            stitched_weights.loc[live.index],
            daily_returns(panel.prices).loc[live.index],
            stitched_trend_component.loc[live.index],
            stitched_carry_component.loc[live.index],
        ),
        walk_forward_summary=walk_forward_summary(pd.DataFrame(selections)),
        parameter_grid_summary=parameter_grid_summary(parameter_grid_frame),
        selection_quality=selection_quality_summary(parameter_grid_frame),
        bootstrap_bands=bootstrap_bands(live, samples=settings.bootstrap_samples),
        bootstrap_reliability=bootstrap_reliability_summary(
            live,
            samples=settings.bootstrap_samples,
        ),
    )
