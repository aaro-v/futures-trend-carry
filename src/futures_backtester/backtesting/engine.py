from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from futures_backtester.data import FuturesPanel, make_sample_panel
from futures_backtester.portfolio import (
    apply_vol_target,
    inverse_volatility_weights,
    return_components,
    rolling_covariance,
)
from futures_backtester.signals import blended_signal, carry_signal, daily_returns, realized_vol

from .analytics import (
    bootstrap_bands,
    cost_summary,
    drawdown_summary,
    performance_metrics,
    periodic_returns,
    regime_metrics,
    rolling_metrics,
    sector_summary,
)


@dataclass(frozen=True)
class BacktestSettings:
    trend_lookbacks: tuple[int, ...] = (63, 126, 252)
    carry_weights: tuple[float, ...] = (0.0, 0.35, 0.65)
    train_days: int = 504
    test_days: int = 126
    target_vol: float = 0.12
    max_leverage: float = 8.0
    covariance_window: int = 63
    covariance_shrinkage: float = 0.10
    slippage_bps: float = 0.75
    bootstrap_samples: int = 500
    bootstrap_block: int = 21
    bootstrap_seed: int = 7
    report_start: str | None = None


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame  # Close targets, including training history and the last close.
    sleeve_weights: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    daily: pd.DataFrame
    periodic: pd.DataFrame
    sectors: pd.DataFrame
    parameter_grid: pd.DataFrame
    bootstrap: pd.DataFrame
    costs: pd.DataFrame
    regimes: pd.DataFrame
    drawdowns: pd.DataFrame

    @property
    def selections(self) -> pd.DataFrame:
        return self.parameter_grid.loc[self.parameter_grid["is_selected"]].reset_index(drop=True)


def run_backtest(
    panel: FuturesPanel | None = None, settings: BacktestSettings | None = None
) -> BacktestResult:
    settings = settings or BacktestSettings()
    panel = panel if panel is not None else make_sample_panel()
    dates = panel.prices.index
    if (
        settings.train_days <= max(settings.trend_lookbacks) + 1
        or settings.test_days < 1
        or settings.bootstrap_samples < 1
        or not settings.carry_weights
        or settings.target_vol <= 0
        or settings.max_leverage <= 0
    ):
        raise ValueError("Invalid settings: training must exceed the longest trend warm-up.")
    first = settings.train_days
    if settings.report_start is not None:
        first = max(first, int(dates.searchsorted(pd.Timestamp(settings.report_start))))
    if first >= len(dates):
        raise ValueError("Not enough history for training and at least one out-of-sample return.")
    asset_returns = daily_returns(panel.prices)
    vol = realized_vol(asset_returns, settings.covariance_window)
    covariance = rolling_covariance(
        asset_returns, settings.covariance_window, settings.covariance_shrinkage
    )

    def weights_for(lookback: int, carry_weight: float) -> pd.DataFrame:
        signal = (
            carry_signal(panel.carry)
            if carry_weight == 1.0
            else blended_signal(panel.prices, panel.carry, lookback, carry_weight)
        )
        base = inverse_volatility_weights(signal, vol, panel.sectors, covariance)
        return apply_vol_target(base, covariance, settings.target_vol, settings.max_leverage)

    def components(weights: pd.DataFrame) -> pd.DataFrame:
        return return_components(
            weights,
            asset_returns,
            panel.roll_cost_bps,
            panel.commission_bps,
            settings.slippage_bps,
        )

    candidates = {
        (lookback, blend): weights_for(lookback, blend)
        for lookback in settings.trend_lookbacks
        for blend in settings.carry_weights
    }
    candidate_returns = {params: components(w)["net_return"] for params, w in candidates.items()}
    stitched = pd.DataFrame(0.0, index=dates, columns=panel.prices.columns)
    windows = []
    # Selection occurs at the previous close, with no test returns in the score.
    # Include a decision at the final close so targets are invariant to truncation.
    for decision in range(first - 1, len(dates), settings.test_days):
        train_start = max(decision - settings.train_days + 1, max(settings.trend_lookbacks) + 1)
        train = slice(train_start, decision + 1)
        scores = {
            params: performance_metrics(r.iloc[train])["sharpe"]
            for params, r in candidate_returns.items()
        }
        best = max(scores, key=scores.get)
        end = min(decision + settings.test_days, len(dates))
        stitched.iloc[decision:end] = candidates[best].iloc[decision:end]
        if decision + 1 < len(dates):
            windows.append((decision, min(end, len(dates) - 1), train_start, scores, best))

    sleeves = {"combined": stitched, "trend": weights_for(252, 0.0), "carry": weights_for(252, 1.0)}
    for name in ("trend", "carry"):
        sleeves[name].iloc[: first - 1] = 0.0
    live = dates[first:]
    daily = {name: components(w).loc[live] for name, w in sleeves.items()}
    grid = []
    for decision, end, train_start, scores, best in windows:
        for params, r in candidate_returns.items():
            test = (daily["combined"]["net_return"] if params == best else r).loc[
                dates[decision + 1] : dates[end]
            ]
            grid.append(
                {
                    "decision_date": dates[decision],
                    "train_start": dates[train_start],
                    "train_end": dates[decision],
                    "start": dates[decision + 1],
                    "end": dates[end],
                    "trend_lookback": params[0],
                    "carry_weight": params[1],
                    "is_selected": params == best,
                    "train_sharpe": scores[params],
                    **{f"test_{k}": v for k, v in performance_metrics(test).items()},
                }
            )

    equity_names = panel.sectors.index[panel.sectors == "equity_index"]
    benchmark = asset_returns[equity_names].mean(axis=1)
    summary, periods, sectors, bootstrap, costs, regimes, drawdowns = [], [], [], [], [], [], []
    for name, frame in daily.items():
        r = frame["net_return"]
        frame["equity"] = (1 + r).cumprod()
        frame["drawdown"] = frame["equity"] / frame["equity"].cummax().clip(lower=1) - 1
        frame[["rolling_vol", "rolling_sharpe"]] = rolling_metrics(r)
        frame["equity_index_return"] = benchmark
        risk = sleeves[name].shift(1).loc[live].abs().sum(axis=1)
        corr = r.corr(benchmark)
        cost = cost_summary(frame)
        summary.append(
            {
                "strategy": name,
                "start": live[0],
                "end": live[-1],
                "observations": len(r),
                **performance_metrics(r),
                "ann_gross_return": cost["ann_gross_return"],
                "ann_cost_drag": cost["ann_total_cost"],
                "equity_index_correlation": corr if np.isfinite(corr) else 0.0,
                "avg_gross_weight": risk.mean(),
                "leverage_cap_fraction": (risk >= settings.max_leverage - 1e-8).mean(),
            }
        )
        periods.append(periodic_returns(r).assign(strategy=name))
        sectors.append(
            sector_summary(sleeves[name], asset_returns, panel.sectors, live).assign(strategy=name)
        )
        bootstrap.append(
            bootstrap_bands(
                r, settings.bootstrap_samples, settings.bootstrap_block, settings.bootstrap_seed
            ).assign(strategy=name)
        )
        costs.append({"strategy": name, **cost})
        regimes.append(regime_metrics(r, benchmark).assign(strategy=name))
        drawdowns.append(drawdown_summary(r).assign(strategy=name))
    return BacktestResult(
        returns=daily["combined"]["net_return"],
        weights=stitched,
        sleeve_weights=sleeves,
        summary=pd.DataFrame(summary),
        daily=pd.concat(daily, names=["strategy", "date"]).reset_index(),
        periodic=pd.concat(periods, ignore_index=True),
        sectors=pd.concat(sectors, ignore_index=True),
        parameter_grid=pd.DataFrame(grid),
        bootstrap=pd.concat(bootstrap, ignore_index=True),
        costs=pd.DataFrame(costs),
        regimes=pd.concat(regimes, ignore_index=True),
        drawdowns=pd.concat(drawdowns, ignore_index=True),
    )
