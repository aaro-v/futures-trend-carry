"""Untuned 12-month sign momentum and cross-sectional carry reference strategies."""

import numpy as np
import pandas as pd

from futures_backtester.data import FuturesPanel
from futures_backtester.portfolio import apply_vol_target, return_components, rolling_covariance
from futures_backtester.signals import daily_returns, realized_vol

from .analytics import performance_metrics


def reference_metrics(panel: FuturesPanel, live: pd.Index) -> dict:
    returns = daily_returns(panel.prices)
    vol = realized_vol(returns)
    covariance = rolling_covariance(returns)
    count = panel.carry.notna().sum(axis=1)
    rank = panel.carry.rank(axis=1).sub(1).div((count - 1).replace(0, np.nan), axis=0) * 2 - 1
    signals = {
        "trend": np.sign(panel.prices.pct_change(252, fill_method=None)),
        "carry": rank.fillna(0),
    }
    paths = {}
    for name, signal in signals.items():
        raw = (signal / vol).fillna(0)
        base = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        weights = apply_vol_target(base, covariance, target_vol=0.10, max_leverage=8)
        first_target = int(weights.index.get_indexer([live[0]])[0]) - 1
        weights.iloc[:first_target] = 0
        paths[name] = return_components(
            weights, returns, panel.roll_cost_bps, panel.commission_bps, slippage_bps=0.75
        ).net_return.loc[live]
    paths["blend"] = (paths["trend"] + paths["carry"]) / 2
    return {
        "target_vol_each_sleeve": 0.10,
        "max_leverage": 8.0,
        "sizing": "global inverse volatility; no sector budgets; blend of separate sleeve returns",
        "metrics": {name: performance_metrics(path) for name, path in paths.items()},
    }
