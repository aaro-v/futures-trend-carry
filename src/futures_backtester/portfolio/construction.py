from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _turnover_costs(
    weights: np.ndarray,
    commission_bps: np.ndarray,
    slippage_bps: float,
) -> np.ndarray:
    rows, cols = weights.shape
    costs = np.zeros(rows)
    for i in range(1, rows):
        day_cost = 0.0
        for j in range(cols):
            turnover = abs(weights[i, j] - weights[i - 1, j])
            day_cost += turnover * (commission_bps[j] + slippage_bps) / 10_000.0
        costs[i] = day_cost
    return costs


def risk_parity_weights(
    signal: pd.DataFrame,
    vol: pd.DataFrame,
    max_gross: float = 2.5,
) -> pd.DataFrame:
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    raw = signal.clip(-1.0, 1.0) * inv_vol
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    weights = raw.div(gross, axis=0).fillna(0.0)
    return weights.clip(-max_gross, max_gross)


def correlation_scaler(returns: pd.DataFrame, window: int = 126, floor: float = 0.35) -> pd.Series:
    avg_corr = []
    for i in range(len(returns)):
        if i < window:
            avg_corr.append(0.25)
            continue
        corr = returns.iloc[i - window : i].corr().to_numpy()
        upper = corr[np.triu_indices_from(corr, k=1)]
        avg_corr.append(float(np.nanmean(upper)))
    avg = pd.Series(avg_corr, index=returns.index).clip(lower=0.0)
    return (1.0 / np.sqrt(1.0 + avg * 5.0)).clip(lower=floor, upper=1.0)


def apply_vol_target(
    base_weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.12,
    window: int = 63,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    pre_returns = (base_weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    realized = pre_returns.rolling(window).std() * np.sqrt(252)
    leverage = (target_vol / realized.replace(0.0, np.nan)).clip(upper=max_leverage).fillna(1.0)
    leverage *= correlation_scaler(returns)
    return base_weights.mul(leverage, axis=0)


def net_strategy_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    roll_cost_bps: pd.DataFrame,
    commission_bps: pd.Series,
    slippage_bps: float = 0.75,
) -> pd.Series:
    aligned_weights = weights.shift(1).fillna(0.0)
    gross = (aligned_weights * returns).sum(axis=1)
    roll_cost = (aligned_weights.abs() * roll_cost_bps / 10_000.0).sum(axis=1)
    trade_cost = _turnover_costs(
        weights.to_numpy(dtype=float),
        commission_bps.reindex(weights.columns).to_numpy(dtype=float),
        slippage_bps,
    )
    return gross - roll_cost - pd.Series(trade_cost, index=weights.index)
