from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None)


def realized_vol(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return returns.rolling(window, min_periods=window).std().clip(lower=1e-4)


def trend_signal(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    raw = prices.pct_change(lookback, fill_method=None)
    vol = daily_returns(prices).rolling(lookback).std() * np.sqrt(lookback)
    signal = raw / vol.replace(0.0, np.nan)
    return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)


def carry_signal(carry: pd.DataFrame) -> pd.DataFrame:
    """Equal blend of carry level / daily cross-contract dispersion and centered rank."""
    dispersion = carry.std(axis=1).clip(lower=0.005)
    level = carry.div(dispersion, axis=0).clip(-3.0, 3.0)
    count = carry.notna().sum(axis=1)
    rank = carry.rank(axis=1, method="average").sub(1).div((count - 1).replace(0, np.nan), axis=0)
    rank = (2 * rank - 1).fillna(0.0)
    return (0.5 * level + 0.5 * rank).where(carry.notna(), 0.0).fillna(0.0)


def blended_signal(
    prices: pd.DataFrame,
    carry: pd.DataFrame,
    trend_lookback: int,
    carry_weight: float,
) -> pd.DataFrame:
    trend = trend_signal(prices, trend_lookback)
    carry_z = carry_signal(carry)
    blend = (1.0 - carry_weight) * trend + carry_weight * carry_z
    return blend.clip(-3.0, 3.0)
