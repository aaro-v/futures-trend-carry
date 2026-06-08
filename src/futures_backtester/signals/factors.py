from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def realized_vol(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return returns.rolling(window).std().bfill().clip(lower=1e-4)


def trend_signal(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    raw = prices.pct_change(lookback)
    vol = prices.pct_change().rolling(lookback).std() * np.sqrt(lookback)
    signal = raw / vol.replace(0.0, np.nan)
    return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)


def carry_signal(carry: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    centered = carry.sub(carry.rolling(window).mean())
    scaled = centered.div(carry.rolling(window).std().replace(0.0, np.nan))
    return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)


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
