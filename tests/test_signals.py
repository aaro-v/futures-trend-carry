import numpy as np

from futures_backtester.data import make_sample_panel
from futures_backtester.signals import (
    blended_signal,
    carry_signal,
    daily_returns,
    realized_vol,
    trend_signal,
)


def test_signals_are_finite_and_aligned() -> None:
    panel = make_sample_panel(days=300, contracts=8, seed=3)

    returns = daily_returns(panel.prices)
    vol = realized_vol(returns)
    trend = trend_signal(panel.prices, lookback=63)
    carry = carry_signal(panel.carry)
    blend = blended_signal(panel.prices, panel.carry, trend_lookback=63, carry_weight=0.5)

    assert returns.shape == panel.prices.shape
    assert vol.shape == panel.prices.shape
    assert trend.index.equals(panel.prices.index)
    assert carry.columns.equals(panel.prices.columns)
    assert np.isfinite(blend.to_numpy()).all()
    assert blend.abs().max().max() <= 3.0
