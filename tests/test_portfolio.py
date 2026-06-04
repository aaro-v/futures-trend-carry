import numpy as np

from futures_backtester.data import make_sample_panel
from futures_backtester.portfolio import apply_vol_target, net_strategy_returns, risk_parity_weights
from futures_backtester.signals import blended_signal, daily_returns, realized_vol


def test_portfolio_weights_and_net_returns_are_valid() -> None:
    panel = make_sample_panel(days=360, contracts=10, seed=4)
    returns = daily_returns(panel.prices)
    signal = blended_signal(panel.prices, panel.carry, trend_lookback=63, carry_weight=0.35)
    base = risk_parity_weights(signal, realized_vol(returns))
    weights = apply_vol_target(base, returns)
    net = net_strategy_returns(weights, returns, panel.roll_cost_bps, panel.commission_bps)

    assert weights.shape == panel.prices.shape
    assert np.isfinite(weights.to_numpy()).all()
    assert net.index.equals(panel.prices.index)
    assert np.isfinite(net.to_numpy()).all()
