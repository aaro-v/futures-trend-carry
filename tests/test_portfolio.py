import numpy as np
import pandas as pd
import pytest

from futures_backtester.portfolio import (
    apply_vol_target,
    inverse_volatility_weights,
    return_components,
)


def test_sector_forecast_risk_is_equal_despite_contract_count_and_volatility() -> None:
    signal = pd.DataFrame([[1.0, 1.0, 1.0]], columns=["ES", "NQ", "ZT"])
    cov = np.array([[[0.0004, 0.00032, 0], [0.00032, 0.0004, 0], [0, 0, 0.000004]]])
    vol = pd.DataFrame([[0.02, 0.02, 0.002]], columns=signal.columns)
    sectors = pd.Series({"ES": "equity", "NQ": "equity", "ZT": "rates"})
    w = inverse_volatility_weights(signal, vol, sectors, cov).to_numpy()[0]
    assert np.isclose(w[:2] @ cov[0, :2, :2] @ w[:2], w[2] ** 2 * cov[0, 2, 2])
    scaled = apply_vol_target(pd.DataFrame([w], columns=signal.columns), cov, max_leverage=8)
    assert scaled.abs().sum(axis=1).iloc[0] <= 8


def test_roll_uses_held_position_and_trades_include_opening_and_liquidation() -> None:
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0]})
    r = pd.DataFrame({"A": [0.0, 0.01, 0.02, 0.0]})
    roll = pd.DataFrame({"A": [0.0, 0.0, 3.0, 0.0]})
    result = return_components(w, r, roll, pd.Series({"A": 1.0}), slippage_bps=2)
    assert result["gross_return"].tolist() == [0.0, 0.01, 0.02, 0.0]
    assert result["roll_cost"].tolist() == [0.0, 0.0, 0.0003, 0.0]
    assert result["commission_cost"].tolist() == [0.0, 0.0001, 0.0, 0.0001]
    assert result["slippage_cost"].tolist() == [0.0, 0.0002, 0.0, 0.0002]
    assert result.loc[2, "net_return"] == pytest.approx(0.0197)
    np.testing.assert_allclose(result.net_return, result.gross_return - result.total_cost)


def test_costs_use_daily_contract_notional_instead_of_future_median() -> None:
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0]})
    daily_commission = pd.DataFrame({"A": [1.0, 2.0, 4.0]})
    zero = w * 0
    result = return_components(w, zero, zero, daily_commission, slippage_bps=0)
    np.testing.assert_allclose(result.commission_cost, [0, 0.0001, 0.0002])
