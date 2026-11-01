import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from futures_backtester.data import make_sample_panel
from futures_backtester.signals import carry_signal, daily_returns, realized_vol, trend_signal


def test_carry_levels_and_missing_coverage() -> None:
    carry = pd.DataFrame({"positive": [0.03] * 160, "negative": [-0.03] * 160, "missing": np.nan})
    signal = carry_signal(carry)
    assert (signal["positive"] > 0).all()
    assert (signal["negative"] < 0).all()
    assert (signal["missing"] == 0).all()
    assert (carry_signal(carry * np.nan) == 0).all().all()


def test_volatility_requires_observed_history_for_each_contract() -> None:
    panel = make_sample_panel(days=200, contracts=2)
    prices = panel.prices.copy()
    prices.iloc[:90, 1] = np.nan
    vol = realized_vol(daily_returns(prices))
    assert vol.iloc[:63, 0].isna().all()
    assert vol.iloc[:153, 1].isna().all()
    assert vol.iloc[153:, 1].notna().all()


def test_signals_are_invariant_to_truncation() -> None:
    panel = make_sample_panel(days=420, contracts=25)
    for function in (lambda p: realized_vol(daily_returns(p)), lambda p: trend_signal(p, 126)):
        assert_frame_equal(function(panel.prices).iloc[:300], function(panel.prices.iloc[:300]))
    assert_frame_equal(carry_signal(panel.carry).iloc[:300], carry_signal(panel.carry.iloc[:300]))
