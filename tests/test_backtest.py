from dataclasses import replace

import numpy as np
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from futures_backtester import BacktestSettings, run_backtest
from futures_backtester.data import make_sample_panel
from futures_backtester.portfolio import return_components
from futures_backtester.signals import daily_returns


@pytest.fixture(scope="module")
def research_run():
    panel = make_sample_panel(days=940, contracts=24, seed=5)
    # A late listing and an interruption must not inherit future volatility.
    panel.prices.iloc[:600, -1] = np.nan
    panel.carry.iloc[:600, -1] = np.nan
    settings = BacktestSettings(bootstrap_samples=10)
    return panel, settings, run_backtest(panel, settings)


def test_stitched_returns_reconcile_and_costs_reduce_gross(research_run) -> None:
    panel, settings, result = research_run
    expected = return_components(
        result.weights,
        daily_returns(panel.prices),
        panel.roll_cost_bps,
        panel.commission_bps,
        settings.slippage_bps,
    )
    assert_series_equal(result.returns, expected.net_return.loc[result.returns.index])
    assert result.weights.iloc[:663, -1].eq(0).all()
    assert np.isfinite(result.weights.to_numpy()).all()
    assert (result.weights.abs().sum(axis=1) <= settings.max_leverage + 1e-10).all()
    costs = result.costs
    np.testing.assert_allclose(costs.ann_gross_return - costs.ann_net_return, costs.ann_total_cost)
    for _, group in result.sectors.groupby("strategy"):
        assert group.risk_contribution.sum() == pytest.approx(1)
    assert result.sleeve_weights["carry"].loc[:, panel.carry.isna().all()].eq(0).all().all()


@pytest.mark.parametrize("cut", [756, 801])
def test_truncation_does_not_change_past_weights_or_returns(research_run, cut) -> None:
    panel, settings, full = research_run
    short = replace(
        panel,
        **{
            name: getattr(panel, name).iloc[:cut]
            for name in ("prices", "carry", "roll_cost_bps", "commission_bps")
        },
    )
    result = run_backtest(short, settings)
    assert_frame_equal(full.weights.iloc[:cut], result.weights)
    assert_series_equal(full.returns.loc[result.returns.index], result.returns)
    for name in ("trend", "carry"):
        assert_frame_equal(full.sleeve_weights[name].iloc[:cut], result.sleeve_weights[name])


def test_selection_uses_only_training_scores(research_run) -> None:
    _, _, result = research_run
    for _, window in result.parameter_grid.groupby("decision_date"):
        selected = window.loc[window.is_selected]
        assert len(selected) == 1
        assert selected.train_sharpe.iloc[0] == window.train_sharpe.max()
        assert (window.train_end < window.start).all()


def test_short_panel_raises_a_clear_error() -> None:
    with pytest.raises(ValueError, match="Not enough history"):
        run_backtest(make_sample_panel(days=300, contracts=2))
