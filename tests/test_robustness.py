import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from futures_backtester import BacktestSettings, run_backtest
from futures_backtester.backtesting.robustness import (
    robustness_tables,
    scenario_components,
    write_robustness,
)
from futures_backtester.data import FuturesPanel, make_sample_panel, write_sample_data


def test_next_session_close_execution_earns_returns_two_sessions_after_signal() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    panel = FuturesPanel(
        prices=pd.DataFrame({"A": [100, 101, 103.02, 106.1106, 110.355024]}, index=dates),
        carry=pd.DataFrame({"A": 0.0}, index=dates),
        roll_cost_bps=pd.DataFrame({"A": [0, 0, 3, 4, 0]}, index=dates),
        commission_bps=pd.Series({"A": 1.0}),
        sectors=pd.Series({"A": "equity"}),
    )
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=dates)
    _, baseline = scenario_components(panel, weights, BacktestSettings())
    _, delayed = scenario_components(panel, weights, BacktestSettings(), execution_delay=1)
    np.testing.assert_allclose(baseline.gross_return, [0, 0.01, 0.02, 0, 0], atol=1e-14)
    np.testing.assert_allclose(delayed.gross_return, [0, 0, 0.02, 0.03, 0], atol=1e-14)
    np.testing.assert_allclose(delayed.roll_cost, [0, 0, 0.0003, 0.0004, 0])
    assert delayed.commission_cost.iloc[1] == 0
    assert delayed.commission_cost.iloc[2] == pytest.approx(0.0001)


def test_robustness_preserves_allocations_and_scales_only_trading_costs() -> None:
    panel = make_sample_panel(days=780, contracts=12)
    settings = BacktestSettings(bootstrap_samples=5)
    result = run_backtest(panel, settings)
    original = result.weights.copy()
    summary, daily, sectors = robustness_tables(panel, result, settings)
    assert_frame_equal(original, result.weights)
    paths = {name: group.set_index("date") for name, group in daily.groupby("scenario")}
    baseline = paths["baseline"]
    np.testing.assert_allclose(baseline.net_return, result.returns)
    for multiplier in (2, 3):
        stressed = paths[f"trading_costs_{multiplier}x"]
        np.testing.assert_allclose(stressed.gross_return, baseline.gross_return)
        np.testing.assert_allclose(stressed.roll_cost, baseline.roll_cost)
        np.testing.assert_allclose(stressed.commission_cost, baseline.commission_cost * multiplier)
        np.testing.assert_allclose(stressed.slippage_cost, baseline.slippage_cost * multiplier)
        np.testing.assert_allclose(
            stressed.net_return,
            baseline.net_return
            - (multiplier - 1) * (baseline.commission_cost + baseline.slippage_cost),
        )
    assert summary.observations.nunique() == 1
    for name, group in sectors.groupby("scenario"):
        assert group.risk_contribution.sum() == pytest.approx(1)
        leverage = paths[name].gross_leverage
        row = summary.set_index("scenario").loc[name]
        assert row.gross_leverage_mean == pytest.approx(leverage.mean())
        assert row.gross_leverage_p95 == pytest.approx(leverage.quantile(0.95))
        assert row.gross_leverage_max == pytest.approx(leverage.max())


def test_changed_input_hash_prevents_robustness_output(tmp_path) -> None:
    data = tmp_path / "data"
    write_sample_data(data, days=80, contracts=2)
    report = tmp_path / "report"
    report.mkdir()
    (report / "run_config.json").write_text(
        json.dumps({"data_dir": str(data), "input_sha256": {"panel.parquet": "changed"}})
    )
    with pytest.raises(ValueError, match="Input hashes differ"):
        write_robustness(report)
    assert not list(report.glob("robustness*"))


def test_fixed_blend_matches_a_single_candidate_run() -> None:
    panel = make_sample_panel(days=780, contracts=12)
    settings = BacktestSettings(trend_lookbacks=(252,), carry_weights=(0.35,), bootstrap_samples=5)
    result = run_backtest(panel, settings)
    _, daily, _ = robustness_tables(panel, result, settings)
    fixed = daily.loc[daily.scenario == "fixed_blend_252_0.35"].set_index("date")
    np.testing.assert_allclose(fixed.net_return, result.returns, atol=1e-14)
