import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from futures_backtester.backtesting.analytics import (
    bootstrap_bands,
    drawdown_summary,
    performance_metrics,
    periodic_returns,
)


def test_metrics_include_initial_loss_and_distinguish_arithmetic_return_from_cagr() -> None:
    returns = pd.Series([-0.1, 0.02, 0.01])
    metrics = performance_metrics(returns)
    assert metrics["max_drawdown"] == pytest.approx(-0.1)
    assert metrics["ann_return"] == pytest.approx(returns.mean() * 252)
    assert metrics["cagr"] == pytest.approx((1 + returns).prod() ** (252 / len(returns)) - 1)
    assert np.isfinite(list(metrics.values())).all()
    assert set(performance_metrics(pd.Series(dtype=float)).values()) == {0.0}


def test_periodic_returns_compound_and_mark_partial_years() -> None:
    returns = pd.Series(
        [0.1, -0.05, 0.02], index=pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02"])
    )
    yearly = periodic_returns(returns).query("frequency == 'year'")
    np.testing.assert_allclose(yearly["return"], [1.1 * 0.95 - 1, 0.02])
    assert yearly.observations.tolist() == [2, 1]


def test_drawdown_includes_unrecovered_first_day_loss() -> None:
    returns = pd.Series([-0.1, 0.02], index=pd.bdate_range("2024-01-01", periods=2))
    table = drawdown_summary(returns)
    assert table.depth.iloc[0] == pytest.approx(-0.1)
    assert pd.isna(table.recovery.iloc[0])


def test_block_bootstrap_is_reproducible_and_orders_intervals() -> None:
    returns = pd.Series(np.random.default_rng(1).normal(0.0002, 0.01, 500))
    first = bootstrap_bands(returns, samples=30, block=21)
    assert_frame_equal(first, bootstrap_bands(returns, samples=30, block=21))
    assert (first.lower_95 <= first["median"]).all()
    assert (first["median"] <= first.upper_95).all()
