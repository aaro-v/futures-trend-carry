import numpy as np

from futures_backtester.backtest import BacktestSettings, run_backtest
from futures_backtester.data import make_sample_panel


def test_run_backtest_outputs_reviewer_artifacts() -> None:
    panel = make_sample_panel(days=780, contracts=12, seed=5)
    settings = BacktestSettings(
        trend_lookbacks=(63, 126),
        carry_weights=(0.0, 0.5),
        train_days=252,
        test_days=126,
        bootstrap_samples=40,
    )
    result = run_backtest(panel, settings)

    assert not result.returns.empty
    assert result.weights.shape[0] == len(result.returns)
    assert not result.selections.empty
    expected_columns = {"ann_return", "ann_vol", "sharpe", "max_drawdown", "cvar_5"}
    assert expected_columns <= set(result.summary.columns)
    assert not result.regime_metrics.empty
    assert not result.sector_exposure.empty
    assert {"avg_gross_weight", "avg_net_weight", "max_gross_weight", "days_active"} <= set(
        result.sector_exposure.columns
    )
    assert set(result.bootstrap_bands.index) == {"p05", "p50", "p95"}
    assert np.isfinite(result.summary.to_numpy()).all()
