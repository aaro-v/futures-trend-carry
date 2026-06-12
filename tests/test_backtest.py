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
    assert {
        "start",
        "end",
        "trend_lookback",
        "carry_weight",
        "train_sharpe",
        "test_ann_return",
        "test_sharpe",
        "test_max_drawdown",
    } <= set(result.selections.columns)
    expected_columns = {
        "ann_return",
        "ann_vol",
        "sharpe",
        "downside_vol",
        "sortino",
        "max_drawdown",
        "calmar",
        "cvar_5",
    }
    assert expected_columns <= set(result.summary.columns)
    assert not result.regime_metrics.empty
    assert not result.sector_exposure.empty
    assert {"avg_gross_weight", "avg_net_weight", "max_gross_weight", "days_active"} <= set(
        result.sector_exposure.columns
    )
    assert {
        "avg_gross_weight",
        "avg_max_contract_gross_share",
        "max_contract_gross_share",
        "avg_top_5_gross_share",
        "avg_hhi",
        "effective_contract_count",
    } <= set(result.concentration.columns)
    assert {
        "start",
        "trough",
        "recovery",
        "depth",
        "length_days",
        "recovery_days",
    } <= set(result.drawdowns.columns)
    assert {
        "avg_daily_turnover",
        "annualized_turnover",
        "ann_roll_cost",
        "ann_commission_cost",
        "ann_slippage_cost",
        "ann_trade_cost",
        "ann_total_cost",
        "max_daily_cost",
    } <= set(result.execution_costs.columns)
    assert {
        "windows",
        "unique_parameter_sets",
        "most_common_trend_lookback",
        "most_common_carry_weight",
        "avg_train_sharpe",
        "avg_test_sharpe",
        "sharpe_degradation",
        "positive_test_sharpe_rate",
        "avg_test_max_drawdown",
        "worst_test_max_drawdown",
    } <= set(result.walk_forward_summary.columns)
    assert set(result.bootstrap_bands.index) == {"p05", "p50", "p95"}
    assert np.isfinite(result.summary.to_numpy()).all()
    assert np.isfinite(
        result.selections[["train_sharpe", "test_ann_return", "test_sharpe", "test_max_drawdown"]]
        .to_numpy()
        .astype(float)
    ).all()
    assert np.isfinite(result.concentration.to_numpy()).all()
    assert np.isfinite(result.drawdowns[["depth", "length_days"]].to_numpy()).all()
    assert np.isfinite(result.execution_costs.to_numpy()).all()
    assert np.isfinite(result.walk_forward_summary.to_numpy()).all()
