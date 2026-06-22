import numpy as np
import pandas as pd

from futures_backtester.backtest import BacktestSettings, run_backtest
from futures_backtester.data import make_sample_panel
from futures_backtester.portfolio import net_strategy_returns
from futures_backtester.signals import daily_returns


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
    assert not result.parameter_grid.empty
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
    assert {
        "start",
        "end",
        "trend_lookback",
        "carry_weight",
        "is_selected",
        "train_sharpe",
        "test_ann_return",
        "test_sharpe",
        "test_max_drawdown",
    } <= set(result.parameter_grid.columns)
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
    assert {
        "observations",
        "hit_rate",
        "avg_gain",
        "avg_loss",
        "gain_loss_ratio",
        "best_day",
        "worst_day",
        "skew",
        "excess_kurtosis",
        "positive_month_rate",
        "best_month",
        "worst_month",
    } <= set(result.return_distribution.columns)
    assert {
        "window_days",
        "observations",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
    } <= set(result.rolling_metrics.columns)
    assert not result.rolling_metrics.empty
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
        "avg_abs_weight",
        "avg_daily_turnover",
        "annualized_turnover",
        "ann_roll_cost",
        "ann_commission_cost",
        "ann_slippage_cost",
        "ann_trade_cost",
        "ann_total_cost",
        "max_daily_cost",
    } <= set(result.execution_costs_by_contract.columns)
    assert result.execution_costs_by_contract.index.name == "contract"
    assert {
        "ann_gross_return",
        "ann_trend_attributed_return",
        "ann_carry_attributed_return",
        "ann_unassigned_return",
        "avg_abs_trend_component_share",
        "avg_abs_carry_component_share",
        "trend_dominant_gross_share",
        "carry_dominant_gross_share",
        "trend_carry_agreement_rate",
    } <= set(result.signal_contribution.columns)
    assert {
        "windows",
        "unique_parameter_sets",
        "most_common_trend_lookback",
        "most_common_carry_weight",
        "parameter_switch_rate",
        "avg_trend_lookback_change",
        "avg_carry_weight_change",
        "avg_train_sharpe",
        "avg_test_sharpe",
        "sharpe_degradation",
        "positive_test_sharpe_rate",
        "avg_test_max_drawdown",
        "worst_test_max_drawdown",
    } <= set(result.walk_forward_summary.columns)
    assert {
        "trend_lookback",
        "carry_weight",
        "windows",
        "selection_count",
        "selection_rate",
        "avg_train_sharpe",
        "avg_test_ann_return",
        "avg_test_sharpe",
        "sharpe_degradation",
        "positive_test_sharpe_rate",
        "avg_test_max_drawdown",
        "worst_test_max_drawdown",
    } <= set(result.parameter_grid_summary.columns)
    assert {
        "windows",
        "avg_selected_test_sharpe",
        "avg_best_available_test_sharpe",
        "avg_test_sharpe_regret",
        "max_test_sharpe_regret",
        "zero_regret_window_rate",
        "avg_selected_rank",
        "worst_selected_rank",
    } <= set(result.selection_quality.columns)
    assert result.parameter_grid["is_selected"].sum() == len(result.selections)
    assert result.parameter_grid_summary["selection_count"].sum() == len(result.selections)
    assert set(result.bootstrap_bands.index) == {"p05", "p50", "p95"}
    assert {
        "samples",
        "block_days",
        "observations",
        "prob_positive_ann_return",
        "prob_positive_sharpe",
        "prob_positive_sortino",
        "prob_drawdown_below_limit",
        "drawdown_limit",
        "median_sharpe",
        "median_max_drawdown",
        "median_cvar_5",
    } <= set(result.bootstrap_reliability.columns)
    assert np.isfinite(result.summary.to_numpy()).all()
    assert np.isfinite(result.return_distribution.to_numpy()).all()
    assert np.isfinite(result.rolling_metrics.to_numpy()).all()
    assert np.isfinite(
        result.selections[["train_sharpe", "test_ann_return", "test_sharpe", "test_max_drawdown"]]
        .to_numpy()
        .astype(float)
    ).all()
    assert np.isfinite(result.concentration.to_numpy()).all()
    assert np.isfinite(result.drawdowns[["depth", "length_days"]].to_numpy()).all()
    assert np.isfinite(result.execution_costs.to_numpy()).all()
    assert np.isfinite(result.execution_costs_by_contract.to_numpy()).all()
    assert np.isfinite(result.signal_contribution.to_numpy()).all()
    assert np.isfinite(result.walk_forward_summary.to_numpy()).all()
    assert np.isfinite(result.selection_quality.to_numpy()).all()
    assert np.isfinite(
        result.parameter_grid[
            ["train_sharpe", "test_ann_return", "test_sharpe", "test_max_drawdown"]
        ]
        .to_numpy()
        .astype(float)
    ).all()
    assert np.isfinite(result.parameter_grid_summary.to_numpy()).all()
    assert np.isfinite(result.bootstrap_reliability.to_numpy()).all()


def test_walk_forward_returns_reconcile_to_stitched_portfolio_weights() -> None:
    panel = make_sample_panel(days=780, contracts=8, seed=9)
    settings = BacktestSettings(
        trend_lookbacks=(63, 126),
        carry_weights=(0.0, 0.5),
        train_days=252,
        test_days=63,
        bootstrap_samples=20,
    )

    result = run_backtest(panel, settings)

    parameter_switches = (
        result.selections[["trend_lookback", "carry_weight"]]
        .diff()
        .ne(0.0)
        .any(axis=1)
        .iloc[1:]
    )
    assert parameter_switches.any()

    prior_date = panel.prices.index[settings.train_days - 1]
    weights = pd.concat(
        [
            pd.DataFrame(0.0, index=[prior_date], columns=result.weights.columns),
            result.weights,
        ]
    )
    expected = net_strategy_returns(
        weights,
        daily_returns(panel.prices).loc[weights.index],
        panel.roll_cost_bps.loc[weights.index],
        panel.commission_bps,
        slippage_bps=settings.slippage_bps,
    ).iloc[1:]

    assert np.allclose(result.returns, expected)
