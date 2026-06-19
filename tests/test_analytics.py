import numpy as np
import pandas as pd

from futures_backtester.analytics import (
    bootstrap_reliability_summary,
    concentration_summary,
    drawdown_summary,
    execution_cost_by_contract,
    execution_cost_summary,
    parameter_grid_summary,
    performance_metrics,
    rolling_metrics,
    walk_forward_summary,
)


def test_performance_metrics_include_downside_risk_ratios() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.003])

    metrics = performance_metrics(returns)

    assert {
        "ann_return",
        "ann_vol",
        "sharpe",
        "downside_vol",
        "sortino",
        "max_drawdown",
        "calmar",
        "cvar_5",
    } == set(metrics)
    assert metrics["downside_vol"] > 0.0
    assert np.isfinite(metrics["sortino"])
    assert np.isfinite(metrics["calmar"])


def test_performance_metrics_keep_zero_ratios_without_drawdown_or_downside_vol() -> None:
    metrics = performance_metrics(pd.Series([0.001, 0.002, 0.0015]))

    assert metrics["downside_vol"] == 0.0
    assert metrics["sortino"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["calmar"] == 0.0


def test_performance_metrics_measure_constant_loss_downside_vol() -> None:
    metrics = performance_metrics(pd.Series([-0.01, -0.01, -0.01]))

    assert metrics["ann_vol"] == 0.0
    assert metrics["downside_vol"] > 0.0
    assert metrics["sortino"] < 0.0


def test_rolling_metrics_emit_time_varying_risk_diagnostics() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01, 0.02],
        index=pd.date_range("2024-01-01", periods=5),
    )

    summary = rolling_metrics(returns, window=3, min_periods=2)

    assert list(summary.columns) == [
        "window_days",
        "observations",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
    ]
    assert list(summary.index) == list(returns.index[1:])
    assert summary.loc[returns.index[1], "observations"] == 2.0
    assert summary.loc[returns.index[-1], "observations"] == 3.0
    assert summary.loc[returns.index[-1], "window_days"] == 3.0
    expected_last = performance_metrics(returns.iloc[-3:])
    assert summary.loc[returns.index[-1], "sharpe"] == expected_last["sharpe"]
    assert summary.loc[returns.index[-1], "max_drawdown"] == expected_last["max_drawdown"]


def test_drawdown_summary_ranks_recovered_episodes_by_depth() -> None:
    returns = pd.Series(
        [0.05, -0.10, 0.02, 0.10, -0.03, 0.04],
        index=pd.date_range("2024-01-01", periods=6),
    )

    summary = drawdown_summary(returns)

    assert list(summary.columns) == [
        "start",
        "trough",
        "recovery",
        "depth",
        "length_days",
        "recovery_days",
    ]
    assert len(summary) == 2
    assert summary.loc[0, "depth"] < summary.loc[1, "depth"]
    assert summary.loc[0, "length_days"] == 3
    assert summary.loc[0, "recovery_days"] == 3


def test_drawdown_summary_marks_unrecovered_episode() -> None:
    returns = pd.Series([0.02, -0.05, -0.02], index=pd.date_range("2024-01-01", periods=3))

    summary = drawdown_summary(returns)

    assert len(summary) == 1
    assert pd.isna(summary.loc[0, "recovery"])
    assert pd.isna(summary.loc[0, "recovery_days"])
    assert summary.loc[0, "length_days"] == 2


def test_concentration_summary_handles_zero_exposure() -> None:
    weights = pd.DataFrame(0.0, index=pd.RangeIndex(3), columns=["ES", "TY"])

    summary = concentration_summary(weights)

    assert np.isfinite(summary.to_numpy()).all()
    assert summary.loc[0, "avg_gross_weight"] == 0.0
    assert summary.loc[0, "effective_contract_count"] == 0.0


def test_concentration_summary_reports_effective_contract_count() -> None:
    weights = pd.DataFrame(
        {
            "ES": [0.5, 0.2],
            "TY": [-0.5, -0.2],
            "CL": [0.0, 0.6],
        }
    )

    summary = concentration_summary(weights)

    assert 2.0 < summary.loc[0, "effective_contract_count"] < 3.0
    assert summary.loc[0, "avg_top_5_gross_share"] == 1.0


def test_execution_cost_summary_separates_roll_and_trade_costs() -> None:
    weights = pd.DataFrame(
        {
            "ES": [0.0, 0.5, 0.25],
            "TY": [0.0, -0.5, -0.25],
        }
    )
    roll_cost_bps = pd.DataFrame(
        {
            "ES": [1.0, 1.0, 1.0],
            "TY": [2.0, 2.0, 2.0],
        }
    )
    commission_bps = pd.Series({"ES": 0.5, "TY": 0.5})

    summary = execution_cost_summary(weights, roll_cost_bps, commission_bps, slippage_bps=1.0)

    expected_columns = {
        "avg_daily_turnover",
        "annualized_turnover",
        "ann_roll_cost",
        "ann_commission_cost",
        "ann_slippage_cost",
        "ann_trade_cost",
        "ann_total_cost",
        "max_daily_cost",
    }
    assert expected_columns <= set(summary.columns)
    assert summary.loc[0, "annualized_turnover"] == 126.0
    assert summary.loc[0, "ann_roll_cost"] > 0.0
    assert summary.loc[0, "ann_trade_cost"] > summary.loc[0, "ann_commission_cost"]
    assert summary.loc[0, "ann_total_cost"] > summary.loc[0, "ann_trade_cost"]


def test_execution_cost_by_contract_reconciles_to_portfolio_costs() -> None:
    weights = pd.DataFrame(
        {
            "ES": [0.0, 0.5, 0.25],
            "TY": [0.0, -0.5, -0.25],
        }
    )
    roll_cost_bps = pd.DataFrame(
        {
            "ES": [1.0, 1.0, 1.0],
            "TY": [2.0, 2.0, 2.0],
        }
    )
    commission_bps = pd.Series({"ES": 0.5, "TY": 0.5})

    portfolio = execution_cost_summary(weights, roll_cost_bps, commission_bps, slippage_bps=1.0)
    by_contract = execution_cost_by_contract(
        weights,
        roll_cost_bps,
        commission_bps,
        slippage_bps=1.0,
    )

    assert list(by_contract.columns) == [
        "avg_abs_weight",
        "avg_daily_turnover",
        "annualized_turnover",
        "ann_roll_cost",
        "ann_commission_cost",
        "ann_slippage_cost",
        "ann_trade_cost",
        "ann_total_cost",
        "max_daily_cost",
    ]
    assert list(by_contract.index) == ["TY", "ES"]
    assert by_contract.loc["TY", "ann_total_cost"] > by_contract.loc["ES", "ann_total_cost"]
    assert np.isclose(
        by_contract["annualized_turnover"].sum(),
        portfolio.loc[0, "annualized_turnover"],
    )
    assert np.isclose(by_contract["ann_roll_cost"].sum(), portfolio.loc[0, "ann_roll_cost"])
    assert np.isclose(
        by_contract["ann_commission_cost"].sum(),
        portfolio.loc[0, "ann_commission_cost"],
    )
    assert np.isclose(
        by_contract["ann_slippage_cost"].sum(),
        portfolio.loc[0, "ann_slippage_cost"],
    )
    assert np.isclose(by_contract["ann_trade_cost"].sum(), portfolio.loc[0, "ann_trade_cost"])
    assert np.isclose(by_contract["ann_total_cost"].sum(), portfolio.loc[0, "ann_total_cost"])


def test_bootstrap_reliability_summary_reports_resample_probabilities() -> None:
    returns = pd.Series(
        [0.01, -0.005, 0.004, 0.002, -0.003, 0.006, -0.002, 0.005],
        index=pd.date_range("2024-01-01", periods=8),
    )

    summary = bootstrap_reliability_summary(
        returns,
        samples=20,
        block=3,
        seed=3,
        drawdown_limit=-0.02,
    )

    expected_columns = [
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
    ]
    assert list(summary.columns) == expected_columns
    assert summary.loc[0, "samples"] == 20.0
    assert summary.loc[0, "block_days"] == 3.0
    assert summary.loc[0, "observations"] == 8.0
    assert summary.loc[0, "drawdown_limit"] == -0.02
    for column in [
        "prob_positive_ann_return",
        "prob_positive_sharpe",
        "prob_positive_sortino",
        "prob_drawdown_below_limit",
    ]:
        assert 0.0 <= summary.loc[0, column] <= 1.0
    assert np.isfinite(summary[["median_sharpe", "median_max_drawdown", "median_cvar_5"]]).all(
        axis=None
    )


def test_walk_forward_summary_reports_selection_stability() -> None:
    selections = pd.DataFrame(
        {
            "trend_lookback": [63, 63, 126],
            "carry_weight": [0.5, 0.5, 0.0],
            "train_sharpe": [1.2, 0.8, 0.4],
            "test_sharpe": [0.6, -0.2, 0.3],
            "test_max_drawdown": [-0.03, -0.08, -0.05],
        }
    )

    summary = walk_forward_summary(selections)

    assert summary.loc[0, "windows"] == 3.0
    assert summary.loc[0, "unique_parameter_sets"] == 2.0
    assert summary.loc[0, "most_common_trend_lookback"] == 63.0
    assert summary.loc[0, "most_common_carry_weight"] == 0.5
    assert summary.loc[0, "parameter_switch_rate"] == 0.5
    assert summary.loc[0, "avg_trend_lookback_change"] == 31.5
    assert summary.loc[0, "avg_carry_weight_change"] == 0.25
    assert summary.loc[0, "avg_train_sharpe"] == selections["train_sharpe"].mean()
    assert summary.loc[0, "avg_test_sharpe"] == selections["test_sharpe"].mean()
    assert summary.loc[0, "sharpe_degradation"] == (
        selections["train_sharpe"].mean() - selections["test_sharpe"].mean()
    )
    assert summary.loc[0, "positive_test_sharpe_rate"] == 2 / 3
    assert summary.loc[0, "worst_test_max_drawdown"] == -0.08


def test_parameter_grid_summary_ranks_candidates_by_test_sharpe() -> None:
    grid = pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-01-01"] * 4),
            "end": pd.to_datetime(["2024-03-31"] * 4),
            "trend_lookback": [63, 63, 126, 126],
            "carry_weight": [0.0, 0.0, 0.5, 0.5],
            "is_selected": [True, True, False, False],
            "train_sharpe": [0.8, 0.6, 1.1, 1.0],
            "test_ann_return": [0.05, 0.04, 0.02, -0.01],
            "test_sharpe": [0.7, 0.5, 0.2, -0.1],
            "test_max_drawdown": [-0.04, -0.03, -0.08, -0.12],
        }
    )

    summary = parameter_grid_summary(grid)

    assert list(summary.columns) == [
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
    ]
    assert summary.loc[0, "trend_lookback"] == 63
    assert summary.loc[0, "carry_weight"] == 0.0
    assert summary.loc[0, "windows"] == 2
    assert summary.loc[0, "selection_count"] == 2
    assert summary.loc[0, "selection_rate"] == 1.0
    assert summary.loc[0, "avg_test_sharpe"] == 0.6
    assert np.isclose(summary.loc[0, "sharpe_degradation"], 0.1)
    assert summary.loc[1, "worst_test_max_drawdown"] == -0.12
