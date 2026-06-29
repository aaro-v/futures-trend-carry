import numpy as np
import pandas as pd

from futures_backtester.analytics import (
    bootstrap_reliability_summary,
    concentration_summary,
    correlation_pair_summary,
    diversification_summary,
    drawdown_summary,
    execution_cost_by_contract,
    execution_cost_summary,
    exposure_change_summary,
    monthly_return_summary,
    parameter_grid_summary,
    path_stress_summary,
    performance_metrics,
    rebalance_summary,
    return_distribution_summary,
    rolling_metrics,
    selection_quality_summary,
    signal_contribution_summary,
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


def test_return_distribution_summary_reports_daily_and_monthly_shape() -> None:
    returns = pd.Series(
        [0.02, -0.01, 0.01, -0.03, 0.04],
        index=pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02", "2024-02-05"]
        ),
    )

    summary = return_distribution_summary(returns)

    assert list(summary.columns) == [
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
    ]
    assert summary.loc[0, "observations"] == 5.0
    assert summary.loc[0, "hit_rate"] == 0.6
    assert np.isclose(summary.loc[0, "avg_gain"], (0.02 + 0.01 + 0.04) / 3)
    assert np.isclose(summary.loc[0, "avg_loss"], (-0.01 - 0.03) / 2)
    assert np.isclose(summary.loc[0, "gain_loss_ratio"], ((0.02 + 0.01 + 0.04) / 3) / 0.02)
    assert summary.loc[0, "best_day"] == 0.04
    assert summary.loc[0, "worst_day"] == -0.03
    assert summary.loc[0, "positive_month_rate"] == 1.0
    assert summary.loc[0, "best_month"] > summary.loc[0, "worst_month"]
    assert np.isfinite(summary[["skew", "excess_kurtosis"]]).all(axis=None)


def test_return_distribution_summary_handles_empty_returns() -> None:
    summary = return_distribution_summary(pd.Series(dtype=float))

    assert summary.shape == (1, 12)
    assert (summary.iloc[0] == 0.0).all()


def test_monthly_return_summary_reports_path_metrics_by_month() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.03, 0.02],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"]),
    )

    summary = monthly_return_summary(returns)

    assert list(summary.columns) == [
        "month",
        "return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "positive_days",
        "observations",
    ]
    assert summary["month"].tolist() == ["2024-01", "2024-02"]
    assert np.isclose(summary.loc[0, "return"], (1.01 * 0.98) - 1.0)
    assert np.isclose(summary.loc[1, "return"], (1.03 * 1.02) - 1.0)
    assert summary.loc[0, "positive_days"] == 0.5
    assert summary.loc[1, "positive_days"] == 1.0
    assert summary.loc[0, "observations"] == 2.0
    assert np.isfinite(summary[["ann_vol", "sharpe", "max_drawdown"]]).all(axis=None)


def test_monthly_return_summary_requires_datetime_index() -> None:
    summary = monthly_return_summary(pd.Series([0.01, -0.02]))

    assert summary.empty
    assert list(summary.columns) == [
        "month",
        "return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "positive_days",
        "observations",
    ]


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


def test_path_stress_summary_ranks_worst_rolling_windows() -> None:
    returns = pd.Series(
        [0.01, -0.03, -0.02, 0.01, -0.04, 0.02],
        index=pd.date_range("2024-01-01", periods=6),
    )

    summary = path_stress_summary(returns, horizons=(3,), top_n=2)

    assert list(summary.columns) == [
        "horizon_days",
        "rank",
        "start",
        "end",
        "cumulative_return",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
    ]
    assert summary["horizon_days"].tolist() == [3.0, 3.0]
    assert summary["rank"].tolist() == [1.0, 2.0]
    assert summary.loc[0, "start"] == returns.index[2]
    assert summary.loc[0, "end"] == returns.index[4]
    expected_worst = (1.0 - 0.02) * (1.0 + 0.01) * (1.0 - 0.04) - 1.0
    assert np.isclose(summary.loc[0, "cumulative_return"], expected_worst)
    assert summary.loc[0, "cumulative_return"] <= summary.loc[1, "cumulative_return"]
    assert np.isfinite(summary[["ann_return", "ann_vol", "sharpe", "max_drawdown"]]).all(
        axis=None
    )


def test_path_stress_summary_handles_empty_or_short_returns() -> None:
    empty = path_stress_summary(pd.Series(dtype=float))
    short = path_stress_summary(pd.Series([0.01, -0.02]), horizons=(3,), top_n=2)

    assert empty.empty
    assert short.empty
    assert list(empty.columns) == list(short.columns)


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


def test_diversification_summary_measures_weighted_leg_vol_benefit() -> None:
    weights = pd.DataFrame(
        {
            "ES": [1.0, 1.0, 1.0, 1.0],
            "TY": [1.0, 1.0, 1.0, 1.0],
        }
    )
    returns = pd.DataFrame(
        {
            "ES": [0.01, -0.01, 0.01, -0.01],
            "TY": [-0.005, 0.005, -0.005, 0.005],
        }
    )

    summary = diversification_summary(weights, returns)

    assert list(summary.columns) == [
        "ann_portfolio_gross_vol",
        "ann_sum_component_vol",
        "diversification_ratio",
        "vol_reduction",
        "avg_pairwise_corr",
        "max_pairwise_corr",
        "contracts_with_risk",
    ]
    assert summary.loc[0, "ann_sum_component_vol"] > 0.0
    assert summary.loc[0, "ann_portfolio_gross_vol"] < summary.loc[0, "ann_sum_component_vol"]
    assert summary.loc[0, "diversification_ratio"] > 1.0
    assert summary.loc[0, "vol_reduction"] > 0.0
    assert summary.loc[0, "avg_pairwise_corr"] < 0.0
    assert summary.loc[0, "max_pairwise_corr"] < 0.0
    assert summary.loc[0, "contracts_with_risk"] == 2.0


def test_diversification_summary_handles_empty_and_zero_risk_inputs() -> None:
    empty = diversification_summary(pd.DataFrame(), pd.DataFrame())
    zero_risk = diversification_summary(
        pd.DataFrame({"ES": [0.0, 0.0], "TY": [0.0, 0.0]}),
        pd.DataFrame({"ES": [0.01, -0.01], "TY": [-0.01, 0.01]}),
    )

    assert empty.shape == (1, 7)
    assert zero_risk.shape == (1, 7)
    assert (empty.iloc[0] == 0.0).all()
    assert (zero_risk.iloc[0] == 0.0).all()


def test_correlation_pair_summary_ranks_weighted_leg_covariance_pairs() -> None:
    weights = pd.DataFrame(
        {
            "ES": [1.0, 1.0, 1.0, 1.0, 1.0],
            "NQ": [1.0, 1.0, 1.0, 1.0, 1.0],
            "TY": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    returns = pd.DataFrame(
        {
            "ES": [0.00, 0.01, -0.01, 0.02, -0.02],
            "NQ": [0.00, 0.01, -0.01, 0.02, -0.02],
            "TY": [0.00, -0.01, 0.01, -0.02, 0.02],
        }
    )

    summary = correlation_pair_summary(weights, returns, top_n=2)

    assert list(summary.columns) == [
        "contract_a",
        "contract_b",
        "correlation",
        "ann_covariance",
        "abs_ann_covariance",
        "avg_abs_weight_a",
        "avg_abs_weight_b",
        "combined_avg_abs_weight",
    ]
    assert len(summary) == 2
    assert summary.loc[0, "contract_a"] == "ES"
    assert summary.loc[0, "contract_b"] == "NQ"
    assert summary.loc[0, "correlation"] > 0.99
    assert summary.loc[0, "ann_covariance"] > 0.0
    assert summary.loc[0, "abs_ann_covariance"] >= summary.loc[1, "abs_ann_covariance"]
    assert summary.loc[0, "avg_abs_weight_a"] == 0.8
    assert summary.loc[0, "avg_abs_weight_b"] == 0.8
    assert summary.loc[0, "combined_avg_abs_weight"] == 1.6


def test_correlation_pair_summary_handles_empty_or_single_risk_inputs() -> None:
    empty = correlation_pair_summary(pd.DataFrame(), pd.DataFrame())
    single_risk = correlation_pair_summary(
        pd.DataFrame({"ES": [1.0, 1.0], "TY": [0.0, 0.0]}),
        pd.DataFrame({"ES": [0.01, -0.01], "TY": [-0.01, 0.01]}),
    )

    assert empty.empty
    assert single_risk.empty
    assert list(empty.columns) == list(single_risk.columns)


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


def test_rebalance_summary_measures_turnover_frequency_and_size() -> None:
    weights = pd.DataFrame(
        {
            "ES": [0.0, 0.5, 0.5, 0.25, 0.25],
            "TY": [0.0, -0.5, -0.5, -0.25, 0.0],
        }
    )

    summary = rebalance_summary(weights)

    assert list(summary.columns) == [
        "observations",
        "rebalance_days",
        "rebalance_day_rate",
        "avg_daily_turnover",
        "avg_rebalance_turnover",
        "median_rebalance_turnover",
        "p95_rebalance_turnover",
        "max_rebalance_turnover",
        "avg_days_between_rebalances",
        "max_contract_turnover_share",
    ]
    assert summary.loc[0, "observations"] == 5.0
    assert summary.loc[0, "rebalance_days"] == 3.0
    assert summary.loc[0, "rebalance_day_rate"] == 0.6
    assert np.isclose(summary.loc[0, "avg_daily_turnover"], 1.75 / 5)
    assert np.isclose(summary.loc[0, "avg_rebalance_turnover"], 1.75 / 3)
    assert summary.loc[0, "median_rebalance_turnover"] == 0.5
    assert summary.loc[0, "max_rebalance_turnover"] == 1.0
    assert summary.loc[0, "avg_days_between_rebalances"] == 1.5
    assert summary.loc[0, "max_contract_turnover_share"] == 1.0


def test_rebalance_summary_handles_empty_or_static_weights() -> None:
    empty = rebalance_summary(pd.DataFrame())
    static = rebalance_summary(pd.DataFrame({"ES": [0.0, 0.0], "TY": [0.0, 0.0]}))

    assert empty.shape == (1, 10)
    assert static.shape == (1, 10)
    assert (empty.iloc[0] == 0.0).all()
    assert static.loc[0, "observations"] == 2.0
    assert static.loc[0, "rebalance_days"] == 0.0
    assert static.loc[0, "max_contract_turnover_share"] == 0.0


def test_exposure_change_summary_measures_contract_churn_and_sign_flips() -> None:
    weights = pd.DataFrame(
        {
            "ES": [0.0, 0.5, -0.25, -0.25],
            "TY": [0.0, -0.5, -0.5, 0.0],
            "CL": [0.0, 0.0, 0.25, 0.25],
        }
    )

    summary = exposure_change_summary(weights)

    assert list(summary.columns) == [
        "observations",
        "avg_gross_exposure",
        "avg_net_exposure",
        "avg_abs_net_exposure",
        "avg_daily_gross_exposure_change",
        "max_daily_gross_exposure_change",
        "gross_exposure_change_vol",
        "avg_active_contracts",
        "avg_entries",
        "avg_exits",
        "avg_sign_flips",
        "max_sign_flips",
    ]
    assert summary.loc[0, "observations"] == 4.0
    assert summary.loc[0, "avg_gross_exposure"] == 0.625
    assert summary.loc[0, "avg_net_exposure"] == -0.125
    assert summary.loc[0, "avg_abs_net_exposure"] == 0.125
    assert summary.loc[0, "avg_daily_gross_exposure_change"] == 0.375
    assert summary.loc[0, "max_daily_gross_exposure_change"] == 1.0
    assert summary.loc[0, "avg_active_contracts"] == 1.75
    assert summary.loc[0, "avg_entries"] == 0.75
    assert summary.loc[0, "avg_exits"] == 0.25
    assert summary.loc[0, "avg_sign_flips"] == 0.25
    assert summary.loc[0, "max_sign_flips"] == 1.0
    assert summary.loc[0, "gross_exposure_change_vol"] > 0.0


def test_exposure_change_summary_handles_empty_or_static_weights() -> None:
    empty = exposure_change_summary(pd.DataFrame())
    static = exposure_change_summary(pd.DataFrame({"ES": [0.0, 0.0], "TY": [0.0, 0.0]}))

    assert empty.shape == (1, 12)
    assert static.shape == (1, 12)
    assert (empty.iloc[0] == 0.0).all()
    assert static.loc[0, "observations"] == 2.0
    assert static.loc[0, "avg_active_contracts"] == 0.0
    assert static.loc[0, "max_sign_flips"] == 0.0


def test_signal_contribution_summary_reconciles_gross_return_attribution() -> None:
    weights = pd.DataFrame(
        {
            "ES": [0.0, 0.6, 0.6],
            "TY": [0.0, -0.4, -0.4],
        }
    )
    returns = pd.DataFrame(
        {
            "ES": [0.0, 0.01, 0.02],
            "TY": [0.0, -0.01, -0.02],
        }
    )
    trend = pd.DataFrame(
        {
            "ES": [1.0, 2.0, 2.0],
            "TY": [-1.0, -1.0, -1.0],
        }
    )
    carry = pd.DataFrame(
        {
            "ES": [1.0, 1.0, 1.0],
            "TY": [-1.0, -1.0, -1.0],
        }
    )

    summary = signal_contribution_summary(weights, returns, trend, carry)

    row = summary.iloc[0]
    assert list(summary.columns) == [
        "ann_gross_return",
        "ann_trend_attributed_return",
        "ann_carry_attributed_return",
        "ann_unassigned_return",
        "avg_abs_trend_component_share",
        "avg_abs_carry_component_share",
        "trend_dominant_gross_share",
        "carry_dominant_gross_share",
        "trend_carry_agreement_rate",
    ]
    assert np.isclose(
        row["ann_gross_return"],
        row["ann_trend_attributed_return"]
        + row["ann_carry_attributed_return"]
        + row["ann_unassigned_return"],
    )
    assert row["ann_trend_attributed_return"] > row["ann_carry_attributed_return"]
    assert 0.0 < row["avg_abs_carry_component_share"] < row["avg_abs_trend_component_share"]
    assert row["trend_carry_agreement_rate"] == 1.0


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


def test_selection_quality_summary_measures_out_of_sample_regret() -> None:
    grid = pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-01-01"] * 3 + ["2024-04-01"] * 3),
            "end": pd.to_datetime(["2024-03-31"] * 3 + ["2024-06-30"] * 3),
            "trend_lookback": [63, 126, 252, 63, 126, 252],
            "carry_weight": [0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
            "is_selected": [True, False, False, False, True, False],
            "train_sharpe": [1.0, 0.8, 0.2, 0.6, 1.1, 0.4],
            "test_ann_return": [0.02, 0.10, -0.01, 0.03, 0.04, 0.00],
            "test_sharpe": [0.2, 0.8, -0.1, 0.1, 0.4, -0.2],
            "test_max_drawdown": [-0.04, -0.03, -0.08, -0.05, -0.02, -0.06],
        }
    )

    summary = selection_quality_summary(grid)

    assert list(summary.columns) == [
        "windows",
        "avg_selected_test_sharpe",
        "avg_best_available_test_sharpe",
        "avg_test_sharpe_regret",
        "max_test_sharpe_regret",
        "zero_regret_window_rate",
        "avg_selected_rank",
        "worst_selected_rank",
    ]
    assert summary.loc[0, "windows"] == 2.0
    assert np.isclose(summary.loc[0, "avg_selected_test_sharpe"], 0.3)
    assert np.isclose(summary.loc[0, "avg_best_available_test_sharpe"], 0.6)
    assert np.isclose(summary.loc[0, "avg_test_sharpe_regret"], 0.3)
    assert np.isclose(summary.loc[0, "max_test_sharpe_regret"], 0.6)
    assert summary.loc[0, "zero_regret_window_rate"] == 0.5
    assert summary.loc[0, "avg_selected_rank"] == 1.5
    assert summary.loc[0, "worst_selected_rank"] == 2.0


def test_selection_quality_summary_handles_empty_or_unselected_grid() -> None:
    empty = selection_quality_summary(pd.DataFrame())
    unselected = selection_quality_summary(
        pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01"]),
                "end": pd.to_datetime(["2024-03-31"]),
                "is_selected": [False],
                "test_sharpe": [0.5],
            }
        )
    )

    assert empty.shape == (1, 8)
    assert unselected.shape == (1, 8)
    assert (empty.iloc[0] == 0.0).all()
    assert (unselected.iloc[0] == 0.0).all()
