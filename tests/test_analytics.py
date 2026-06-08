import numpy as np
import pandas as pd

from futures_backtester.analytics import (
    concentration_summary,
    execution_cost_summary,
    performance_metrics,
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
