"""Execution, cost and allocation checks without reselecting the strategy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from futures_backtester.data import FuturesPanel, load_panel_frames
from futures_backtester.portfolio import (
    apply_vol_target,
    inverse_volatility_weights,
    return_components,
    rolling_covariance,
)
from futures_backtester.signals import blended_signal, daily_returns, realized_vol

from .analytics import cost_summary, performance_metrics, sector_summary
from .engine import BacktestResult, BacktestSettings, run_backtest
from .reporting import input_hashes


def scenario_components(
    panel: FuturesPanel,
    weights: pd.DataFrame,
    settings: BacktestSettings,
    execution_delay: int = 0,
    trade_cost_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delay close targets by sessions; scale trading fees, preserving roll costs."""
    executed = weights.shift(execution_delay).fillna(0.0)
    components = return_components(
        executed,
        daily_returns(panel.prices),
        panel.roll_cost_bps,
        panel.commission_bps * trade_cost_multiplier,
        settings.slippage_bps * trade_cost_multiplier,
    )
    return executed, components


def robustness_tables(
    panel: FuturesPanel, result: BacktestResult, settings: BacktestSettings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = daily_returns(panel.prices)
    covariance = rolling_covariance(
        returns, settings.covariance_window, settings.covariance_shrinkage
    )
    signal = blended_signal(panel.prices, panel.carry, trend_lookback=252, carry_weight=0.35)
    base = inverse_volatility_weights(
        signal, realized_vol(returns, settings.covariance_window), panel.sectors, covariance
    )
    fixed = apply_vol_target(base, covariance, settings.target_vol, settings.max_leverage)
    live = result.returns.index
    first_target = int(panel.prices.index.get_indexer([live[0]])[0]) - 1
    fixed.iloc[:first_target] = 0.0
    scenarios = (
        ("baseline", result.weights, 0, 1.0),
        ("next_session_close", result.weights, 1, 1.0),
        ("trading_costs_2x", result.weights, 0, 2.0),
        ("trading_costs_3x", result.weights, 0, 3.0),
        ("fixed_blend_252_0.35", fixed, 0, 1.0),
    )
    rows, daily, sectors = [], [], []
    for name, weights, delay, multiplier in scenarios:
        executed, frame = scenario_components(panel, weights, settings, delay, multiplier)
        frame = frame.loc[live].copy()
        # Held gross notional / NAV on each P&L date, not unexecuted close targets.
        gross = executed.shift(1).fillna(0).loc[live].abs().sum(axis=1)
        frame["gross_leverage"] = gross
        rows.append(
            {
                "scenario": name,
                "start": live[0],
                "end": live[-1],
                "observations": len(live),
                "execution_delay_sessions": delay,
                "trade_cost_multiplier": multiplier,
                **performance_metrics(frame.net_return),
                **cost_summary(frame),
                "gross_leverage_mean": gross.mean(),
                "gross_leverage_p95": gross.quantile(0.95),
                "gross_leverage_max": gross.max(),
            }
        )
        daily.append(frame.assign(scenario=name).rename_axis("date").reset_index())
        sectors.append(sector_summary(executed, returns, panel.sectors, live).assign(scenario=name))
    return (
        pd.DataFrame(rows),
        pd.concat(daily, ignore_index=True),
        pd.concat(sectors, ignore_index=True),
    )


def write_robustness(report_dir: Path, data_dir: Path | None = None) -> pd.DataFrame:
    """Recreate the saved baseline and require unchanged inputs before writing checks."""
    config_path = report_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = data_dir or Path(config["data_dir"])
    if input_hashes(data_dir) != config["input_sha256"]:
        raise ValueError("Input hashes differ from the saved baseline; robustness was not written.")
    kwargs = config["settings"].copy()
    for name in ("trend_lookbacks", "carry_weights"):
        kwargs[name] = tuple(kwargs[name])
    settings = BacktestSettings(**kwargs)
    panel = FuturesPanel(**load_panel_frames(data_dir, start=config["start"], end=config["end"]))
    result = run_backtest(panel, settings)
    saved = pd.read_csv(report_dir / "summary.csv").set_index("strategy")
    actual = result.summary.set_index("strategy")
    numeric = saved.select_dtypes("number").columns
    np.testing.assert_allclose(
        actual.loc[saved.index, numeric], saved[numeric], rtol=1e-8, atol=1e-10
    )
    summary, daily, sectors = robustness_tables(panel, result, settings)
    for name, frame in (
        ("robustness", summary),
        ("robustness_daily", daily),
        ("robustness_sectors", sectors),
    ):
        frame.to_csv(report_dir / f"{name}.csv", index=False, float_format="%.10g")
    energy_dates = panel.carry.loc[:, panel.sectors == "energy"].dropna(how="all").index
    provenance = {
        "baseline_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "input_sha256": config["input_sha256"],
        "settings": asdict(settings),
        "selection": "Baseline decisions and targets frozen for execution and cost scenarios.",
        "execution": "Delay 1: signal at close t, trade at close t+1, earn return t+2.",
        "costs": "Multiply commission and slippage only; roll_cost_bps remains unchanged.",
        "fixed_blend": {"trend_lookback": 252, "carry_weight": 0.35},
        "gross_leverage": "Sum of absolute held notional weights on each P&L date.",
        "sector_risk": "Sector covariance with gross P&L divided by portfolio variance.",
        "carry_coverage": {
            "contracts_with_history": int(panel.carry.notna().any().sum()),
            "universe_contracts": len(panel.prices.columns),
            "energy_last_available_date": str(energy_dates.max().date())
            if len(energy_dates)
            else None,
        },
    }
    (report_dir / "robustness_config.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return summary
