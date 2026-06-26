from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import typer

from futures_backtester.backtesting.engine import BacktestSettings, run_backtest
from futures_backtester.data import make_sample_panel, panel_to_frame

app = typer.Typer(help="Trend and carry global futures backtester.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_manifest(output_dir: Path, artifact_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in artifact_names:
        path = output_dir / name
        row = {
            "artifact": name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "rows": pd.NA,
            "columns": pd.NA,
        }
        if path.suffix == ".csv":
            table = pd.read_csv(path)
            row["rows"] = len(table)
            row["columns"] = len(table.columns)
        rows.append(row)
    return pd.DataFrame(rows)


def _run_config(
    *,
    days: int,
    contracts: int,
    seed: int,
    settings: BacktestSettings,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "days": days,
                "contracts": contracts,
                "seed": seed,
                "target_vol": settings.target_vol,
                "slippage_bps": settings.slippage_bps,
                "bootstrap_samples": settings.bootstrap_samples,
                "rolling_window": settings.rolling_window,
                "rolling_min_periods": settings.rolling_min_periods,
                "train_days": settings.train_days,
                "test_days": settings.test_days,
                "trend_lookbacks": "|".join(str(value) for value in settings.trend_lookbacks),
                "carry_weights": "|".join(f"{value:g}" for value in settings.carry_weights),
            }
        ]
    )


@app.command()
def sample_data(
    output: Annotated[
        Path,
        typer.Option(help="Output parquet path."),
    ] = Path("data/cache/sample_futures.parquet"),
    days: Annotated[int, typer.Option(min=260)] = 1_500,
    contracts: Annotated[int, typer.Option(min=4)] = 50,
    seed: Annotated[int, typer.Option()] = 7,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_to_frame(make_sample_panel(days=days, contracts=contracts, seed=seed)).to_parquet(output)
    typer.echo(f"Wrote {output}")


@app.command()
def run(
    output_dir: Annotated[
        Path,
        typer.Option(help="Output directory."),
    ] = Path("reports/generated"),
    days: Annotated[int, typer.Option(min=520)] = 1_500,
    contracts: Annotated[int, typer.Option(min=4)] = 50,
    bootstrap_samples: Annotated[int, typer.Option(min=20)] = 500,
    target_vol: Annotated[float, typer.Option(min=0.01, max=1.0)] = 0.12,
    seed: Annotated[int, typer.Option(help="Synthetic data random seed.")] = 7,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = make_sample_panel(days=days, contracts=contracts, seed=seed)
    settings = BacktestSettings(bootstrap_samples=bootstrap_samples, target_vol=target_vol)
    result = run_backtest(panel, settings)

    _run_config(days=days, contracts=contracts, seed=seed, settings=settings).to_csv(
        output_dir / "run_config.csv",
        index=False,
    )
    result.summary.to_csv(output_dir / "summary.csv", index=False)
    result.return_distribution.to_csv(output_dir / "return_distribution.csv", index=False)
    result.rolling_metrics.to_csv(output_dir / "rolling_metrics.csv")
    result.returns.to_frame().to_csv(output_dir / "daily_returns.csv")
    result.weights.to_csv(output_dir / "weights.csv")
    result.selections.to_csv(output_dir / "walk_forward_selections.csv", index=False)
    result.parameter_grid.to_csv(output_dir / "parameter_grid.csv", index=False)
    result.parameter_grid_summary.to_csv(output_dir / "parameter_grid_summary.csv", index=False)
    result.regime_metrics.to_csv(output_dir / "regime_metrics.csv")
    result.sector_exposure.to_csv(output_dir / "sector_exposure.csv")
    result.concentration.to_csv(output_dir / "concentration.csv", index=False)
    result.drawdowns.to_csv(output_dir / "drawdowns.csv", index=False)
    result.path_stress.to_csv(output_dir / "path_stress.csv", index=False)
    result.execution_costs.to_csv(output_dir / "execution_costs.csv", index=False)
    result.execution_costs_by_contract.to_csv(output_dir / "execution_costs_by_contract.csv")
    result.rebalance.to_csv(output_dir / "rebalance.csv", index=False)
    result.exposure_changes.to_csv(output_dir / "exposure_changes.csv", index=False)
    result.signal_contribution.to_csv(output_dir / "signal_contribution.csv", index=False)
    result.walk_forward_summary.to_csv(output_dir / "walk_forward_summary.csv", index=False)
    result.selection_quality.to_csv(output_dir / "selection_quality.csv", index=False)
    result.bootstrap_bands.to_csv(output_dir / "bootstrap_bands.csv")
    result.bootstrap_reliability.to_csv(output_dir / "bootstrap_reliability.csv", index=False)
    artifact_names = [
        "run_config.csv",
        "summary.csv",
        "return_distribution.csv",
        "rolling_metrics.csv",
        "daily_returns.csv",
        "weights.csv",
        "walk_forward_selections.csv",
        "parameter_grid.csv",
        "parameter_grid_summary.csv",
        "regime_metrics.csv",
        "sector_exposure.csv",
        "concentration.csv",
        "drawdowns.csv",
        "path_stress.csv",
        "execution_costs.csv",
        "execution_costs_by_contract.csv",
        "rebalance.csv",
        "exposure_changes.csv",
        "signal_contribution.csv",
        "walk_forward_summary.csv",
        "selection_quality.csv",
        "bootstrap_bands.csv",
        "bootstrap_reliability.csv",
    ]

    equity = (1.0 + result.returns).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    equity.plot(ax=ax, color="#1f5a5f", linewidth=2)
    ax.set_title("Synthetic Trend & Carry Futures Backtest")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=160)
    plt.close(fig)
    artifact_names.append("equity_curve.png")

    summary = result.summary.iloc[0]
    distribution = result.return_distribution.iloc[0]
    top_sector = result.sector_exposure.iloc[0]
    concentration = result.concentration.iloc[0]
    worst_drawdown = result.drawdowns.iloc[0] if not result.drawdowns.empty else None
    worst_path_stress = result.path_stress.iloc[0] if not result.path_stress.empty else None
    execution_costs = result.execution_costs.iloc[0]
    rebalance = result.rebalance.iloc[0]
    exposure_changes = result.exposure_changes.iloc[0]
    top_contract_cost = (
        result.execution_costs_by_contract.iloc[0]
        if not result.execution_costs_by_contract.empty
        else None
    )
    signal_contribution = result.signal_contribution.iloc[0]
    walk_forward = result.walk_forward_summary.iloc[0]
    selection_quality = result.selection_quality.iloc[0]
    best_grid = result.parameter_grid_summary.iloc[0]
    bootstrap_reliability = result.bootstrap_reliability.iloc[0]
    rolling = result.rolling_metrics.iloc[-1] if not result.rolling_metrics.empty else None
    drawdown_line = (
        f"- Worst drawdown episode: {worst_drawdown['depth']:.2%} "
        f"over {int(worst_drawdown['length_days'])} trading days\n"
        if worst_drawdown is not None
        else "- Worst drawdown episode: none observed\n"
    )
    rolling_line = (
        f"- Latest {int(rolling['window_days'])}-day rolling Sharpe "
        f"({int(rolling['observations'])} observations): {rolling['sharpe']:.2f} "
        f"with {rolling['max_drawdown']:.2%} max drawdown\n"
        if rolling is not None
        else "- Latest rolling Sharpe: unavailable for this sample length\n"
    )
    path_stress_line = (
        f"- Worst {int(worst_path_stress['horizon_days'])}-day path: "
        f"{worst_path_stress['cumulative_return']:.2%} "
        f"from {worst_path_stress['start']} to {worst_path_stress['end']} "
        f"(Sharpe {worst_path_stress['sharpe']:.2f})\n"
        if worst_path_stress is not None
        else "- Worst rolling path: unavailable for this sample length\n"
    )
    contract_cost_line = (
        f"- Highest annualized contract cost: {top_contract_cost.name} "
        f"({top_contract_cost['ann_total_cost']:.2%}, "
        f"{top_contract_cost['annualized_turnover']:.1f}x turnover)\n"
        if top_contract_cost is not None
        else "- Highest annualized contract cost: unavailable\n"
    )
    report = (
        "# Research Report\n\n"
        "Synthetic global futures trend and carry backtest.\n\n"
        f"- Run configuration: {days} business days, {contracts} contracts, "
        f"seed {seed}, {bootstrap_samples} bootstrap samples, "
        f"{target_vol:.0%} target volatility\n"
        f"- Annualized return: {summary['ann_return']:.2%}\n"
        f"- Annualized volatility: {summary['ann_vol']:.2%}\n"
        f"- Sharpe: {summary['sharpe']:.2f}\n"
        f"- Downside volatility: {summary['downside_vol']:.2%}\n"
        f"- Sortino: {summary['sortino']:.2f}\n"
        f"- Maximum drawdown: {summary['max_drawdown']:.2%}\n"
        f"- Calmar: {summary['calmar']:.2f}\n"
        f"- CVaR 5%: {summary['cvar_5']:.2%}\n"
        f"- Daily hit rate: {distribution['hit_rate']:.0%} "
        f"with {distribution['gain_loss_ratio']:.2f}x average gain/loss ratio\n"
        f"- Best/worst daily return: {distribution['best_day']:.2%} / "
        f"{distribution['worst_day']:.2%}\n"
        f"- Positive months: {distribution['positive_month_rate']:.0%} "
        f"(best {distribution['best_month']:.2%}, worst {distribution['worst_month']:.2%})\n"
        f"{rolling_line}"
        f"- Largest average sector gross weight: {top_sector.name} "
        f"({top_sector['avg_gross_weight']:.2f})\n"
        f"- Effective contract count: {concentration['effective_contract_count']:.1f}\n"
        f"- Average top-5 contract gross share: {concentration['avg_top_5_gross_share']:.2%}\n\n"
        "Walk-forward validation:\n\n"
        f"- Windows evaluated: {walk_forward['windows']:.0f}\n"
        f"- Unique selected parameter sets: {walk_forward['unique_parameter_sets']:.0f}\n"
        f"- Most common selection: {walk_forward['most_common_trend_lookback']:.0f}-day trend, "
        f"{walk_forward['most_common_carry_weight']:.2f} carry weight\n"
        f"- Parameter switch rate: {walk_forward['parameter_switch_rate']:.0%}\n"
        f"- Average trend lookback change: {walk_forward['avg_trend_lookback_change']:.1f} days\n"
        f"- Average carry weight change: {walk_forward['avg_carry_weight_change']:.2f}\n"
        f"- Average train Sharpe: {walk_forward['avg_train_sharpe']:.2f}\n"
        f"- Average test Sharpe: {walk_forward['avg_test_sharpe']:.2f}\n"
        f"- Train-to-test Sharpe degradation: {walk_forward['sharpe_degradation']:.2f}\n"
        f"- Positive test Sharpe windows: {walk_forward['positive_test_sharpe_rate']:.0%}\n"
        f"- Worst selected-window test drawdown: {walk_forward['worst_test_max_drawdown']:.2%}\n\n"
        "Selection quality diagnostics:\n\n"
        f"- Average selected candidate test Sharpe: "
        f"{selection_quality['avg_selected_test_sharpe']:.2f}\n"
        f"- Average best available candidate test Sharpe: "
        f"{selection_quality['avg_best_available_test_sharpe']:.2f}\n"
        f"- Average selected-candidate Sharpe regret: "
        f"{selection_quality['avg_test_sharpe_regret']:.2f}\n"
        f"- Worst selected-candidate Sharpe regret: "
        f"{selection_quality['max_test_sharpe_regret']:.2f}\n"
        f"- Windows where the selected candidate was best available: "
        f"{selection_quality['zero_regret_window_rate']:.0%}\n"
        f"- Average selected candidate out-of-sample rank: "
        f"{selection_quality['avg_selected_rank']:.1f}\n\n"
        "Parameter grid diagnostics:\n\n"
        f"- Candidate grid evaluations: {len(result.parameter_grid):.0f}\n"
        f"- Best average test Sharpe candidate: {best_grid['trend_lookback']:.0f}-day trend, "
        f"{best_grid['carry_weight']:.2f} carry weight "
        f"({best_grid['avg_test_sharpe']:.2f} average test Sharpe)\n"
        f"- Best candidate selection rate: {best_grid['selection_rate']:.0%}\n"
        f"- Best candidate positive test Sharpe windows: "
        f"{best_grid['positive_test_sharpe_rate']:.0%}\n"
        f"- Best candidate worst test drawdown: {best_grid['worst_test_max_drawdown']:.2%}\n\n"
        "Bootstrap reliability:\n\n"
        f"- Resamples: {bootstrap_reliability['samples']:.0f} "
        f"with {bootstrap_reliability['block_days']:.0f}-day blocks\n"
        f"- Positive annualized return probability: "
        f"{bootstrap_reliability['prob_positive_ann_return']:.0%}\n"
        f"- Positive Sharpe probability: {bootstrap_reliability['prob_positive_sharpe']:.0%}\n"
        f"- Drawdown worse than {bootstrap_reliability['drawdown_limit']:.0%} probability: "
        f"{bootstrap_reliability['prob_drawdown_below_limit']:.0%}\n"
        f"- Median resampled Sharpe: {bootstrap_reliability['median_sharpe']:.2f}\n"
        f"- Median resampled CVaR 5%: {bootstrap_reliability['median_cvar_5']:.2%}\n\n"
        "Drawdown diagnostics:\n\n"
        f"{drawdown_line}\n"
        "Path stress diagnostics:\n\n"
        f"{path_stress_line}\n"
        "Execution diagnostics:\n\n"
        f"- Annualized turnover: {execution_costs['annualized_turnover']:.1f}x\n"
        f"- Annualized roll cost: {execution_costs['ann_roll_cost']:.2%}\n"
        f"- Annualized trade cost: {execution_costs['ann_trade_cost']:.2%}\n"
        f"- Annualized total cost: {execution_costs['ann_total_cost']:.2%}\n\n"
        "Rebalance diagnostics:\n\n"
        f"- Rebalance days: {rebalance['rebalance_days']:.0f} "
        f"({rebalance['rebalance_day_rate']:.0%} of evaluated days)\n"
        f"- Average rebalance turnover: {rebalance['avg_rebalance_turnover']:.2f}\n"
        f"- 95th percentile rebalance turnover: {rebalance['p95_rebalance_turnover']:.2f}\n"
        f"- Average days between rebalances: "
        f"{rebalance['avg_days_between_rebalances']:.1f}\n"
        f"- Largest single-contract share of a rebalance: "
        f"{rebalance['max_contract_turnover_share']:.0%}\n\n"
        "Exposure change diagnostics:\n\n"
        f"- Average gross exposure: {exposure_changes['avg_gross_exposure']:.2f}\n"
        f"- Average active contracts: {exposure_changes['avg_active_contracts']:.1f}\n"
        f"- Average daily gross exposure change: "
        f"{exposure_changes['avg_daily_gross_exposure_change']:.2f}\n"
        f"- Maximum daily gross exposure change: "
        f"{exposure_changes['max_daily_gross_exposure_change']:.2f}\n"
        f"- Average daily entries/exits/sign flips: "
        f"{exposure_changes['avg_entries']:.1f} / "
        f"{exposure_changes['avg_exits']:.1f} / "
        f"{exposure_changes['avg_sign_flips']:.1f}\n\n"
        "Contract cost diagnostics:\n\n"
        f"{contract_cost_line}\n"
        "Signal contribution diagnostics:\n\n"
        f"- Annualized gross return before costs: "
        f"{signal_contribution['ann_gross_return']:.2%}\n"
        f"- Trend-attributed gross return: "
        f"{signal_contribution['ann_trend_attributed_return']:.2%}\n"
        f"- Carry-attributed gross return: "
        f"{signal_contribution['ann_carry_attributed_return']:.2%}\n"
        f"- Average absolute trend component share: "
        f"{signal_contribution['avg_abs_trend_component_share']:.0%}\n"
        f"- Average absolute carry component share: "
        f"{signal_contribution['avg_abs_carry_component_share']:.0%}\n"
        f"- Trend/carry signal agreement rate: "
        f"{signal_contribution['trend_carry_agreement_rate']:.0%}\n\n"
        "Outputs are generated from deterministic synthetic data and are intended "
        "for engineering review, not live-market performance claims.\n"
    )
    (output_dir / "research_report.md").write_text(report, encoding="utf-8")
    artifact_names.append("research_report.md")
    _artifact_manifest(output_dir, artifact_names).to_csv(
        output_dir / "output_manifest.csv",
        index=False,
    )
    typer.echo(f"Wrote artifacts to {output_dir}")
