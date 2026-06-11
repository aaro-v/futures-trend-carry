from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer

from futures_backtester.backtesting.engine import BacktestSettings, run_backtest
from futures_backtester.data import make_sample_panel, panel_to_frame

app = typer.Typer(help="Trend and carry global futures backtester.")


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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = make_sample_panel(days=days, contracts=contracts)
    settings = BacktestSettings(bootstrap_samples=bootstrap_samples, target_vol=target_vol)
    result = run_backtest(panel, settings)

    result.summary.to_csv(output_dir / "summary.csv", index=False)
    result.returns.to_frame().to_csv(output_dir / "daily_returns.csv")
    result.weights.to_csv(output_dir / "weights.csv")
    result.selections.to_csv(output_dir / "walk_forward_selections.csv", index=False)
    result.regime_metrics.to_csv(output_dir / "regime_metrics.csv")
    result.sector_exposure.to_csv(output_dir / "sector_exposure.csv")
    result.concentration.to_csv(output_dir / "concentration.csv", index=False)
    result.drawdowns.to_csv(output_dir / "drawdowns.csv", index=False)
    result.execution_costs.to_csv(output_dir / "execution_costs.csv", index=False)
    result.bootstrap_bands.to_csv(output_dir / "bootstrap_bands.csv")

    equity = (1.0 + result.returns).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    equity.plot(ax=ax, color="#1f5a5f", linewidth=2)
    ax.set_title("Synthetic Trend & Carry Futures Backtest")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=160)
    plt.close(fig)

    summary = result.summary.iloc[0]
    top_sector = result.sector_exposure.iloc[0]
    concentration = result.concentration.iloc[0]
    worst_drawdown = result.drawdowns.iloc[0] if not result.drawdowns.empty else None
    execution_costs = result.execution_costs.iloc[0]
    avg_test_sharpe = result.selections["test_sharpe"].mean()
    avg_test_drawdown = result.selections["test_max_drawdown"].mean()
    drawdown_line = (
        f"- Worst drawdown episode: {worst_drawdown['depth']:.2%} "
        f"over {int(worst_drawdown['length_days'])} trading days\n"
        if worst_drawdown is not None
        else "- Worst drawdown episode: none observed\n"
    )
    report = (
        "# Research Report\n\n"
        "Synthetic global futures trend and carry backtest.\n\n"
        f"- Annualized return: {summary['ann_return']:.2%}\n"
        f"- Annualized volatility: {summary['ann_vol']:.2%}\n"
        f"- Sharpe: {summary['sharpe']:.2f}\n"
        f"- Downside volatility: {summary['downside_vol']:.2%}\n"
        f"- Sortino: {summary['sortino']:.2f}\n"
        f"- Maximum drawdown: {summary['max_drawdown']:.2%}\n"
        f"- Calmar: {summary['calmar']:.2f}\n"
        f"- CVaR 5%: {summary['cvar_5']:.2%}\n"
        f"- Largest average sector gross weight: {top_sector.name} "
        f"({top_sector['avg_gross_weight']:.2f})\n"
        f"- Effective contract count: {concentration['effective_contract_count']:.1f}\n"
        f"- Average top-5 contract gross share: {concentration['avg_top_5_gross_share']:.2%}\n\n"
        "Walk-forward validation:\n\n"
        f"- Average selected-window test Sharpe: {avg_test_sharpe:.2f}\n"
        f"- Average selected-window test max drawdown: {avg_test_drawdown:.2%}\n\n"
        "Drawdown diagnostics:\n\n"
        f"{drawdown_line}\n"
        "Execution diagnostics:\n\n"
        f"- Annualized turnover: {execution_costs['annualized_turnover']:.1f}x\n"
        f"- Annualized roll cost: {execution_costs['ann_roll_cost']:.2%}\n"
        f"- Annualized trade cost: {execution_costs['ann_trade_cost']:.2%}\n"
        f"- Annualized total cost: {execution_costs['ann_total_cost']:.2%}\n\n"
        "Outputs are generated from deterministic synthetic data and are intended "
        "for engineering review, not live-market performance claims.\n"
    )
    (output_dir / "research_report.md").write_text(report, encoding="utf-8")
    typer.echo(f"Wrote artifacts to {output_dir}")
