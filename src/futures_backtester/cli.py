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
    result.rolling_metrics.to_csv(output_dir / "rolling_metrics.csv")
    result.returns.to_frame().to_csv(output_dir / "daily_returns.csv")
    result.weights.to_csv(output_dir / "weights.csv")
    result.selections.to_csv(output_dir / "walk_forward_selections.csv", index=False)
    result.regime_metrics.to_csv(output_dir / "regime_metrics.csv")
    result.sector_exposure.to_csv(output_dir / "sector_exposure.csv")
    result.concentration.to_csv(output_dir / "concentration.csv", index=False)
    result.drawdowns.to_csv(output_dir / "drawdowns.csv", index=False)
    result.execution_costs.to_csv(output_dir / "execution_costs.csv", index=False)
    result.walk_forward_summary.to_csv(output_dir / "walk_forward_summary.csv", index=False)
    result.bootstrap_bands.to_csv(output_dir / "bootstrap_bands.csv")
    artifact_names = [
        "summary.csv",
        "rolling_metrics.csv",
        "daily_returns.csv",
        "weights.csv",
        "walk_forward_selections.csv",
        "regime_metrics.csv",
        "sector_exposure.csv",
        "concentration.csv",
        "drawdowns.csv",
        "execution_costs.csv",
        "walk_forward_summary.csv",
        "bootstrap_bands.csv",
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
    top_sector = result.sector_exposure.iloc[0]
    concentration = result.concentration.iloc[0]
    worst_drawdown = result.drawdowns.iloc[0] if not result.drawdowns.empty else None
    execution_costs = result.execution_costs.iloc[0]
    walk_forward = result.walk_forward_summary.iloc[0]
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
    artifact_names.append("research_report.md")
    _artifact_manifest(output_dir, artifact_names).to_csv(
        output_dir / "output_manifest.csv",
        index=False,
    )
    typer.echo(f"Wrote artifacts to {output_dir}")
