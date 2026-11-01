from __future__ import annotations

import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from futures_backtester.backtesting.engine import BacktestSettings, run_backtest
from futures_backtester.backtesting.reference import reference_metrics
from futures_backtester.backtesting.reporting import input_hashes, write_report
from futures_backtester.data import (
    FuturesPanel,
    load_panel_frames,
    make_sample_panel,
    write_sample_data,
)

app = typer.Typer(help="Trend and carry research on futures.")


@app.command()
def build_data(
    out: Annotated[Path, typer.Option(help="Private output directory.")] = Path("data/real"),
    end: Annotated[str | None, typer.Option(help="Last observation date, inclusive.")] = None,
) -> None:
    """Download Yahoo, FRED and EIA histories (install the data extra)."""
    from futures_backtester.data.market_data.build import build

    build(out, end=end)


@app.command()
def sample_data(
    out: Path = Path("data/synthetic"),
    days: int = 6_217,
    contracts: int = 44,
    seed: int = 7,
) -> None:
    """Write the synthetic panel and companion files in the real-data format."""
    write_sample_data(out, days, contracts, seed)
    typer.echo(f"Wrote synthetic data to {out}")


@app.command()
def run(
    data_dir: Annotated[
        Path | None, typer.Option(help="Directory containing panel.parquet.")
    ] = None,
    output_dir: Path = Path("reports/synthetic"),
    start: str = "2003-01-01",
    end: str | None = None,
    report_start: str = "2005-01-01",
    days: int = 6_217,
    contracts: int = 44,
    seed: int = 7,
    bootstrap_samples: int = 500,
) -> None:
    """Run training-only parameter selection and save net performance for three sleeves."""
    if data_dir is None:
        panel = make_sample_panel(days=days, contracts=contracts, seed=seed)
        # Apply the same date restriction to generated and file-based panels.
        dates = panel.prices.loc[start:end].index
        panel = FuturesPanel(
            panel.prices.loc[dates],
            panel.carry.loc[dates],
            panel.roll_cost_bps.loc[dates],
            panel.commission_bps.loc[dates],
            panel.sectors,
        )
    else:
        panel = FuturesPanel(**load_panel_frames(data_dir, start=start, end=end))
    settings = BacktestSettings(bootstrap_samples=bootstrap_samples, report_start=report_start)
    result = run_backtest(panel=panel, settings=settings)
    config = {
        "data_source": "synthetic" if data_dir is None else "file_panel",
        "data_dir": data_dir.as_posix() if data_dir else None,
        "input_sha256": input_hashes(data_dir) if data_dir else {},
        "synthetic": {"days": days, "contracts": contracts, "seed": seed}
        if data_dir is None
        else None,
        "start": start,
        "end": end,
        "settings": asdict(settings),
        "panel_start": str(panel.prices.index[0].date()),
        "panel_end": str(panel.prices.index[-1].date()),
        "contracts": len(panel.prices.columns),
        "sectors": panel.sectors.nunique(),
        "carry_contracts": int(panel.carry.notna().any().sum()),
        "reference_strategies": reference_metrics(panel, result.returns.index),
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("numpy", "pandas", "pyarrow", "matplotlib", "typer")
        },
    }
    write_report(result, output_dir, config)
    typer.echo(
        result.summary[["strategy", "ann_return", "ann_vol", "sharpe", "max_drawdown"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    typer.echo(f"Wrote reports to {output_dir}")


if __name__ == "__main__":
    app()
