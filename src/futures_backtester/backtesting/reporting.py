from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import BacktestResult


def input_hashes(data_dir: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in (
            "panel.parquet",
            "contracts_resolved.csv",
            "roll_schedule.csv",
            "rates_daily.parquet",
        )
    }


def write_report(result: BacktestResult, output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary": result.summary,
        "daily": result.daily,
        "periodic_returns": result.periodic,
        "sectors": result.sectors,
        "walk_forward": result.parameter_grid,
        "bootstrap": result.bootstrap,
        "costs": result.costs,
        "regimes": result.regimes,
        "drawdowns": result.drawdowns,
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.10g")
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    for name, color in (("combined", "#145a6a"), ("trend", "#be783c"), ("carry", "#778674")):
        frame = result.daily.loc[result.daily["strategy"] == name]
        ax.plot(frame["date"], frame["equity"], label=name.capitalize(), color=color, lw=1.6)
    ax.set(title="Trend and carry futures · net of modeled costs", ylabel="Growth of 1 · log scale")
    ax.set_yscale("log")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output_dir / "equity_curve.png", dpi=180)
    plt.close(fig)
