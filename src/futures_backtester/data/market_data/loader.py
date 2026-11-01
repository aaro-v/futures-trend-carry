"""Read the long panel written by ``build`` into wide frames for the backtester.

frames = load_panel_frames("data/real", start="2004-01-01")
panel = FuturesPanel(**frames)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_panel_frames(
    data_dir: str | Path = "data/real",
    start: str | None = None,
    end: str | None = None,
    contracts: list[str] | None = None,
    sectors: list[str] | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Wide date x contract frames: prices, carry, roll_cost_bps, commission_bps, sectors."""

    data_dir = Path(data_dir)
    long = pd.read_parquet(data_dir / "panel.parquet")
    specs = pd.read_csv(data_dir / "contracts_resolved.csv").set_index("root")
    if contracts is not None:
        long = long[long["contract"].isin(contracts)]
    if sectors is not None:
        long = long[long["sector"].isin(sectors)]
    long = long.assign(contract=long["contract"].astype(str))
    if start is not None:
        long = long[long["date"] >= pd.Timestamp(start)]
    if end is not None:
        long = long[long["date"] <= pd.Timestamp(end)]

    def wide(column: str) -> pd.DataFrame:
        return long.pivot(index="date", columns="contract", values=column).sort_index()

    prices = wide("price")
    # Carry prices over exchange holidays inside each contract's life (zero return),
    # but leave the period before a contract starts trading empty.
    prices = prices.ffill(limit=5)
    names = list(prices.columns)
    return {
        "prices": prices,
        "carry": wide("carry").reindex(prices.index),
        "roll_cost_bps": wide("roll_cost_bps").reindex(prices.index).fillna(0.0),
        # Retain the last known commission for an exit after a stale-price gap.
        "commission_bps": wide("commission_bps").reindex(prices.index).ffill(),
        "sectors": specs.loc[names, "sector"].rename("sector"),
    }
