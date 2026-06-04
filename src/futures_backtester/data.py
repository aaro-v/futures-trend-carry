from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SECTORS = {
    "equity_index": ["ES", "NQ", "RTY", "VG", "NK", "HSI"],
    "rates": ["TY", "FV", "US", "RX", "G", "JGB"],
    "fx": ["AD", "BP", "CD", "EC", "JY", "SF", "NE", "MX"],
    "energy": ["CL", "HO", "RB", "NG", "BZ"],
    "metals": ["GC", "SI", "HG", "PL", "PA"],
    "grains": ["C", "W", "S", "BO", "SM", "KW"],
    "softs": ["CC", "KC", "CT", "SB", "OJ"],
    "livestock": ["LC", "FC", "LH"],
}


@dataclass(frozen=True)
class FuturesPanel:
    prices: pd.DataFrame
    carry: pd.DataFrame
    roll_cost_bps: pd.DataFrame
    commission_bps: pd.Series
    sectors: pd.Series


def _contract_names(count: int) -> tuple[list[str], pd.Series]:
    names: list[str] = []
    sector_map: dict[str, str] = {}
    for sector, roots in SECTORS.items():
        for root in roots:
            if len(names) == count:
                return names, pd.Series(sector_map, name="sector")
            names.append(root)
            sector_map[root] = sector
    while len(names) < count:
        root = f"F{len(names) + 1:02d}"
        names.append(root)
        sector_map[root] = "diversified"
    return names, pd.Series(sector_map, name="sector")


def make_sample_panel(days: int = 1_500, contracts: int = 50, seed: int = 7) -> FuturesPanel:
    """Create deterministic futures-like prices and carry with sector structure."""

    rng = np.random.default_rng(seed)
    names, sectors = _contract_names(contracts)
    dates = pd.bdate_range("2018-01-02", periods=days)

    sector_names = list(dict.fromkeys(sectors.to_list()))
    sector_shocks = rng.normal(0.0, 0.006, size=(days, len(sector_names)))
    global_shock = rng.normal(0.0, 0.004, size=(days, 1))

    returns = np.zeros((days, contracts))
    carry = np.zeros((days, contracts))
    vol_levels = rng.uniform(0.010, 0.024, size=contracts)
    drift = rng.normal(0.00002, 0.00012, size=contracts)

    for j, name in enumerate(names):
        sector_idx = sector_names.index(str(sectors.loc[name]))
        idio = rng.normal(0.0, vol_levels[j], size=days)
        trend_state = np.zeros(days)
        carry_state = np.zeros(days)
        for t in range(1, days):
            trend_state[t] = 0.985 * trend_state[t - 1] + rng.normal(0, 0.0018)
            carry_state[t] = 0.992 * carry_state[t - 1] + rng.normal(0, 0.0009)
        returns[:, j] = (
            drift[j]
            + global_shock[:, 0]
            + 0.45 * sector_shocks[:, sector_idx]
            + trend_state
            + 0.35 * carry_state
            + idio
        )
        carry[:, j] = 252 * carry_state + rng.normal(0, 0.015, size=days)

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    roll_days = np.zeros((days, contracts), dtype=float)
    offsets = rng.integers(0, 21, size=contracts)
    for j, offset in enumerate(offsets):
        roll_days[offset::21, j] = rng.uniform(0.2, 2.4)

    commission = pd.Series(
        rng.uniform(0.25, 1.25, size=contracts),
        index=names,
        name="commission_bps",
    )
    return FuturesPanel(
        prices=pd.DataFrame(prices, index=dates, columns=names),
        carry=pd.DataFrame(carry, index=dates, columns=names),
        roll_cost_bps=pd.DataFrame(roll_days, index=dates, columns=names),
        commission_bps=commission,
        sectors=sectors,
    )


def panel_to_frame(panel: FuturesPanel) -> pd.DataFrame:
    frames = []
    for field, data in {
        "price": panel.prices,
        "carry": panel.carry,
        "roll_cost_bps": panel.roll_cost_bps,
    }.items():
        stacked = data.stack().rename(field)
        frames.append(stacked)
    out = pd.concat(frames, axis=1).reset_index(names=["date", "contract"])
    out["sector"] = out["contract"].map(panel.sectors)
    out["commission_bps"] = out["contract"].map(panel.commission_bps)
    return out
