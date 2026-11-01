"""A futures-like simulation with fixed, modest trend and carry premia."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .market_data.loader import load_panel_frames
from .panel import FuturesPanel

# Annual volatility, common-sector correlation, carry mean/std, commission and roll bps.
PROFILE = {
    "equity_index": (0.215, 0.80, 0.000, 0.017, 0.13, 1.17),
    "rates": (0.059, 0.80, 0.018, 0.016, 0.21, 1.72),
    "fx": (0.101, 0.42, 0.006, 0.014, 0.33, 1.23),
    "energy": (0.397, 0.55, -0.088, 0.384, 0.30, 1.98),
    "metals": (0.280, 0.53, 0.000, 0.000, 0.19, 2.19),
    "grains": (0.264, 0.39, 0.000, 0.000, 0.78, 6.13),
    "softs": (0.329, 0.12, 0.000, 0.000, 0.63, 5.95),
    "livestock": (0.156, 0.32, 0.000, 0.000, 0.34, 3.25),
}
RATE_VOLS = {"ZT": 0.015, "ZF": 0.039, "ZN": 0.059, "ZB": 0.106, "UB": 0.142}


def _ar1(rng: np.random.Generator, days: int, columns: int, rho: float) -> np.ndarray:
    state = rng.normal(size=(days, columns))
    for t in range(1, days):
        state[t] = rho * state[t - 1] + np.sqrt(1 - rho**2) * state[t]
    return state


def write_sample_data(
    out: str | Path, days: int = 6_217, contracts: int = 44, seed: int = 7
) -> None:
    """Write the real builder's four files, including its 16-column panel schema."""
    if days < 2 or not 1 <= contracts <= 44:
        raise ValueError("Use at least two days and between 1 and 44 contracts.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    specs = pd.read_csv(Path(__file__).parent / "market_data" / "contracts.csv").iloc[:contracts]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2003-01-02", periods=days).as_unit("us")
    sector_names = list(PROFILE)
    common = rng.normal(size=(days, 1))
    sector_noise = rng.normal(size=(days, len(PROFILE)))
    sector_trend = _ar1(rng, days, len(PROFILE), 0.997)
    carry_state = _ar1(rng, days, contracts, 0.992)
    frames, schedules, metadata = [], [], []
    for j, spec in enumerate(specs.itertuples(index=False)):
        vol, corr, mean, dispersion, commission, roll_bps = PROFILE[spec.sector]
        vol = RATE_VOLS.get(spec.root, vol)
        k = sector_names.index(spec.sector)
        loading = -0.20 if spec.sector == "rates" else np.sqrt(0.08)
        shock = (
            loading * common[:, 0]
            + np.sqrt(corr - loading**2) * sector_noise[:, k]
            + np.sqrt(1 - corr) * rng.normal(size=days)
        )
        c = mean + dispersion * (carry_state[:, j] + 0.5 * rng.normal())
        has_carry = dispersion > 0 and spec.root != "BZ"
        carry_score = np.clip(c / max(dispersion, 0.005), -2, 2) if has_carry else 0.0
        # Annual drift / annual volatility: 0.65 times the latent trend plus 0.20 carry.
        drift = vol * (0.65 * sector_trend[:, k] + 0.20 * carry_score) / 252
        r = np.expm1(vol / np.sqrt(252) * shock + drift - 0.5 * vol**2 / 252)
        r[0] = 0.0
        price = 100 * np.cumprod(1 + r)
        interval = max(1, round(252 / (12 if spec.sector == "energy" else len(spec.cycle))))
        is_roll = (np.arange(days) + j * 7) % interval == 0
        carry = c.copy() if has_carry else np.full(days, np.nan)
        if spec.sector == "energy":
            carry[dates > "2024-04-05"] = np.nan
        frame = pd.DataFrame(
            {
                "date": dates,
                "contract": spec.root,
                "price": price,
                "carry": carry,
                "roll_cost_bps": np.where(is_roll, roll_bps, 0.0),
                "sector": spec.sector,
                "commission_bps": commission,
                "return": r,
                "price_raw": price,
                "open": price / (1 + r),
                "high": np.maximum(price, price / (1 + r)) * (1 + np.abs(r) / 2),
                "low": np.minimum(price, price / (1 + r)) / (1 + np.abs(r) / 2),
                "volume": rng.integers(1_000, 100_000, days).astype(float),
                "is_roll_day": is_roll,
                "bad_tick": False,
                "after_data_gap": False,
            }
        )
        frames.append(frame)
        schedules.append(pd.DataFrame({"root": spec.root, "switch_day": dates[is_roll]}))
        metadata.append({"root": spec.root, "commission_bps_median_5y": commission})
    long = pd.concat(frames).sort_values(["date", "contract"], ignore_index=True)
    for col in ("contract", "sector"):
        long[col] = long[col].astype("category")
    long.to_parquet(out / "panel.parquet", index=False)
    specs.merge(pd.DataFrame(metadata), on="root").to_csv(
        out / "contracts_resolved.csv", index=False
    )
    pd.concat(schedules).to_csv(out / "roll_schedule.csv", index=False)
    pd.DataFrame({"us_bill_3m": np.full(days, 0.02, dtype="float32")}, index=dates).to_parquet(
        out / "rates_daily.parquet"
    )


def make_sample_panel(days: int = 6_217, contracts: int = 44, seed: int = 7) -> FuturesPanel:
    """Use the same serialization and loading path as a downloaded panel."""
    with TemporaryDirectory() as temp:
        write_sample_data(temp, days, contracts, seed)
        return FuturesPanel(**load_panel_frames(temp))


def panel_to_frame(panel: FuturesPanel) -> pd.DataFrame:
    """Return the seven common fields without discarding missing carry observations."""
    out = pd.concat(
        {
            "price": panel.prices.stack(future_stack=True),
            "carry": panel.carry.stack(future_stack=True),
            "roll_cost_bps": panel.roll_cost_bps.stack(future_stack=True),
        },
        axis=1,
    ).reset_index(names=["date", "contract"])
    out["sector"] = out["contract"].map(panel.sectors)
    if isinstance(panel.commission_bps, pd.DataFrame):
        costs = panel.commission_bps.stack(future_stack=True).rename("commission_bps")
        out = out.merge(
            costs.rename_axis(["date", "contract"]).reset_index(), on=["date", "contract"]
        )
    else:
        out["commission_bps"] = out["contract"].map(panel.commission_bps)
    return out.dropna(subset=["price"])
