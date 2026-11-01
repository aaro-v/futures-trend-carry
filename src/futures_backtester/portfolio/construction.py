from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_covariance(
    returns: pd.DataFrame, window: int = 63, shrinkage: float = 0.10
) -> np.ndarray:
    """Trailing daily covariance, shrunk toward its diagonal; no future filling.

    Missing returns enter as zero only for covariance calculation. The separate
    full-history volatility eligibility rule keeps newly listed contracts out.
    """
    values = returns.fillna(0.0).to_numpy()
    n, m = values.shape
    cross = np.einsum("ti,tj->tij", values, values)
    cumulative = np.concatenate([np.zeros((1, m, m)), cross.cumsum(axis=0)])
    totals = np.vstack([np.zeros(m), values.cumsum(axis=0)])
    cov = np.zeros((n, m, m))
    if n >= window:
        sums = totals[window:] - totals[:-window]
        products = cumulative[window:] - cumulative[:-window]
        cov[window - 1 :] = (products - np.einsum("ti,tj->tij", sums, sums) / window) / (window - 1)
    diagonal = np.arange(m)
    cov *= 1 - shrinkage
    cov[:, diagonal, diagonal] /= 1 - shrinkage
    return cov


def inverse_volatility_weights(
    signal: pd.DataFrame, vol: pd.DataFrame, sectors: pd.Series, covariance: np.ndarray
) -> pd.DataFrame:
    """Inverse vol within sectors, equal forecast standalone risk across active sectors."""
    raw = (signal.clip(-1, 1) / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = raw.to_numpy().copy()
    labels = sectors.reindex(raw.columns)
    for sector in labels.unique():
        members = np.flatnonzero(labels.to_numpy() == sector)
        w = values[:, members]
        block = covariance[:, members][:, :, members]
        risk = np.sqrt(np.maximum(np.einsum("ti,tij,tj->t", w, block, w), 0))
        values[:, members] = np.divide(
            w, risk[:, None], out=np.zeros_like(w), where=risk[:, None] > 1e-12
        )
    gross = np.abs(values).sum(axis=1)
    values = np.divide(values, gross[:, None], out=np.zeros_like(values), where=gross[:, None] > 0)
    return pd.DataFrame(values, index=signal.index, columns=signal.columns)


def apply_vol_target(
    base_weights: pd.DataFrame,
    covariance: np.ndarray,
    target_vol: float = 0.12,
    max_leverage: float = 8.0,
) -> pd.DataFrame:
    """One covariance-based scale, capped at max_leverage times gross notional / NAV."""
    w = base_weights.to_numpy()
    forecast = np.sqrt(np.maximum(np.einsum("ti,tij,tj->t", w, covariance, w), 0) * 252)
    leverage = np.divide(target_vol, forecast, out=np.zeros_like(forecast), where=forecast > 1e-12)
    return base_weights.mul(np.minimum(leverage, max_leverage), axis=0)


def return_components(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    roll_cost_bps: pd.DataFrame,
    commission_bps: pd.Series | pd.DataFrame,
    slippage_bps: float = 0.75,
) -> pd.DataFrame:
    """Targets at close t earn t+1 returns; their execution cost is booked on t+1.

    The opening trade is from zero. A roll is charged on the position held on
    that day, including a position closed at that day's close.
    """
    held = weights.shift(1).fillna(0.0)
    turnover = held.diff().fillna(held).abs()
    execution_commission = (
        commission_bps.shift(1) if isinstance(commission_bps, pd.DataFrame) else commission_bps
    )
    out = pd.DataFrame(
        {
            "gross_return": (held * returns).sum(axis=1),
            "roll_cost": (held.abs() * roll_cost_bps / 10_000).sum(axis=1),
            "commission_cost": (turnover * execution_commission / 10_000).sum(axis=1),
            "slippage_cost": turnover.sum(axis=1) * slippage_bps / 10_000,
            "turnover": turnover.sum(axis=1),
        }
    )
    out["total_cost"] = out[["roll_cost", "commission_cost", "slippage_cost"]].sum(axis=1)
    out["net_return"] = out["gross_return"] - out["total_cost"]
    return out


def net_strategy_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    roll_cost_bps: pd.DataFrame,
    commission_bps: pd.Series | pd.DataFrame,
    slippage_bps: float = 0.75,
) -> pd.Series:
    return return_components(weights, returns, roll_cost_bps, commission_bps, slippage_bps)[
        "net_return"
    ]
