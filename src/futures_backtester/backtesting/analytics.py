from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    annual = float(clean.mean() * 252) if len(clean) else 0.0
    vol = float(clean.std(ddof=0) * np.sqrt(252)) if len(clean) else 0.0
    downside = float(np.sqrt(np.square(clean.clip(upper=0)).mean()) * np.sqrt(252))
    downside = downside if np.isfinite(downside) else 0.0
    equity = (1 + clean).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1
    depth = float(drawdown.min()) if len(clean) else 0.0
    cagr = float(equity.iloc[-1] ** (252 / len(clean)) - 1) if len(clean) else 0.0
    return {
        "ann_return": annual,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": annual / vol if vol > 0 else 0.0,
        "downside_vol": downside,
        "sortino": annual / downside if downside > 0 else 0.0,
        "max_drawdown": depth,
        "calmar": cagr / abs(depth) if depth < 0 else 0.0,
        "cvar_5": float(clean[clean <= clean.quantile(0.05)].mean()) if len(clean) else 0.0,
    }


def periodic_returns(returns: pd.Series) -> pd.DataFrame:
    rows = []
    for kind, frequency in (("month", "ME"), ("year", "YE")):
        for end, group in returns.resample(frequency):
            if len(group):
                rows.append(
                    {
                        "period": end,
                        "frequency": kind,
                        "return": float((1 + group).prod() - 1),
                        "observations": len(group),
                    }
                )
    return pd.DataFrame(rows)


def rolling_metrics(returns: pd.Series, window: int = 252, min_periods: int = 63) -> pd.DataFrame:
    rolling = returns.rolling(window, min_periods=min_periods)
    vol = rolling.std(ddof=0) * np.sqrt(252)
    annual = rolling.mean() * 252
    return pd.DataFrame({"rolling_vol": vol, "rolling_sharpe": annual / vol.replace(0, np.nan)})


def drawdown_summary(returns: pd.Series) -> pd.DataFrame:
    columns = ["start", "trough", "recovery", "depth", "sessions"]
    if returns.empty:
        return pd.DataFrame(columns=columns)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1) - 1
    rows, start, trough, depth, sessions = [], None, None, 0.0, 0
    for date, value in drawdown.items():
        if value < 0:
            if start is None:
                start, trough, depth, sessions = date, date, value, 0
            sessions += 1
            if value < depth:
                trough, depth = date, value
        elif start is not None:
            rows.append([start, trough, date, depth, sessions])
            start = None
    if start is not None:
        rows.append([start, trough, pd.NaT, depth, sessions])
    return pd.DataFrame(rows, columns=columns).sort_values("depth", ignore_index=True)


def sector_summary(
    weights: pd.DataFrame, returns: pd.DataFrame, sectors: pd.Series, live: pd.Index
) -> pd.DataFrame:
    """Realized sector covariance with the gross portfolio, divided by total variance.

    Contributions sum to one and can be negative for a diversifying sector.
    They need not equal the forecast standalone sector risk budgets.
    """
    held = weights.shift(1).fillna(0).loc[live]
    legs = (held * returns.loc[live]).T.groupby(sectors).sum().T
    portfolio = legs.sum(axis=1)
    variance = portfolio.var(ddof=0)
    rows = []
    for sector in legs:
        members = sectors.index[sectors == sector]
        contribution = float(
            np.mean((legs[sector] - legs[sector].mean()) * (portfolio - portfolio.mean()))
        )
        rows.append(
            {
                "sector": sector,
                "avg_gross_weight": held[members].abs().sum(axis=1).mean(),
                "avg_net_weight": held[members].sum(axis=1).mean(),
                "ann_gross_return": legs[sector].mean() * 252,
                "standalone_ann_vol": legs[sector].std(ddof=0) * np.sqrt(252),
                "risk_contribution": contribution / variance if variance > 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def cost_summary(components: pd.DataFrame) -> dict[str, float]:
    return {
        "ann_gross_return": components["gross_return"].mean() * 252,
        "ann_net_return": components["net_return"].mean() * 252,
        "ann_roll_cost": components["roll_cost"].mean() * 252,
        "ann_commission_cost": components["commission_cost"].mean() * 252,
        "ann_slippage_cost": components["slippage_cost"].mean() * 252,
        "ann_total_cost": components["total_cost"].mean() * 252,
        "annualized_turnover": components["turnover"].mean() * 252,
    }


def regime_metrics(returns: pd.Series, equity_returns: pd.Series) -> pd.DataFrame:
    # Regimes are descriptive, classified using information available the previous day.
    state = equity_returns.rolling(126, min_periods=63).sum().shift(1).reindex(returns.index)
    rows = []
    for name, mask in (("equity_up", state >= 0), ("equity_down", state < 0)):
        group = returns[mask]
        rows.append({"regime": name, "observations": len(group), **performance_metrics(group)})
    return pd.DataFrame(rows)


def bootstrap_bands(
    returns: pd.Series, samples: int = 500, block: int = 21, seed: int = 7
) -> pd.DataFrame:
    """Percentile 95% intervals from a circular moving-block bootstrap."""
    values = returns.dropna().to_numpy()
    if not len(values) or samples < 1 or block < 1:
        raise ValueError("Bootstrap requires returns and positive sample/block counts.")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        indices = ((starts[:, None] + np.arange(block)) % len(values)).ravel()[: len(values)]
        rows.append(performance_metrics(pd.Series(values[indices])))
    metrics = pd.DataFrame(rows)
    result = metrics.quantile([0.025, 0.5, 0.975]).T
    result.columns = ["lower_95", "median", "upper_95"]
    result.index.name = "metric"
    result["samples"], result["block_days"] = samples, block
    return result.reset_index()
