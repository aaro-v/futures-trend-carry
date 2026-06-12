from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "downside_vol": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "cvar_5": 0.0,
        }
    ann_return = float(clean.mean() * 252)
    ann_vol = float(clean.std(ddof=0) * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    downside = clean.clip(upper=0.0)
    downside_vol = float(np.sqrt(np.square(downside).mean()) * np.sqrt(252))
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0.0 else 0.0
    cvar = float(clean[clean <= clean.quantile(0.05)].mean())
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "downside_vol": downside_vol,
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "cvar_5": cvar,
    }


def drawdown_summary(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    columns = ["start", "trough", "recovery", "depth", "length_days", "recovery_days"]
    clean = returns.dropna()
    if clean.empty:
        return pd.DataFrame(columns=columns)

    equity = (1.0 + clean).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    underwater = drawdown < 0.0

    rows = []
    in_episode = False
    start = clean.index[0]
    trough = clean.index[0]
    depth = 0.0
    for date, is_underwater in underwater.items():
        if is_underwater and not in_episode:
            in_episode = True
            start = date
            trough = date
            depth = float(drawdown.loc[date])
        elif is_underwater:
            if drawdown.loc[date] < depth:
                trough = date
                depth = float(drawdown.loc[date])
        elif in_episode:
            rows.append(
                {
                    "start": start,
                    "trough": trough,
                    "recovery": date,
                    "depth": depth,
                    "length_days": len(clean.loc[start:date]),
                    "recovery_days": len(clean.loc[trough:date]),
                }
            )
            in_episode = False

    if in_episode:
        rows.append(
            {
                "start": start,
                "trough": trough,
                "recovery": pd.NaT,
                "depth": depth,
                "length_days": len(clean.loc[start:]),
                "recovery_days": pd.NA,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("depth")
        .head(top_n)
        .reset_index(drop=True)
    )


def regime_metrics(returns: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    market_proxy = prices.pct_change().mean(axis=1).reindex(returns.index).fillna(0.0)
    vol = market_proxy.rolling(63).std().bfill()
    regimes = pd.Series("normal", index=returns.index)
    regimes[market_proxy < market_proxy.quantile(0.25)] = "risk_off"
    regimes[market_proxy > market_proxy.quantile(0.75)] = "risk_on"
    regimes[vol > vol.quantile(0.75)] = regimes[vol > vol.quantile(0.75)] + "_high_vol"

    rows = []
    for regime, regime_returns in returns.groupby(regimes):
        row = performance_metrics(regime_returns)
        row["regime"] = regime
        row["days"] = float(len(regime_returns))
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime").sort_index()


def sector_exposure_summary(weights: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    sector_map = sectors.reindex(weights.columns).fillna("unknown")
    gross_by_sector = weights.abs().T.groupby(sector_map).sum().T
    net_by_sector = weights.T.groupby(sector_map).sum().T
    summary = pd.DataFrame(
        {
            "avg_gross_weight": gross_by_sector.mean(),
            "avg_net_weight": net_by_sector.mean(),
            "max_gross_weight": gross_by_sector.max(),
            "days_active": (gross_by_sector > 0.0).sum(),
        }
    )
    return summary.sort_values("avg_gross_weight", ascending=False).rename_axis("sector")


def concentration_summary(weights: pd.DataFrame) -> pd.DataFrame:
    gross = weights.abs()
    gross_total = gross.sum(axis=1).replace(0.0, np.nan)
    gross_share = gross.div(gross_total, axis=0).fillna(0.0)
    hhi = gross_share.pow(2).sum(axis=1)
    top_5_share = gross_share.apply(lambda row: row.nlargest(min(5, len(row))).sum(), axis=1)
    avg_hhi = float(hhi.mean())
    effective_contract_count = float(1.0 / avg_hhi) if avg_hhi > 0.0 else 0.0

    return pd.DataFrame(
        [
            {
                "avg_gross_weight": float(gross_total.fillna(0.0).mean()),
                "avg_max_contract_gross_share": float(gross_share.max(axis=1).mean()),
                "max_contract_gross_share": float(gross_share.max(axis=1).max()),
                "avg_top_5_gross_share": float(top_5_share.mean()),
                "avg_hhi": avg_hhi,
                "effective_contract_count": effective_contract_count,
            }
        ]
    )


def execution_cost_summary(
    weights: pd.DataFrame,
    roll_cost_bps: pd.DataFrame,
    commission_bps: pd.Series,
    slippage_bps: float,
) -> pd.DataFrame:
    aligned_roll_costs = roll_cost_bps.reindex(
        index=weights.index,
        columns=weights.columns,
    ).fillna(0.0)
    aligned_commissions = commission_bps.reindex(weights.columns).fillna(0.0)

    prior_weights = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs()
    turnover.iloc[0] = weights.iloc[0].abs()

    roll_cost = (prior_weights.abs() * aligned_roll_costs / 10_000.0).sum(axis=1)
    commission_cost = (turnover * aligned_commissions / 10_000.0).sum(axis=1)
    slippage_cost = turnover.sum(axis=1) * slippage_bps / 10_000.0
    trade_cost = commission_cost + slippage_cost
    total_cost = roll_cost + trade_cost

    return pd.DataFrame(
        [
            {
                "avg_daily_turnover": float(turnover.sum(axis=1).mean()),
                "annualized_turnover": float(turnover.sum(axis=1).mean() * 252),
                "ann_roll_cost": float(roll_cost.mean() * 252),
                "ann_commission_cost": float(commission_cost.mean() * 252),
                "ann_slippage_cost": float(slippage_cost.mean() * 252),
                "ann_trade_cost": float(trade_cost.mean() * 252),
                "ann_total_cost": float(total_cost.mean() * 252),
                "max_daily_cost": float(total_cost.max()),
            }
        ]
    )


def walk_forward_summary(selections: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "windows",
        "unique_parameter_sets",
        "most_common_trend_lookback",
        "most_common_carry_weight",
        "avg_train_sharpe",
        "avg_test_sharpe",
        "sharpe_degradation",
        "positive_test_sharpe_rate",
        "avg_test_max_drawdown",
        "worst_test_max_drawdown",
    ]
    if selections.empty:
        return pd.DataFrame([dict.fromkeys(columns, 0.0)], columns=columns)

    parameter_sets = selections[["trend_lookback", "carry_weight"]].drop_duplicates()
    most_common_params = (
        selections.groupby(["trend_lookback", "carry_weight"], sort=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "trend_lookback", "carry_weight"], ascending=[False, True, True])
        .iloc[0]
    )
    avg_train_sharpe = float(selections["train_sharpe"].mean())
    avg_test_sharpe = float(selections["test_sharpe"].mean())

    return pd.DataFrame(
        [
            {
                "windows": float(len(selections)),
                "unique_parameter_sets": float(len(parameter_sets)),
                "most_common_trend_lookback": float(most_common_params["trend_lookback"]),
                "most_common_carry_weight": float(most_common_params["carry_weight"]),
                "avg_train_sharpe": avg_train_sharpe,
                "avg_test_sharpe": avg_test_sharpe,
                "sharpe_degradation": avg_train_sharpe - avg_test_sharpe,
                "positive_test_sharpe_rate": float((selections["test_sharpe"] > 0.0).mean()),
                "avg_test_max_drawdown": float(selections["test_max_drawdown"].mean()),
                "worst_test_max_drawdown": float(selections["test_max_drawdown"].min()),
            }
        ],
        columns=columns,
    )


def bootstrap_bands(
    returns: pd.Series,
    samples: int = 500,
    block: int = 21,
    seed: int = 11,
) -> pd.DataFrame:
    clean = returns.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    if len(clean) < block:
        return pd.DataFrame()
    for _ in range(samples):
        chunks = []
        while sum(len(chunk) for chunk in chunks) < len(clean):
            start = rng.integers(0, len(clean) - block + 1)
            chunks.append(clean[start : start + block])
        sample = pd.Series(np.concatenate(chunks)[: len(clean)])
        rows.append(performance_metrics(sample))
    metrics = pd.DataFrame(rows)
    return metrics.quantile([0.05, 0.5, 0.95]).rename(index={0.05: "p05", 0.5: "p50", 0.95: "p95"})
