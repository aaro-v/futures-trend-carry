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


def return_distribution_summary(returns: pd.Series) -> pd.DataFrame:
    columns = [
        "observations",
        "hit_rate",
        "avg_gain",
        "avg_loss",
        "gain_loss_ratio",
        "best_day",
        "worst_day",
        "skew",
        "excess_kurtosis",
        "positive_month_rate",
        "best_month",
        "worst_month",
    ]
    clean = returns.dropna()
    if clean.empty:
        return pd.DataFrame([dict.fromkeys(columns, 0.0)], columns=columns)

    gains = clean[clean > 0.0]
    losses = clean[clean < 0.0]
    avg_gain = float(gains.mean()) if not gains.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    gain_loss_ratio = avg_gain / abs(avg_loss) if avg_loss < 0.0 else 0.0
    skew = float(clean.skew()) if len(clean) >= 3 else 0.0
    excess_kurtosis = float(clean.kurt()) if len(clean) >= 4 else 0.0

    monthly_returns = pd.Series(dtype=float)
    if isinstance(clean.index, pd.DatetimeIndex):
        monthly_returns = clean.resample("ME").apply(lambda values: (1.0 + values).prod() - 1.0)

    return pd.DataFrame(
        [
            {
                "observations": float(len(clean)),
                "hit_rate": float((clean > 0.0).mean()),
                "avg_gain": avg_gain,
                "avg_loss": avg_loss,
                "gain_loss_ratio": float(gain_loss_ratio),
                "best_day": float(clean.max()),
                "worst_day": float(clean.min()),
                "skew": skew,
                "excess_kurtosis": excess_kurtosis,
                "positive_month_rate": (
                    float((monthly_returns > 0.0).mean()) if not monthly_returns.empty else 0.0
                ),
                "best_month": float(monthly_returns.max()) if not monthly_returns.empty else 0.0,
                "worst_month": float(monthly_returns.min()) if not monthly_returns.empty else 0.0,
            }
        ],
        columns=columns,
    )


def monthly_return_summary(returns: pd.Series) -> pd.DataFrame:
    columns = [
        "month",
        "return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "positive_days",
        "observations",
    ]
    clean = returns.dropna()
    if clean.empty or not isinstance(clean.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=columns)

    rows = []
    for month_end, monthly_returns in clean.resample("ME"):
        if monthly_returns.empty:
            continue
        metrics = performance_metrics(monthly_returns)
        rows.append(
            {
                "month": month_end.strftime("%Y-%m"),
                "return": float((1.0 + monthly_returns).prod() - 1.0),
                "ann_vol": metrics["ann_vol"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "positive_days": float((monthly_returns > 0.0).mean()),
                "observations": float(len(monthly_returns)),
            }
        )

    return pd.DataFrame(rows, columns=columns)


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


def path_stress_summary(
    returns: pd.Series,
    horizons: tuple[int, ...] = (21, 63, 126),
    top_n: int = 3,
) -> pd.DataFrame:
    columns = [
        "horizon_days",
        "rank",
        "start",
        "end",
        "cumulative_return",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
    ]
    clean = returns.dropna()
    if clean.empty or top_n <= 0:
        return pd.DataFrame(columns=columns)

    rows = []
    for horizon in horizons:
        if horizon <= 0 or len(clean) < horizon:
            continue
        window_growth = (1.0 + clean).rolling(horizon).apply(np.prod, raw=True) - 1.0
        worst_windows = window_growth.dropna().nsmallest(top_n, keep="first")
        for rank, (end_date, cumulative_return) in enumerate(worst_windows.items(), start=1):
            end_position = clean.index.get_loc(end_date)
            window_returns = clean.iloc[end_position - horizon + 1 : end_position + 1]
            metrics = performance_metrics(window_returns)
            rows.append(
                {
                    "horizon_days": float(horizon),
                    "rank": float(rank),
                    "start": window_returns.index[0],
                    "end": end_date,
                    "cumulative_return": float(cumulative_return),
                    "ann_return": metrics["ann_return"],
                    "ann_vol": metrics["ann_vol"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )

    return pd.DataFrame(rows, columns=columns)


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


def rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    min_periods: int = 63,
) -> pd.DataFrame:
    columns = [
        "window_days",
        "observations",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
    ]
    clean = returns.dropna()
    if clean.empty or min_periods <= 0:
        return pd.DataFrame(columns=columns).rename_axis("end_date")

    rows = []
    lookback = max(window, min_periods)
    for end_position in range(min_periods, len(clean) + 1):
        window_returns = clean.iloc[max(0, end_position - lookback) : end_position]
        metrics = performance_metrics(window_returns)
        rows.append(
            {
                "end_date": clean.index[end_position - 1],
                "window_days": float(window),
                "observations": float(len(window_returns)),
                "ann_return": metrics["ann_return"],
                "ann_vol": metrics["ann_vol"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns).rename_axis("end_date")
    return pd.DataFrame(rows).set_index("end_date")[columns]


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


def diversification_summary(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ann_portfolio_gross_vol",
        "ann_sum_component_vol",
        "diversification_ratio",
        "vol_reduction",
        "avg_pairwise_corr",
        "max_pairwise_corr",
        "contracts_with_risk",
    ]
    if weights.empty:
        return pd.DataFrame([dict.fromkeys(columns, 0.0)], columns=columns)

    aligned_returns = returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    prior_weights = weights.shift(1).fillna(0.0)
    weighted_leg_returns = prior_weights * aligned_returns
    portfolio_gross_return = weighted_leg_returns.sum(axis=1)

    portfolio_vol = float(portfolio_gross_return.std(ddof=0) * np.sqrt(252))
    component_vol = weighted_leg_returns.std(ddof=0) * np.sqrt(252)
    active_component_vol = component_vol[component_vol > 0.0]
    sum_component_vol = float(active_component_vol.sum())
    diversification_ratio = sum_component_vol / portfolio_vol if portfolio_vol > 0.0 else 0.0
    vol_reduction = 1.0 - portfolio_vol / sum_component_vol if sum_component_vol > 0.0 else 0.0

    avg_pairwise_corr = 0.0
    max_pairwise_corr = 0.0
    if len(active_component_vol) >= 2:
        corr = weighted_leg_returns[active_component_vol.index].corr()
        upper_triangle = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairwise_corr = corr.where(upper_triangle).stack()
        if not pairwise_corr.empty:
            avg_pairwise_corr = float(pairwise_corr.mean())
            max_pairwise_corr = float(pairwise_corr.max())

    return pd.DataFrame(
        [
            {
                "ann_portfolio_gross_vol": portfolio_vol,
                "ann_sum_component_vol": sum_component_vol,
                "diversification_ratio": float(diversification_ratio),
                "vol_reduction": float(vol_reduction),
                "avg_pairwise_corr": avg_pairwise_corr,
                "max_pairwise_corr": max_pairwise_corr,
                "contracts_with_risk": float(len(active_component_vol)),
            }
        ],
        columns=columns,
    )


def correlation_pair_summary(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    columns = [
        "contract_a",
        "contract_b",
        "correlation",
        "ann_covariance",
        "abs_ann_covariance",
        "avg_abs_weight_a",
        "avg_abs_weight_b",
        "combined_avg_abs_weight",
    ]
    if weights.empty or top_n <= 0:
        return pd.DataFrame(columns=columns)

    aligned_returns = returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    prior_weights = weights.shift(1).fillna(0.0)
    weighted_leg_returns = prior_weights * aligned_returns
    component_vol = weighted_leg_returns.std(ddof=0)
    active_contracts = component_vol[component_vol > 0.0].index
    if len(active_contracts) < 2:
        return pd.DataFrame(columns=columns)

    active_returns = weighted_leg_returns[active_contracts]
    corr = active_returns.corr()
    cov = active_returns.cov(ddof=0) * 252
    avg_abs_weight = prior_weights[active_contracts].abs().mean()

    rows = []
    contracts = list(active_contracts)
    for left_position, contract_a in enumerate(contracts[:-1]):
        for contract_b in contracts[left_position + 1 :]:
            ann_covariance = float(cov.loc[contract_a, contract_b])
            rows.append(
                {
                    "contract_a": contract_a,
                    "contract_b": contract_b,
                    "correlation": float(corr.loc[contract_a, contract_b]),
                    "ann_covariance": ann_covariance,
                    "abs_ann_covariance": abs(ann_covariance),
                    "avg_abs_weight_a": float(avg_abs_weight.loc[contract_a]),
                    "avg_abs_weight_b": float(avg_abs_weight.loc[contract_b]),
                    "combined_avg_abs_weight": float(
                        avg_abs_weight.loc[contract_a] + avg_abs_weight.loc[contract_b]
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["abs_ann_covariance", "correlation", "combined_avg_abs_weight"],
            ascending=[False, False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
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


def execution_cost_by_contract(
    weights: pd.DataFrame,
    roll_cost_bps: pd.DataFrame,
    commission_bps: pd.Series,
    slippage_bps: float,
) -> pd.DataFrame:
    columns = [
        "avg_abs_weight",
        "avg_daily_turnover",
        "annualized_turnover",
        "ann_roll_cost",
        "ann_commission_cost",
        "ann_slippage_cost",
        "ann_trade_cost",
        "ann_total_cost",
        "max_daily_cost",
    ]
    if weights.empty:
        return pd.DataFrame(columns=columns).rename_axis("contract")

    aligned_roll_costs = roll_cost_bps.reindex(
        index=weights.index,
        columns=weights.columns,
    ).fillna(0.0)
    aligned_commissions = commission_bps.reindex(weights.columns).fillna(0.0)

    prior_weights = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs()
    turnover.iloc[0] = weights.iloc[0].abs()

    roll_cost = prior_weights.abs() * aligned_roll_costs / 10_000.0
    commission_cost = turnover.mul(aligned_commissions, axis=1) / 10_000.0
    slippage_cost = turnover * slippage_bps / 10_000.0
    trade_cost = commission_cost + slippage_cost
    total_cost = roll_cost + trade_cost

    summary = pd.DataFrame(
        {
            "avg_abs_weight": weights.abs().mean(),
            "avg_daily_turnover": turnover.mean(),
            "annualized_turnover": turnover.mean() * 252,
            "ann_roll_cost": roll_cost.mean() * 252,
            "ann_commission_cost": commission_cost.mean() * 252,
            "ann_slippage_cost": slippage_cost.mean() * 252,
            "ann_trade_cost": trade_cost.mean() * 252,
            "ann_total_cost": total_cost.mean() * 252,
            "max_daily_cost": total_cost.max(),
        }
    )
    return summary[columns].sort_values("ann_total_cost", ascending=False).rename_axis("contract")


def rebalance_summary(weights: pd.DataFrame, threshold: float = 1e-12) -> pd.DataFrame:
    columns = [
        "observations",
        "rebalance_days",
        "rebalance_day_rate",
        "avg_daily_turnover",
        "avg_rebalance_turnover",
        "median_rebalance_turnover",
        "p95_rebalance_turnover",
        "max_rebalance_turnover",
        "avg_days_between_rebalances",
        "max_contract_turnover_share",
    ]
    if weights.empty:
        return pd.DataFrame([dict.fromkeys(columns, 0.0)], columns=columns)

    turnover = weights.diff().abs()
    turnover.iloc[0] = weights.iloc[0].abs()
    total_turnover = turnover.sum(axis=1)
    rebalance_turnover = total_turnover[total_turnover > threshold]
    rebalance_days = len(rebalance_turnover)

    contract_turnover_share = turnover.div(total_turnover.replace(0.0, np.nan), axis=0).fillna(0.0)
    rebalance_positions = np.flatnonzero(total_turnover.to_numpy() > threshold)
    days_between = np.diff(rebalance_positions)

    return pd.DataFrame(
        [
            {
                "observations": float(len(weights)),
                "rebalance_days": float(rebalance_days),
                "rebalance_day_rate": float(rebalance_days / len(weights)),
                "avg_daily_turnover": float(total_turnover.mean()),
                "avg_rebalance_turnover": (
                    float(rebalance_turnover.mean()) if not rebalance_turnover.empty else 0.0
                ),
                "median_rebalance_turnover": (
                    float(rebalance_turnover.median()) if not rebalance_turnover.empty else 0.0
                ),
                "p95_rebalance_turnover": (
                    float(rebalance_turnover.quantile(0.95))
                    if not rebalance_turnover.empty
                    else 0.0
                ),
                "max_rebalance_turnover": float(total_turnover.max()),
                "avg_days_between_rebalances": (
                    float(days_between.mean()) if len(days_between) > 0 else 0.0
                ),
                "max_contract_turnover_share": float(contract_turnover_share.max(axis=1).max()),
            }
        ],
        columns=columns,
    )


def exposure_change_summary(weights: pd.DataFrame, threshold: float = 1e-12) -> pd.DataFrame:
    columns = [
        "observations",
        "avg_gross_exposure",
        "avg_net_exposure",
        "avg_abs_net_exposure",
        "avg_daily_gross_exposure_change",
        "max_daily_gross_exposure_change",
        "gross_exposure_change_vol",
        "avg_active_contracts",
        "avg_entries",
        "avg_exits",
        "avg_sign_flips",
        "max_sign_flips",
    ]
    if weights.empty:
        return pd.DataFrame([dict.fromkeys(columns, 0.0)], columns=columns)

    gross_exposure = weights.abs().sum(axis=1)
    net_exposure = weights.sum(axis=1)
    gross_change = gross_exposure.diff().abs()
    gross_change.iloc[0] = gross_exposure.iloc[0]

    active = weights.abs() > threshold
    prior_active = active.shift(1, fill_value=False)
    entries = active & ~prior_active
    exits = ~active & prior_active

    signs = np.sign(weights.where(active, 0.0))
    prior_signs = signs.shift(1, fill_value=0.0)
    sign_flips = active & prior_active & signs.ne(prior_signs)

    return pd.DataFrame(
        [
            {
                "observations": float(len(weights)),
                "avg_gross_exposure": float(gross_exposure.mean()),
                "avg_net_exposure": float(net_exposure.mean()),
                "avg_abs_net_exposure": float(net_exposure.abs().mean()),
                "avg_daily_gross_exposure_change": float(gross_change.mean()),
                "max_daily_gross_exposure_change": float(gross_change.max()),
                "gross_exposure_change_vol": float(gross_change.std(ddof=0)),
                "avg_active_contracts": float(active.sum(axis=1).mean()),
                "avg_entries": float(entries.sum(axis=1).mean()),
                "avg_exits": float(exits.sum(axis=1).mean()),
                "avg_sign_flips": float(sign_flips.sum(axis=1).mean()),
                "max_sign_flips": float(sign_flips.sum(axis=1).max()),
            }
        ],
        columns=columns,
    )


def signal_contribution_summary(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    trend_component: pd.DataFrame,
    carry_component: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "ann_gross_return",
        "ann_trend_attributed_return",
        "ann_carry_attributed_return",
        "ann_unassigned_return",
        "avg_abs_trend_component_share",
        "avg_abs_carry_component_share",
        "trend_dominant_gross_share",
        "carry_dominant_gross_share",
        "trend_carry_agreement_rate",
    ]
    if weights.empty:
        return pd.DataFrame([dict.fromkeys(columns, 0.0)], columns=columns)

    aligned_returns = returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    aligned_trend = trend_component.reindex(
        index=weights.index,
        columns=weights.columns,
    ).fillna(0.0)
    aligned_carry = carry_component.reindex(
        index=weights.index,
        columns=weights.columns,
    ).fillna(0.0)

    prior_weights = weights.shift(1).fillna(0.0)
    gross_by_contract = prior_weights * aligned_returns
    gross_return = gross_by_contract.sum(axis=1)

    absolute_trend = aligned_trend.abs()
    absolute_carry = aligned_carry.abs()
    component_total = (absolute_trend + absolute_carry).replace(0.0, np.nan)
    trend_share = absolute_trend.div(component_total).fillna(0.0)
    carry_share = absolute_carry.div(component_total).fillna(0.0)

    trend_return = (gross_by_contract * trend_share).sum(axis=1)
    carry_return = (gross_by_contract * carry_share).sum(axis=1)
    unassigned_return = gross_return - trend_return - carry_return

    gross_weight = prior_weights.abs()
    daily_gross_weight = gross_weight.sum(axis=1).replace(0.0, np.nan)
    trend_dominant = absolute_trend >= absolute_carry
    carry_dominant = absolute_carry > absolute_trend
    agreeing = (np.sign(aligned_trend) == np.sign(aligned_carry)) & component_total.notna()

    trend_component_share = (trend_share * gross_weight).sum(axis=1).div(daily_gross_weight)
    carry_component_share = (carry_share * gross_weight).sum(axis=1).div(daily_gross_weight)
    trend_dominant_share = gross_weight.where(trend_dominant, 0.0).sum(axis=1).div(
        daily_gross_weight
    )
    carry_dominant_share = gross_weight.where(carry_dominant, 0.0).sum(axis=1).div(
        daily_gross_weight
    )
    agreement_rate = gross_weight.where(agreeing, 0.0).sum(axis=1).div(daily_gross_weight)

    def active_mean(values: pd.Series) -> float:
        clean = values.dropna()
        return float(clean.mean()) if not clean.empty else 0.0

    return pd.DataFrame(
        [
            {
                "ann_gross_return": float(gross_return.mean() * 252),
                "ann_trend_attributed_return": float(trend_return.mean() * 252),
                "ann_carry_attributed_return": float(carry_return.mean() * 252),
                "ann_unassigned_return": float(unassigned_return.mean() * 252),
                "avg_abs_trend_component_share": active_mean(trend_component_share),
                "avg_abs_carry_component_share": active_mean(carry_component_share),
                "trend_dominant_gross_share": active_mean(trend_dominant_share),
                "carry_dominant_gross_share": active_mean(carry_dominant_share),
                "trend_carry_agreement_rate": active_mean(agreement_rate),
            }
        ],
        columns=columns,
    )


def walk_forward_summary(selections: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "windows",
        "unique_parameter_sets",
        "most_common_trend_lookback",
        "most_common_carry_weight",
        "parameter_switch_rate",
        "avg_trend_lookback_change",
        "avg_carry_weight_change",
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
    parameter_changes = selections[["trend_lookback", "carry_weight"]].diff().iloc[1:]
    switched = parameter_changes.ne(0.0).any(axis=1)
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
                "parameter_switch_rate": float(switched.mean()) if not switched.empty else 0.0,
                "avg_trend_lookback_change": (
                    float(parameter_changes["trend_lookback"].abs().mean())
                    if not parameter_changes.empty
                    else 0.0
                ),
                "avg_carry_weight_change": (
                    float(parameter_changes["carry_weight"].abs().mean())
                    if not parameter_changes.empty
                    else 0.0
                ),
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


def parameter_grid_summary(grid: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trend_lookback",
        "carry_weight",
        "windows",
        "selection_count",
        "selection_rate",
        "avg_train_sharpe",
        "avg_test_ann_return",
        "avg_test_sharpe",
        "sharpe_degradation",
        "positive_test_sharpe_rate",
        "avg_test_max_drawdown",
        "worst_test_max_drawdown",
    ]
    if grid.empty:
        return pd.DataFrame(columns=columns)

    grouped = grid.groupby(["trend_lookback", "carry_weight"], sort=True)
    summary = grouped.agg(
        windows=("start", "count"),
        selection_count=("is_selected", "sum"),
        avg_train_sharpe=("train_sharpe", "mean"),
        avg_test_ann_return=("test_ann_return", "mean"),
        avg_test_sharpe=("test_sharpe", "mean"),
        positive_test_sharpe_rate=("test_sharpe", lambda values: (values > 0.0).mean()),
        avg_test_max_drawdown=("test_max_drawdown", "mean"),
        worst_test_max_drawdown=("test_max_drawdown", "min"),
    ).reset_index()
    summary["selection_rate"] = summary["selection_count"] / summary["windows"]
    summary["sharpe_degradation"] = summary["avg_train_sharpe"] - summary["avg_test_sharpe"]

    return (
        summary[columns]
        .sort_values(
            ["avg_test_sharpe", "selection_rate", "trend_lookback", "carry_weight"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )


def selection_quality_summary(grid: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "windows",
        "avg_selected_test_sharpe",
        "avg_best_available_test_sharpe",
        "avg_test_sharpe_regret",
        "max_test_sharpe_regret",
        "zero_regret_window_rate",
        "avg_selected_rank",
        "worst_selected_rank",
    ]
    if grid.empty:
        return pd.DataFrame([dict.fromkeys(columns, 0.0)], columns=columns)

    rows = []
    for _, window in grid.groupby(["start", "end"], sort=True):
        selected = window.loc[window["is_selected"]]
        if selected.empty:
            continue
        selected_row = selected.iloc[0]
        best_sharpe = float(window["test_sharpe"].max())
        ranks = window["test_sharpe"].rank(method="min", ascending=False)
        selected_rank = float(ranks.loc[selected_row.name])
        selected_sharpe = float(selected_row["test_sharpe"])
        rows.append(
            {
                "selected_test_sharpe": selected_sharpe,
                "best_available_test_sharpe": best_sharpe,
                "test_sharpe_regret": best_sharpe - selected_sharpe,
                "selected_rank": selected_rank,
            }
        )

    if not rows:
        return pd.DataFrame([dict.fromkeys(columns, 0.0)], columns=columns)

    quality = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "windows": float(len(quality)),
                "avg_selected_test_sharpe": float(quality["selected_test_sharpe"].mean()),
                "avg_best_available_test_sharpe": float(
                    quality["best_available_test_sharpe"].mean()
                ),
                "avg_test_sharpe_regret": float(quality["test_sharpe_regret"].mean()),
                "max_test_sharpe_regret": float(quality["test_sharpe_regret"].max()),
                "zero_regret_window_rate": float((quality["test_sharpe_regret"] == 0.0).mean()),
                "avg_selected_rank": float(quality["selected_rank"].mean()),
                "worst_selected_rank": float(quality["selected_rank"].max()),
            }
        ],
        columns=columns,
    )


def _bootstrap_metric_samples(
    returns: pd.Series,
    samples: int,
    block: int,
    seed: int,
) -> pd.DataFrame:
    clean = returns.dropna().to_numpy()
    if len(clean) < block or samples <= 0 or block <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        chunks = []
        while sum(len(chunk) for chunk in chunks) < len(clean):
            start = rng.integers(0, len(clean) - block + 1)
            chunks.append(clean[start : start + block])
        sample = pd.Series(np.concatenate(chunks)[: len(clean)])
        rows.append(performance_metrics(sample))
    return pd.DataFrame(rows)


def bootstrap_bands(
    returns: pd.Series,
    samples: int = 500,
    block: int = 21,
    seed: int = 11,
) -> pd.DataFrame:
    metrics = _bootstrap_metric_samples(returns, samples=samples, block=block, seed=seed)
    if metrics.empty:
        return pd.DataFrame()
    return metrics.quantile([0.05, 0.5, 0.95]).rename(index={0.05: "p05", 0.5: "p50", 0.95: "p95"})


def bootstrap_reliability_summary(
    returns: pd.Series,
    samples: int = 500,
    block: int = 21,
    seed: int = 11,
    drawdown_limit: float = -0.10,
) -> pd.DataFrame:
    columns = [
        "samples",
        "block_days",
        "observations",
        "prob_positive_ann_return",
        "prob_positive_sharpe",
        "prob_positive_sortino",
        "prob_drawdown_below_limit",
        "drawdown_limit",
        "median_sharpe",
        "median_max_drawdown",
        "median_cvar_5",
    ]
    metrics = _bootstrap_metric_samples(returns, samples=samples, block=block, seed=seed)
    if metrics.empty:
        return pd.DataFrame([dict.fromkeys(columns, 0.0)], columns=columns)

    return pd.DataFrame(
        [
            {
                "samples": float(len(metrics)),
                "block_days": float(block),
                "observations": float(len(returns.dropna())),
                "prob_positive_ann_return": float((metrics["ann_return"] > 0.0).mean()),
                "prob_positive_sharpe": float((metrics["sharpe"] > 0.0).mean()),
                "prob_positive_sortino": float((metrics["sortino"] > 0.0).mean()),
                "prob_drawdown_below_limit": float(
                    (metrics["max_drawdown"] <= drawdown_limit).mean()
                ),
                "drawdown_limit": float(drawdown_limit),
                "median_sharpe": float(metrics["sharpe"].median()),
                "median_max_drawdown": float(metrics["max_drawdown"].median()),
                "median_cvar_5": float(metrics["cvar_5"].median()),
            }
        ],
        columns=columns,
    )
