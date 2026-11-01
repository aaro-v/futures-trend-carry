"""Real-data futures panel for the trend and carry backtester.

Prices   Yahoo Finance front-month continuous series (``XX=F``): exchange
         settlements of the front contract, held until its last trading day.
Rolls    Last trading days come from exchange rules (``expiry.py``) evaluated on
         a calendar pooled across contracts. The price jump at each roll is
         removed in one of three ways, chosen per contract in contracts.csv:
           eia_curve      exact, from EIA contract 1 and 2 settlements, rolling
                          five trading days before expiry; EIA stopped publishing
                          futures prices on 2024-04-05, after which the
                          drop_roll_day method is used
           no_arbitrage   jump = cost of carry x time to the next expiry
                          (short rate, dividend yield, rate differential or
                          financing minus coupon)
           drop_roll_day  no term-structure history is freely available, so
                          the switch-day return is set to zero
Carry    Annualised roll yield to a long position, in decimal per year:
         dividend yield minus short rate (equity indices), yield minus bill
         plus roll-down (Treasuries), rate differential (FX), ln(F1/F2) x 12
         (energy, to 2024-04-05). Not available for the other sectors.
Costs    commission_bps is $2.50 per contract side over notional; roll_cost_bps
         is one tick of spread plus two commissions over notional, on roll days.

Outputs, written to ``--out`` (default ``data/real``):

    panel.parquet            long frame; the first seven columns match panel_to_frame
    contracts_resolved.csv   contract specs with coverage, cost and roll diagnostics
    roll_schedule.csv        every roll with its last trading day and the adjustment
    rates_daily.parquet      short rates, Treasury curve and dividend yields
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import yahoo
from .expiry import TradingCalendar, roll_schedule
from .macro import eia_daily, fred_series, lagged_monthly, to_daily

CONTRACTS = Path(__file__).resolve().parent / "contracts.csv"
DOWNLOAD_START = "2000-01-01"
PANEL_START = "2003-01-01"
COMMISSION_USD = 2.50  # per contract per side, all-in (broker + exchange + clearing)
ROLL_SPREAD_TICKS = 1.0  # calendar-spread order assumed to cross one tick
ENERGY_ROLL_LEAD = 5  # trading days before expiry when the EIA-based series rolls
MAX_GAP_DAYS = 10  # calendar days; longer holes in Yahoo's history are treated as missing data

TENORS = {
    0.25: "DGS3MO",
    1: "DGS1",
    2: "DGS2",
    3: "DGS3",
    5: "DGS5",
    7: "DGS7",
    10: "DGS10",
    20: "DGS20",
    30: "DGS30",
}
INTERBANK = {
    "US": "IR3TIB01USM156N",
    "EZ": "IR3TIB01EZM156N",
    "JP": "IR3TIB01JPM156N",
    "GB": "IR3TIB01GBM156N",
    "AU": "IR3TIB01AUM156N",
    "CA": "IR3TIB01CAM156N",
    "CH": "IR3TIB01CHM156N",
    "NZ": "IR3TIB01NZM156N",
    "MX": "IR3TIB01MXM156N",
}
# Daily overnight rates used to extend OECD 3m series that stopped updating.
OVERNIGHT = {"EZ": "ECBESTRVOLWGTTRMDMNRT", "GB": "IUDSOIA"}
PANEL_COLUMNS = ["date", "contract", "price", "carry", "roll_cost_bps", "sector", "commission_bps"]


# --------------------------------------------------------------------------- rates
def build_rates(index: pd.DatetimeIndex) -> pd.DataFrame:
    rates = pd.DataFrame(index=index)
    rates["us_bill_3m"] = to_daily(fred_series("DTB3") / 100, index)
    for tenor, code in TENORS.items():
        rates[f"ust_{tenor:g}y"] = to_daily(fred_series(code) / 100, index)
    for country, code in INTERBANK.items():
        monthly = fred_series(code).dropna() / 100
        daily = to_daily(lagged_monthly(monthly, 1), index)
        if country in OVERNIGHT:
            daily = _extend_with_overnight(monthly, daily, country, index)
        rates[f"ib3m_{country}"] = daily
    return rates


def _extend_with_overnight(monthly, daily, country, index) -> pd.Series:
    """Past the last OECD print, use the daily overnight rate plus the recent 3m-ON spread."""

    overnight = fred_series(OVERNIGHT[country]).dropna() / 100
    last_month = monthly.index.max()
    recent = monthly.loc[last_month - pd.DateOffset(months=11) :]
    on_monthly = overnight.resample("MS").mean().reindex(recent.index)
    spread = float((recent - on_monthly).mean()) if on_monthly.notna().any() else 0.0
    stale_from = last_month + pd.DateOffset(months=2)  # one month after it was last refreshed
    extension = to_daily(overnight.shift(1, freq="B") + spread, index)
    return daily.where(index < stale_from, extension)


def interpolate_yield(rates: pd.DataFrame, maturity: float) -> pd.Series:
    tenors = np.array(sorted(TENORS))
    columns = [f"ust_{t:g}y" for t in tenors]
    values = rates[columns].to_numpy()
    out = np.full(len(rates), np.nan)
    for i, row in enumerate(values):
        ok = ~np.isnan(row)
        if ok.sum() >= 2:
            out[i] = np.interp(maturity, tenors[ok], row[ok])
    return pd.Series(out, index=rates.index)


def dividend_yield(symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    hist = yahoo.history(symbol, start=DOWNLOAD_START)
    close = hist.bars["close"]
    dividends = hist.dividends.reindex(close.index, fill_value=0.0)
    trailing = dividends.rolling("365D").sum()
    yld = (trailing / close).where(close.index >= close.index[0] + pd.Timedelta(days=365))
    return to_daily(yld, index)


# --------------------------------------------------------------------------- cleaning
def flag_bad_ticks(close: pd.Series) -> pd.Series:
    """Quarantine extreme/nonpositive prints using only history known at that close."""

    positive = close.where(close > 0)
    logp = np.log(positive)
    d = logp.diff()
    sigma = (
        1.4826
        * (d - d.rolling(63, min_periods=20).median()).abs().rolling(63, min_periods=20).median()
    )
    threshold = np.maximum(0.35, 12 * sigma.shift(1).fillna(0.03))
    return ((d.abs() > threshold) | (close <= 0) | ~np.isfinite(close)).fillna(False)


# --------------------------------------------------------------------------- per contract
def eia_curve(code: str, area: str) -> pd.DataFrame:
    return pd.concat({f"F{k}": eia_daily(code.format(k), area) for k in (1, 2)}, axis=1).dropna()


def eia_adjusted_returns(curve: pd.DataFrame, rule: str, cycle: str):
    """Daily returns of a position that rolls ENERGY_ROLL_LEAD days before each expiry."""

    cal = TradingCalendar(curve.index)
    schedule = roll_schedule(rule, cycle, cal)
    pos = {d: i for i, d in enumerate(curve.index)}
    n = len(curve)
    held_second = np.zeros(n, dtype=bool)
    first_after_expiry = np.zeros(n, dtype=bool)
    roll_days = []
    for ltd in schedule["last_trading_day"]:
        i = pos.get(ltd)
        if i is None:
            continue
        held_second[max(0, i - ENERGY_ROLL_LEAD) : i + 1] = True
        roll_days.append(curve.index[max(0, i - ENERGY_ROLL_LEAD)])
        if i + 1 < n:
            first_after_expiry[i + 1] = True
    f1, f2 = curve["F1"].to_numpy(), curve["F2"].to_numpy()
    r = np.full(n, np.nan)
    for t in range(1, n):
        if held_second[t - 1]:
            r[t] = (f1[t] if first_after_expiry[t] else f2[t]) / f2[t - 1] - 1
        else:
            r[t] = f1[t] / f1[t - 1] - 1
    returns = pd.Series(r, index=curve.index)
    carry = np.log(curve["F1"].where(curve["F1"] > 0) / curve["F2"].where(curve["F2"] > 0)) * 12
    return returns, carry, pd.DatetimeIndex(roll_days), schedule


def cost_of_carry(
    spec: pd.Series, rates: pd.DataFrame, cache: dict
) -> tuple[pd.Series, pd.Series, str]:
    """(c, carry, source): F = S*exp(c*T); carry is the annual roll yield to a long position."""

    method, *args = spec["carry_method"].split(":")
    idx = rates.index
    nan = pd.Series(np.nan, index=idx)
    if spec["roll_method"] == "no_arbitrage" and spec["sector"] == "metals":
        return rates["us_bill_3m"], nan, "unavailable (metal lease rates are not public)"
    if method == "dividend_yield":
        etf, currency = args
        if etf not in cache:
            cache[etf] = dividend_yield(etf, idx)
        q = cache[etf]
        rates[f"div_yield_{etf}"] = q
        r = rates["us_bill_3m"] if currency == "USD" else rates["ib3m_JP"]
        return r - q, q - r, f"{etf} trailing dividend yield - {currency} short rate"
    if method == "fx":
        foreign = rates[f"ib3m_{args[0]}"]
        us = rates["ib3m_US"]
        return us - foreign, foreign - us, f"OECD 3m interbank {args[0]} - US (lagged 1m)"
    if method == "bond":
        maturity, duration = float(args[0]), float(args[1])
        y = interpolate_yield(rates, maturity)
        y_rolled = interpolate_yield(rates, maturity - 1)
        excess = y - rates["us_bill_3m"]
        carry = excess + duration * (y - y_rolled)
        return -excess, carry, f"UST {maturity:g}y yield - 3m bill + roll-down (D={duration:g})"
    return nan, nan, "unavailable (no free historical term structure)"


def exchange_calendar(
    histories: dict[str, pd.DataFrame], min_share: float = 0.2
) -> TradingCalendar:
    """Dates on which at least ``min_share`` of all contracts traded.

    Individual Yahoo histories occasionally miss a session (e.g. Columbus Day for
    lean hogs), which would shift "n-th business day" expiry rules; the pooled
    calendar does not.
    """

    counts = pd.concat([pd.Series(1, index=b.index) for b in histories.values()], axis=1).sum(
        axis=1
    )
    dates = counts.index[counts >= min_share * len(histories)]
    # Yahoo drops Columbus Day and Veterans Day in some years for every contract, but
    # CME and ICE US are open on both, so they count as business days in expiry rules.
    extra = []
    for year in range(dates[0].year, dates[-1].year + 1):
        october = pd.date_range(f"{year}-10-01", f"{year}-10-14", freq="W-MON")
        extra.append(october[1])  # second Monday of October
        veterans = pd.Timestamp(f"{year}-11-11")
        if veterans.dayofweek < 5:
            extra.append(veterans)
    return TradingCalendar(dates.union(pd.DatetimeIndex(extra)))


def build_contract(
    spec: pd.Series, bars: pd.DataFrame, calendar: TradingCalendar, rates: pd.DataFrame, cache: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Fixed quality cutoffs belong to the contract specification, so a later
    # data gap cannot retrospectively delete an earlier trading history.
    history_start = spec.get("history_start", PANEL_START)
    history_start = PANEL_START if pd.isna(history_start) else history_start
    bars = bars.loc[history_start:].copy()
    gaps = bars.index.to_series().diff().dt.days
    after_gap = (gaps > MAX_GAP_DAYS).to_numpy()
    bad = flag_bad_ticks(bars["close"])
    clean = bars["close"].mask(bad).ffill()
    raw_returns = clean.pct_change()

    schedule = roll_schedule(spec["expiry_rule"], spec["cycle"], calendar)
    schedule = schedule[
        (schedule["switch_day"] > bars.index[0]) & (schedule["switch_day"] <= bars.index[-1])
    ].reset_index(drop=True)
    # Map each switch day onto the first session this contract actually has data for.
    own = np.searchsorted(
        bars.index.values, schedule["switch_day"].values.astype(bars.index.values.dtype)
    )
    schedule["switch_day"] = bars.index[np.minimum(own, len(bars) - 1)]
    c, carry, carry_source = cost_of_carry(spec, rates, cache)
    c, carry = c.reindex(bars.index), carry.reindex(bars.index)

    adjusted = raw_returns.copy()
    schedule["method"] = spec["roll_method"]
    schedule["log_jump_removed"] = np.nan
    roll_days = pd.DatetimeIndex(schedule["switch_day"])
    eia_end = None
    if spec["roll_method"] == "eia_curve":
        code, area = spec["carry_method"].split(":")[1:]
        curve = eia_curve(code, area)
        eia_returns, eia_carry, eia_roll_days, _ = eia_adjusted_returns(
            curve, spec["expiry_rule"], spec["cycle"]
        )
        eia_end = curve.index.max()
        # Price index on EIA dates, then mapped onto Yahoo dates (keeps multi-day moves intact).
        eia_index = (1 + eia_returns.fillna(0)).cumprod()
        mapped = eia_index.reindex(eia_index.index.union(bars.index)).ffill().reindex(bars.index)
        in_eia = (bars.index >= curve.index.min()) & (bars.index <= eia_end)
        adjusted[in_eia] = mapped.pct_change()[in_eia]
        carry = eia_carry.reindex(bars.index).where(in_eia)
        carry_source = "EIA contract 1/2 settlements: ln(F1/F2) x 12 (ends 2024-04-05)"
        late = schedule["switch_day"] > eia_end
        adjusted.loc[schedule.loc[late, "switch_day"]] = 0.0
        schedule.loc[late, "method"] = "drop_roll_day"
        schedule.loc[~late, "method"] = f"eia_curve (roll {ENERGY_ROLL_LEAD}d before expiry)"
        roll_days = pd.DatetimeIndex(
            list(eia_roll_days[eia_roll_days >= bars.index[0]])
            + list(schedule.loc[late, "switch_day"])
        )
    elif spec["roll_method"] == "no_arbitrage":
        for i, row in schedule.iterrows():
            if pd.isna(row["next_last_trading_day"]) or row["switch_day"] not in adjusted.index:
                continue
            years = (row["next_last_trading_day"] - row["last_trading_day"]).days / 365.25
            jump = float(c.get(row["last_trading_day"], np.nan)) * years
            if np.isfinite(jump):
                s = row["switch_day"]
                adjusted[s] = (1 + adjusted[s]) * np.exp(-jump) - 1
                schedule.loc[i, "log_jump_removed"] = jump
    else:  # drop_roll_day
        adjusted.loc[adjusted.index.intersection(roll_days)] = 0.0

    adjusted[after_gap] = 0.0  # P&L across a multi-week hole in the data is not observable
    adjusted = adjusted.fillna(0.0)
    index = (1 + adjusted).cumprod()
    price = clean.iloc[-1] * index / index.iloc[-1]

    point_value = float(spec["point_value_usd"])
    notional = clean.abs().replace(0, np.nan) * point_value
    commission_bps = COMMISSION_USD / notional * 1e4
    is_roll = bars.index.isin(roll_days)
    roll_cost_bps = pd.Series(0.0, index=bars.index)
    roll_cost_bps[is_roll] = (
        (ROLL_SPREAD_TICKS * float(spec["tick_size"]) * point_value + 2 * COMMISSION_USD)
        / notional[is_roll]
        * 1e4
    )

    frame = pd.DataFrame(
        {
            "date": bars.index,
            "contract": spec["root"],
            "price": price.to_numpy(),
            "carry": carry.to_numpy(),
            "roll_cost_bps": roll_cost_bps.to_numpy(),
            "sector": spec["sector"],
            "commission_bps": commission_bps.to_numpy(),
            "return": adjusted.to_numpy(),
            "price_raw": bars["close"].to_numpy(),
            "open": bars["open"].to_numpy(),
            "high": bars["high"].to_numpy(),
            "low": bars["low"].to_numpy(),
            "volume": bars["volume"].to_numpy(),
            "is_roll_day": is_roll,
            "bad_tick": bad.to_numpy(),
            "after_data_gap": after_gap,
        }
    )

    # Roll-date diagnostics on raw prices: a correctly dated switch day carries the roll
    # jump, so |return| is elevated and (for no_arbitrage) correlates with the modelled jump.
    raw_log = np.log(clean.where(clean > 0)).diff()
    on_switch = raw_log.reindex(schedule["switch_day"])
    abs_ratio = float(on_switch.abs().mean() / raw_log.abs().mean())
    jumps = schedule["log_jump_removed"].to_numpy()
    usable = np.isfinite(jumps) & np.isfinite(on_switch.to_numpy())
    jump_corr = (
        float(np.corrcoef(jumps[usable], on_switch.to_numpy()[usable])[0, 1])
        if usable.sum() > 10
        else None
    )
    last5 = commission_bps.loc[bars.index[-1] - pd.DateOffset(years=5) :]
    stats = {
        "root": spec["root"],
        "first_date": bars.index[0].date(),
        "last_date": bars.index[-1].date(),
        "trading_days": len(bars),
        "rolls": len(schedule),
        "bad_ticks_removed": int(bad.sum()),
        "data_gaps_zeroed": int(after_gap.sum()),
        "carry_source": carry_source,
        "carry_coverage": round(float(carry.notna().mean()), 3),
        "commission_bps_median_5y": round(float(last5.median()), 4),
        "roll_cost_bps_median": round(float(roll_cost_bps[is_roll].median()), 4)
        if is_roll.any()
        else None,
        "switch_day_abs_return_ratio": round(abs_ratio, 2),
        "switch_day_jump_corr": round(jump_corr, 2) if jump_corr is not None else None,
        "eia_curve_end": eia_end.date() if eia_end is not None else None,
    }
    schedule.insert(0, "root", spec["root"])
    return frame, schedule, stats


@dataclass
class FuturesBuild:
    panel: pd.DataFrame
    contracts: pd.DataFrame
    schedule: pd.DataFrame
    rates: pd.DataFrame


def build_panel(
    specs: pd.DataFrame | None = None, verbose: bool = True, end: str | None = None
) -> FuturesBuild:
    """Assemble the roll-adjusted panel for every contract in ``specs``."""

    specs = pd.read_csv(CONTRACTS) if specs is None else specs
    last = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    index = pd.bdate_range(DOWNLOAD_START, last)
    rates = build_rates(index)
    cache: dict = {}
    histories = {
        row.root: yahoo.history(
            row.yahoo, start=DOWNLOAD_START, end=str((last + pd.Timedelta(days=1)).date())
        ).bars.loc[:last]
        for row in specs.itertuples(index=False)
    }
    calendar = exchange_calendar(histories)
    frames, schedules, stats = [], [], []
    for spec in specs.itertuples(index=False):
        spec = pd.Series(spec._asdict())
        frame, schedule, stat = build_contract(
            spec, histories[spec["root"]], calendar, rates, cache
        )
        frames.append(frame)
        schedules.append(schedule)
        stats.append(stat)
        if verbose:
            print(
                f"{spec['root']:4s} {stat['first_date']}..{stat['last_date']} "
                f"rolls={stat['rolls']:3d} carry={stat['carry_coverage']:.2f}",
                file=sys.stderr,
            )

    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "contract"], ignore_index=True
    )
    float_cols = [
        "price",
        "carry",
        "roll_cost_bps",
        "commission_bps",
        "return",
        "price_raw",
        "open",
        "high",
        "low",
    ]
    panel[float_cols] = panel[float_cols].astype("float64")
    panel["contract"] = panel["contract"].astype("category")
    panel["sector"] = panel["sector"].astype("category")
    return FuturesBuild(
        panel=panel,
        contracts=specs.merge(pd.DataFrame(stats), on="root"),
        schedule=pd.concat(schedules, ignore_index=True),
        rates=rates.loc[PANEL_START:].dropna(how="all"),
    )


def build(out: Path, end: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    result = build_panel(end=end)
    result.panel.to_parquet(out / "panel.parquet", index=False, compression="zstd")
    result.contracts.to_csv(out / "contracts_resolved.csv", index=False)
    result.schedule.to_csv(out / "roll_schedule.csv", index=False)
    result.rates.astype("float32").to_parquet(out / "rates_daily.parquet", compression="zstd")
    print(
        f"futures panel: {result.panel['contract'].nunique()} contracts, "
        f"{len(result.panel):,} rows -> {out}"
    )
