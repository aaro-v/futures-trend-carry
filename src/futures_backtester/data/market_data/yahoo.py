"""Daily price history from Yahoo Finance's public chart endpoint.

Conventions of the returned frame (verified against EIA settlements for CL=F):

* ``open/high/low/close`` are split-adjusted but NOT dividend-adjusted.
* ``adj_close`` is split- and dividend-adjusted (total-return price).
* ``volume`` is split-adjusted.
* For futures (``XX=F``) the series is the front contract held until expiry,
  unadjusted for rolls; closes are exchange settlement prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .http import get_json

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass
class YahooHistory:
    symbol: str
    bars: pd.DataFrame
    dividends: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    splits: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    meta: dict = field(default_factory=dict)


def history(symbol: str, start: str = "2000-01-01", end: str | None = None) -> YahooHistory:
    end = end or str((pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).date())
    params = {
        "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
        "period2": int(pd.Timestamp(end, tz="UTC").timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    payload = get_json(CHART_URL.format(symbol=symbol), params=params)["chart"]
    if not payload.get("result"):
        raise LookupError(f"Yahoo returned no data for {symbol}: {payload.get('error')}")
    result = payload["result"][0]
    meta = result.get("meta", {})
    timestamps = result.get("timestamp") or []
    if not timestamps:
        raise LookupError(f"Yahoo returned an empty history for {symbol}")

    tz = meta.get("exchangeTimezoneName") or "UTC"
    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    )
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    bars = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adj if adj is not None else quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.DatetimeIndex(dates, name="date"),
        dtype=float,
    )
    bars = _clean(bars)

    events = result.get("events", {})
    dividends = pd.Series(
        {
            _event_date(v["date"], tz): float(v["amount"])
            for v in events.get("dividends", {}).values()
        },
        dtype=float,
    ).sort_index()
    splits = pd.Series(
        {
            _event_date(v["date"], tz): float(v["numerator"]) / float(v["denominator"])
            for v in events.get("splits", {}).values()
        },
        dtype=float,
    ).sort_index()
    return YahooHistory(symbol=symbol, bars=bars, dividends=dividends, splits=splits, meta=meta)


def _event_date(ts: int, tz: str) -> pd.Timestamp:
    return pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(tz).tz_localize(None).normalize()


def _clean(bars: pd.DataFrame) -> pd.DataFrame:
    """Drop weekend stubs, empty rows and duplicate dates (keep the last print)."""

    bars = bars[bars.index.dayofweek < 5]
    bars = bars.dropna(subset=["close"])
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    for column in ("open", "high", "low"):
        bars[column] = bars[column].fillna(bars["close"])
    bars["volume"] = bars["volume"].fillna(0.0)
    bars.loc[bars["adj_close"].isna(), "adj_close"] = bars["close"]
    return bars.replace([np.inf, -np.inf], np.nan)


def split_factor_after(splits: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Product of split ratios strictly after each date (to convert raw counts to today's basis)."""

    factors = np.ones(len(dates))
    for split_date, ratio in splits.items():
        factors[dates < split_date] *= ratio
    return factors
