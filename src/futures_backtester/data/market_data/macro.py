"""Rates and energy term-structure data from FRED and the U.S. EIA."""

from __future__ import annotations

import io
import warnings

import pandas as pd

from .http import get

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
EIA_URL = "https://www.eia.gov/dnav/{area}/hist_xls/{code}.xls"


def fred_series(series_id: str) -> pd.Series:
    """Return a FRED series in percent, indexed by observation date (NaN for missing)."""

    raw = pd.read_csv(io.BytesIO(get(FRED_URL, params={"id": series_id})))
    raw.columns = ["date", "value"]
    values = pd.to_numeric(raw["value"], errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.to_datetime(raw["date"]), name=series_id)


def eia_daily(code: str, area: str = "pet") -> pd.Series:
    """Return an EIA daily history (e.g. RCLC1 = NYMEX WTI contract 1, $/bbl)."""

    body = get(EIA_URL.format(area=area, code=f"{code}d"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = pd.read_excel(io.BytesIO(body), sheet_name="Data 1", skiprows=2)
    frame.columns = ["date", "value"]
    frame = frame.dropna()
    series = pd.Series(
        pd.to_numeric(frame["value"], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame["date"]),
        name=code,
    )
    return series[~series.index.duplicated(keep="last")].sort_index()


def lagged_monthly(series: pd.Series, lag_months: int = 1) -> pd.Series:
    """Shift a monthly series so each value is only used after it would have been published."""

    shifted = series.copy()
    shifted.index = shifted.index + pd.DateOffset(months=lag_months)
    return shifted


def to_daily(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a lower-frequency series onto a daily index (no look-ahead)."""

    combined = series.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.reindex(combined.index.union(index)).ffill().reindex(index)
