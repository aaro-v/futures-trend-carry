import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from typer.testing import CliRunner

from futures_backtester.cli import app
from futures_backtester.data.market_data import build
from futures_backtester.data.market_data.expiry import TradingCalendar, last_trading_day
from futures_backtester.data.market_data.macro import lagged_monthly, to_daily


def test_tick_filter_uses_only_past_observations() -> None:
    index = pd.bdate_range("2003-01-01", periods=250)
    close = pd.Series(np.exp(np.random.default_rng(1).normal(0, 0.01, 250).cumsum()), index=index)
    close.iloc[130] *= 5
    full = build.flag_bad_ticks(close)
    assert full.iloc[130]
    assert_series_equal(full.iloc[:131], build.flag_bad_ticks(close.iloc[:131]))


def test_later_gaps_cannot_delete_earlier_contract_history(monkeypatch) -> None:
    dates = pd.bdate_range("2003-01-02", periods=650)
    missing = dates[300:320].union(dates[400:420]).union(dates[500:520])
    observed = dates.difference(missing)
    prices = np.exp(np.arange(len(observed)) * 0.0001) * 100
    bars = pd.DataFrame(dict.fromkeys(("close", "open", "high", "low"), prices), index=observed)
    bars["volume"] = 1000.0
    spec = pd.read_csv(build.CONTRACTS).set_index("root", drop=False).loc["ES"]
    monkeypatch.setattr(
        build,
        "cost_of_carry",
        lambda *args: (pd.Series(0.0, index=dates), pd.Series(0.01, index=dates), "constant"),
    )
    calendar = TradingCalendar(dates)
    full, _, _ = build.build_contract(spec, bars, calendar, pd.DataFrame(index=dates), {})
    short, _, _ = build.build_contract(
        spec, bars.loc[: dates[250]], calendar, pd.DataFrame(index=dates), {}
    )
    assert full.date.iloc[0] == short.date.iloc[0] == dates[0]
    np.testing.assert_allclose(full["return"].iloc[: len(short)], short["return"])


def test_monthly_values_become_available_after_publication_lag() -> None:
    monthly = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    dates = pd.bdate_range("2024-01-01", "2024-03-04")
    daily = to_daily(lagged_monthly(monthly), dates)
    assert daily.loc[:"2024-01-31"].isna().all()
    assert daily.loc["2024-02-01"] == 0.01
    assert daily.loc["2024-03-01"] == 0.02


def test_expiry_rules_respect_calendar() -> None:
    calendar = TradingCalendar(pd.bdate_range("2024-01-01", "2024-12-31"))
    assert last_trading_day("third_friday", 2024, 3, calendar) == pd.Timestamp("2024-03-15")
    assert last_trading_day("cl_rule", 2024, 4, calendar) == pd.Timestamp("2024-03-20")


def test_build_command_passes_output_and_end_date(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(build, "build", lambda out, end: calls.append((out, end)))
    result = CliRunner().invoke(app, ["build-data", "--out", str(tmp_path), "--end", "2026-10-30"])
    assert result.exit_code == 0, result.exception
    assert calls == [(tmp_path, "2026-10-30")]
