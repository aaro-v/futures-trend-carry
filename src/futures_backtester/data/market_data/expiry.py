"""Last-trading-day rules for the futures in ``contracts.csv``.

Rules follow the exchange contract specifications (CME Group / ICE US) and are
evaluated on the supplied trading calendar. The builder pools observed sessions
across contracts. It assumes that Yahoo's ``XX=F`` series holds the front
contract until expiry and switches on the following session.
"""

from __future__ import annotations

import calendar as _calendar

import numpy as np
import pandas as pd

MONTH_CODES = {c: i for i, c in enumerate("FGHJKMNQUVXZ", start=1)}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=_calendar.monthrange(year, month)[1])
    return last - pd.Timedelta(days=(last.weekday() - weekday) % 7)


def _shift_month(year: int, month: int, k: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + k
    return index // 12, index % 12 + 1


class TradingCalendar:
    def __init__(self, dates: pd.DatetimeIndex):
        self.dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates).normalize())))
        self._values = self.dates.values

    def _at(self, i: int) -> pd.Timestamp | None:
        return self.dates[i] if 0 <= i < len(self.dates) else None

    def first_on_or_after(self, day: pd.Timestamp) -> int:
        return int(np.searchsorted(self._values, np.datetime64(day), side="left"))

    def on_or_before(self, day: pd.Timestamp) -> pd.Timestamp | None:
        return self._at(int(np.searchsorted(self._values, np.datetime64(day), side="right")) - 1)

    def before(self, day: pd.Timestamp, n: int) -> pd.Timestamp | None:
        """The n-th trading day strictly before ``day``."""
        return self._at(self.first_on_or_after(day) - n)

    def last_of_month(self, year: int, month: int) -> pd.Timestamp | None:
        end = pd.Timestamp(year=year, month=month, day=_calendar.monthrange(year, month)[1])
        last = self.on_or_before(end)
        return last if last is not None and last.month == month and last.year == year else None

    def offset(self, day: pd.Timestamp, n: int) -> pd.Timestamp | None:
        i = int(np.searchsorted(self._values, np.datetime64(day), side="left"))
        return self._at(i + n)

    def next_after(self, day: pd.Timestamp) -> pd.Timestamp | None:
        return self._at(int(np.searchsorted(self._values, np.datetime64(day), side="right")))


def last_trading_day(rule: str, year: int, month: int, cal: TradingCalendar) -> pd.Timestamp | None:
    """Evaluate an expiry rule for the contract delivering in ``year``-``month``."""

    name, _, arg = rule.partition(":")
    n = int(arg) if arg else 0
    if name == "third_friday":
        return cal.on_or_before(_nth_weekday(year, month, 4, 3))
    if name == "bday_before_2nd_friday":
        return cal.before(_nth_weekday(year, month, 4, 2), 1)
    if name == "bdays_before_3rd_wednesday":
        return cal.before(_nth_weekday(year, month, 2, 3), n)
    if name == "last_bday":
        return cal.last_of_month(year, month)
    if name == "bdays_before_last_bday":
        last = cal.last_of_month(year, month)
        return None if last is None else cal.offset(last, -n)
    if name == "third_last_bday":
        last = cal.last_of_month(year, month)
        return None if last is None else cal.offset(last, -2)
    if name == "cl_rule":
        # 3 business days before the 25th of the prior month; if the 25th is not a
        # business day, 3 business days before the business day preceding the 25th.
        y, m = _shift_month(year, month, -1)
        day25 = pd.Timestamp(year=y, month=m, day=25)
        i = cal.first_on_or_after(day25)
        is_trading = i < len(cal.dates) and cal.dates[i] == day25
        return cal._at(i - 3 if is_trading else i - 4)
    if name == "last_bday_prior_month":
        return cal.last_of_month(*_shift_month(year, month, -1))
    if name == "last_bday_2nd_prior_month":
        return cal.last_of_month(*_shift_month(year, month, -2))
    if name == "bdays_before_month_start":
        return cal.before(pd.Timestamp(year=year, month=month, day=1), n)
    if name == "bday_before_15th":
        return cal.before(pd.Timestamp(year=year, month=month, day=15), 1)
    if name == "last_thursday":
        return cal.on_or_before(_last_weekday(year, month, 3))
    if name == "nth_bday":
        return cal.offset(pd.Timestamp(year=year, month=month, day=1), n - 1)
    raise ValueError(f"unknown expiry rule: {rule}")


def roll_schedule(rule: str, cycle: str, cal: TradingCalendar) -> pd.DataFrame:
    """Every listed contract in the calendar window: month, last trading day, switch day."""

    rows = []
    first, last = cal.dates[0], cal.dates[-1]
    for year in range(first.year, last.year + 2):
        for code in cycle:
            month = MONTH_CODES[code]
            ltd = last_trading_day(rule, year, month, cal)
            if ltd is None or ltd < first or ltd >= last:
                continue
            rows.append(
                {
                    "contract": f"{code}{year}",
                    "last_trading_day": ltd,
                    "switch_day": cal.next_after(ltd),
                }
            )
    schedule = pd.DataFrame(rows).dropna().sort_values("last_trading_day", ignore_index=True)
    schedule["next_last_trading_day"] = schedule["last_trading_day"].shift(-1)
    return schedule
