"""NSE trading calendar.

Used to answer one question: on a day the exchange was open, did we get a bar?

Missing sessions are the quiet killer in daily-bar pipelines. A gap looks
identical to a holiday, and a forward-fill over a gap manufactures a zero return
that the model will happily learn from.

Two backends, and the distinction is reported rather than hidden:

``exchange``
    ``pandas_market_calendars`` with the XNSE calendar. Knows real NSE holidays,
    including one-off closures and special trading sessions.

``weekday``
    Fallback when that package is unavailable: Monday-Friday minus a small
    static list of fixed-date national holidays. It will flag Indian festival
    holidays — which move every year — as false "missing sessions".

The fallback deliberately downgrades its findings to INFO severity rather than
WARNING, because a validator that cries wolf on every Diwali trains you to
ignore it.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

log = logging.getLogger(__name__)

try:  # pragma: no cover - depends on environment
    import pandas_market_calendars as mcal

    _HAS_MCAL = True
except ImportError:  # pragma: no cover
    mcal = None
    _HAS_MCAL = False


# Fixed-date national holidays only. Festival dates are lunar/lunisolar and
# cannot be hard-coded; that is precisely the fallback's known weakness.
_FIXED_HOLIDAYS = {(1, 26), (5, 1), (8, 15), (10, 2), (12, 25)}


class TradingCalendar:
    """Trading sessions for a single exchange."""

    def __init__(self, exchange: str = "NSE") -> None:
        self.exchange = exchange
        self.backend = "exchange" if _HAS_MCAL else "weekday"
        self._cal = None

        if _HAS_MCAL:
            try:
                self._cal = mcal.get_calendar("XNSE")
            except Exception as exc:  # pragma: no cover
                log.warning(
                    "XNSE calendar unavailable (%s); falling back to weekday "
                    "approximation. Missing-session findings will be noisy.",
                    exc,
                )
                self.backend = "weekday"
        else:
            log.warning(
                "pandas_market_calendars not installed; using weekday "
                "approximation for %s. Install it for accurate holiday "
                "handling: pip install pandas-market-calendars",
                exchange,
            )

    @property
    def is_approximate(self) -> bool:
        return self.backend == "weekday"

    def sessions(self, start: date, end: date) -> list[date]:
        """Trading days in ``[start, end]``, inclusive, ascending."""
        if start > end:
            return []
        if self.backend == "exchange":
            sched = self._cal.schedule(start_date=start, end_date=end)
            return [d.date() for d in sched.index]
        return self._weekday_sessions(start, end)

    @staticmethod
    def _weekday_sessions(start: date, end: date) -> list[date]:
        out, cur = [], start
        while cur <= end:
            if cur.weekday() < 5 and (cur.month, cur.day) not in _FIXED_HOLIDAYS:
                out.append(cur)
            cur += timedelta(days=1)
        return out

    def is_session(self, day: date) -> bool:
        return bool(self.sessions(day, day))

    def next_session(self, day: date, lookahead: int = 10) -> date | None:
        """The first trading day strictly after ``day``.

        This is what turns a decision at *t* into an execution at *t+1*. Returns
        None if no session is found within ``lookahead`` days, which the caller
        must treat as "cannot execute" rather than silently using ``day``.
        """
        found = self.sessions(day + timedelta(days=1), day + timedelta(days=lookahead))
        return found[0] if found else None

    def missing_sessions(self, observed: pd.Series, start: date, end: date) -> list[date]:
        """Expected sessions with no corresponding bar."""
        have = {pd.Timestamp(d).date() for d in observed}
        return [d for d in self.sessions(start, end) if d not in have]

    def unexpected_sessions(self, observed: pd.Series, start: date, end: date) -> list[date]:
        """Bars on days the calendar says the exchange was closed.

        Under the exchange backend these are worth investigating — usually a
        timezone error or a special session the calendar lacks.
        """
        expected = set(self.sessions(start, end))
        return [
            pd.Timestamp(d).date()
            for d in observed
            if start <= pd.Timestamp(d).date() <= end
            and pd.Timestamp(d).date() not in expected
        ]
