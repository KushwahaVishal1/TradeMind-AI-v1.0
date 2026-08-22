"""Trading calendar behaviour.

``next_session`` is load-bearing: it is what turns a decision at t into an
execution at t+1. If it ever returns the same day, the backtester silently
gains look-ahead.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from trademind.ingestion.calendar import TradingCalendar


def cal():
    return TradingCalendar()


def test_weekends_are_not_sessions():
    c = cal()
    assert not c.is_session(date(2023, 6, 17))   # Saturday
    assert not c.is_session(date(2023, 6, 18))   # Sunday
    assert c.is_session(date(2023, 6, 16))       # Friday


def test_sessions_are_ascending_and_bounded():
    days = cal().sessions(date(2023, 6, 1), date(2023, 6, 30))
    assert days == sorted(days)
    assert all(date(2023, 6, 1) <= d <= date(2023, 6, 30) for d in days)


def test_inverted_range_returns_nothing():
    assert cal().sessions(date(2023, 6, 30), date(2023, 6, 1)) == []


def test_next_session_is_strictly_later():
    """The core t -> t+1 guarantee."""
    c = cal()
    for day in c.sessions(date(2023, 6, 1), date(2023, 6, 30)):
        nxt = c.next_session(day)
        assert nxt is not None
        assert nxt > day


def test_next_session_skips_the_weekend():
    nxt = cal().next_session(date(2023, 6, 16))   # Friday
    assert nxt == date(2023, 6, 19)               # Monday


def test_next_session_gives_up_rather_than_guessing():
    """Returning None forces the caller to handle 'cannot execute' explicitly."""
    assert cal().next_session(date(2023, 6, 16), lookahead=1) is None


def test_missing_sessions_detected():
    c = cal()
    all_days = c.sessions(date(2023, 6, 1), date(2023, 6, 30))
    observed = pd.Series(pd.to_datetime([d for d in all_days if d != all_days[5]]))

    missing = c.missing_sessions(observed, date(2023, 6, 1), date(2023, 6, 30))
    assert all_days[5] in missing


def test_complete_history_has_no_missing_sessions():
    c = cal()
    all_days = c.sessions(date(2023, 6, 1), date(2023, 6, 30))
    observed = pd.Series(pd.to_datetime(all_days))

    assert c.missing_sessions(observed, date(2023, 6, 1), date(2023, 6, 30)) == []


def test_backend_is_reported_honestly():
    """Callers must be able to tell an approximate calendar from a real one."""
    c = cal()
    assert c.backend in ("exchange", "weekday")
    assert c.is_approximate == (c.backend == "weekday")
