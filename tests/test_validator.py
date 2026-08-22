"""Data-quality validator behaviour.

Two properties matter equally: it catches bad data, and it does not repair it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trademind.ingestion.calendar import TradingCalendar
from trademind.ingestion.corporate_actions import apply_corporate_actions
from trademind.ingestion.data_validator import DataValidator


def frame(closes, splits=None, divs=None, volume=None, start="2023-01-02"):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "symbol": "TEST.NS",
        "open_split": closes,
        "high_split": closes * 1.01,
        "low_split": closes * 0.99,
        "close_split": closes,
        "volume": np.asarray(volume if volume is not None else [100_000] * n,
                             dtype="int64"),
        "dividend": np.asarray(divs if divs is not None else [0.0] * n, dtype=float),
        "split_ratio": np.asarray(splits if splits is not None else [1.0] * n,
                                  dtype=float),
    })


def validate(df, symbol="TEST.NS"):
    return DataValidator(TradingCalendar()).validate(df, symbol)


def codes(report):
    return {i.code for i in report.issues}


# -- clean data ----------------------------------------------------------

def test_clean_data_passes():
    report = validate(frame([100 + i * 0.5 for i in range(30)]))
    assert report.ok
    assert not report.errors


def test_empty_frame_is_an_error():
    report = validate(frame([]).iloc[0:0])
    assert not report.ok
    assert "NO_DATA" in codes(report)


# -- structural checks ---------------------------------------------------

def test_unsorted_dates_flagged():
    df = frame([100.0, 101.0, 102.0]).iloc[::-1].reset_index(drop=True)
    report = validate(df)
    assert "NOT_SORTED" in codes(report)
    assert not report.ok


def test_duplicate_dates_flagged():
    df = frame([100.0, 101.0, 102.0])
    df.loc[2, "date"] = df.loc[1, "date"]
    report = validate(df)
    assert "DUPLICATE_DATES" in codes(report)


def test_negative_price_flagged():
    df = frame([100.0, -5.0, 102.0])
    report = validate(df)
    assert "INVALID_PRICE" in codes(report)
    assert not report.ok


def test_high_below_low_flagged():
    df = frame([100.0, 101.0, 102.0])
    df.loc[1, "high_split"] = 50.0
    report = validate(df)
    assert "HIGH_BELOW_LOW" in codes(report)


def test_close_outside_bar_flagged():
    df = frame([100.0, 101.0, 102.0])
    df.loc[1, "close_split"] = 500.0   # above the high
    report = validate(df)
    assert "HIGH_BELOW_CLOSE" in codes(report)


# -- volume --------------------------------------------------------------

def test_negative_volume_is_an_error():
    df = frame([100.0, 101.0], volume=[100, -5])
    report = validate(df)
    assert "NEGATIVE_VOLUME" in codes(report)
    assert not report.ok


def test_zero_volume_is_a_warning_not_an_error():
    df = frame([100.0, 101.0, 102.0], volume=[100_000, 0, 100_000])
    report = validate(df)
    assert "ZERO_VOLUME" in codes(report)
    assert report.ok          # suspicious, but does not block ingestion


# -- jumps and corporate actions -----------------------------------------

def test_unexplained_jump_flagged():
    """A 60% move with no recorded action almost certainly means a missed split."""
    df = frame([100.0, 100.0, 40.0, 40.0])
    report = validate(df)
    assert "ABNORMAL_JUMP" in codes(report)
    assert not report.ok      # >50% is an ERROR


def test_moderate_jump_warns_but_does_not_block():
    df = frame([100.0, 100.0, 75.0, 75.0])   # -25%
    report = validate(df)
    assert "ABNORMAL_JUMP" in codes(report)
    assert report.ok


def test_jump_on_a_split_date_is_not_flagged():
    """The provider already adjusted history, so the split-adjusted series is flat."""
    df = frame([500.0, 500.0, 500.0, 500.0], splits=[1.0, 1.0, 2.0, 1.0])
    report = validate(apply_corporate_actions(df))
    assert "ABNORMAL_JUMP" not in codes(report)
    assert report.ok


def test_split_that_disagrees_with_prices_is_an_error():
    """Recorded 2:1 split but the adjusted series halves anyway: double-adjusted."""
    df = frame([500.0, 500.0, 250.0, 250.0], splits=[1.0, 1.0, 2.0, 1.0])
    report = validate(df)
    assert "SPLIT_RECONCILIATION" in codes(report)
    assert not report.ok


def test_reconciled_split_logged_as_info():
    df = frame([500.0] * 4, splits=[1.0, 1.0, 2.0, 1.0])
    report = validate(df)
    assert "SPLIT_APPLIED" in codes(report)
    assert report.ok


def test_absurd_dividend_warns():
    df = frame([100.0] * 4, divs=[0.0, 0.0, 60.0, 0.0])
    report = validate(df)
    assert "LARGE_DIVIDEND" in codes(report)


# -- staleness -----------------------------------------------------------

def test_stale_price_run_warns():
    df = frame([100.0] * 8)
    report = validate(df)
    assert "STALE_PRICES" in codes(report)
    assert report.ok


def test_short_flat_run_is_fine():
    df = frame([100.0, 100.0, 101.0, 102.0, 103.0, 104.0])
    assert "STALE_PRICES" not in codes(validate(df))


# -- the no-repair guarantee ---------------------------------------------

def test_validator_never_mutates_input():
    """Report, never silently fix. Verified by comparing before and after."""
    df = frame([100.0, -5.0, 500.0, 100.0], volume=[100, 0, -3, 100])
    before = df.copy(deep=True)

    validate(df)

    pd.testing.assert_frame_equal(df, before)


# -- reporting -----------------------------------------------------------

def test_summary_counts_by_severity():
    df = frame([100.0, 101.0, 102.0], volume=[100_000, 0, 100_000])
    report = validate(df)
    assert "warning" in report.summary()


def test_clean_summary_says_clean():
    report = validate(frame([100 + i for i in range(25)]))
    assert "clean" in report.summary()
