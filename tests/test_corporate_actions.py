"""Corporate-action invariants.

Protected tests. A split must move share count and cost basis without moving
wealth; an adjusted series must be continuous where a raw series jumps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trademind.ingestion.base import ProviderError
from trademind.ingestion.corporate_actions import (
    adjust_position_for_split,
    apply_corporate_actions,
    compute_adj_close,
    cumulative_split_factor,
    reconstruct_raw,
)


def frame(closes, splits=None, divs=None, start="2023-01-02"):
    """Build a minimal well-formed bar frame from a list of closes."""
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST.NS",
            "open_split": closes,
            "high_split": closes * 1.01,
            "low_split": closes * 0.99,
            "close_split": closes,
            "volume": np.full(n, 100_000, dtype="int64"),
            "dividend": np.asarray(divs if divs is not None else [0.0] * n, dtype=float),
            "split_ratio": np.asarray(
                splits if splits is not None else [1.0] * n, dtype=float
            ),
        }
    )


# -- THE invariant -------------------------------------------------------


def test_split_preserves_position_value():
    """100 x 1000 == 200 x 500. The number this whole module exists for."""
    shares, basis = 100.0, 1000.0
    value_before = shares * basis

    new_shares, new_basis = adjust_position_for_split(shares, basis, 2.0)

    assert new_shares == 200.0
    assert new_basis == 500.0
    assert new_shares * new_basis == pytest.approx(value_before)


def test_reverse_split_preserves_position_value():
    shares, basis = 300.0, 40.0
    new_shares, new_basis = adjust_position_for_split(shares, basis, 1 / 3)

    assert new_shares == pytest.approx(100.0)
    assert new_basis == pytest.approx(120.0)
    assert new_shares * new_basis == pytest.approx(shares * basis)


def test_split_ratio_must_be_positive():
    with pytest.raises(ValueError):
        adjust_position_for_split(100, 10, 0.0)


# -- split factor --------------------------------------------------------


def test_cumulative_factor_excludes_own_row():
    """The ex-date already trades post-split, so its own ratio must not apply."""
    ratios = pd.Series([1.0, 1.0, 2.0, 1.0, 1.0])
    factor = cumulative_split_factor(ratios)
    assert list(factor) == [2.0, 2.0, 1.0, 1.0, 1.0]


def test_cumulative_factor_compounds_multiple_splits():
    ratios = pd.Series([1.0, 2.0, 1.0, 5.0, 1.0])
    factor = cumulative_split_factor(ratios)
    assert list(factor) == [10.0, 5.0, 5.0, 1.0, 1.0]


def test_no_splits_gives_unit_factor():
    factor = cumulative_split_factor(pd.Series([1.0] * 4))
    assert list(factor) == [1.0, 1.0, 1.0, 1.0]


# -- raw reconstruction --------------------------------------------------


def test_raw_prices_recover_pre_split_levels():
    """Provider shows a continuous 500-ish series; as-traded was 1000 then 500."""
    df = frame([500.0, 500.0, 500.0, 500.0], splits=[1.0, 1.0, 2.0, 1.0])
    out = reconstruct_raw(df)

    assert list(out["close_raw"]) == [1000.0, 1000.0, 500.0, 500.0]


def test_raw_reconstruction_is_identity_without_splits():
    df = frame([100.0, 101.0, 102.0])
    out = reconstruct_raw(df)
    assert np.allclose(out["close_raw"], out["close_split"])


def test_all_four_ohlc_columns_reconstructed():
    df = frame([500.0, 500.0], splits=[1.0, 2.0])
    out = reconstruct_raw(df)
    for col in ("open_raw", "high_raw", "low_raw", "close_raw"):
        assert col in out.columns
    assert out["high_raw"].iloc[0] == pytest.approx(500.0 * 1.01 * 2)


def test_negative_split_ratio_rejected():
    df = frame([100.0, 100.0], splits=[1.0, -2.0])
    with pytest.raises(ProviderError, match="Non-positive"):
        reconstruct_raw(df)


def test_absurd_split_ratio_rejected():
    df = frame([100.0, 100.0], splits=[1.0, 5000.0])
    with pytest.raises(ProviderError, match="plausible range"):
        reconstruct_raw(df)


def test_negative_dividend_rejected():
    df = frame([100.0, 100.0], divs=[0.0, -5.0])
    with pytest.raises(ProviderError, match="Negative dividend"):
        reconstruct_raw(df)


# -- total return series -------------------------------------------------


def test_adj_close_equals_close_when_no_dividends():
    df = frame([100.0, 102.0, 101.0])
    out = compute_adj_close(df)
    assert np.allclose(out["adj_close"], out["close_split"])


def test_dividend_lifts_historical_prices_only():
    """A dividend discounts prior prices; the latest bar is never restated."""
    df = frame([100.0, 100.0, 100.0], divs=[0.0, 0.0, 10.0])
    out = compute_adj_close(df)

    assert out["adj_close"].iloc[-1] == pytest.approx(100.0)
    # Prior closes scaled by (1 - 10/100).
    assert out["adj_close"].iloc[0] == pytest.approx(90.0)
    assert out["adj_close"].iloc[1] == pytest.approx(90.0)


def test_total_return_captures_the_dividend():
    """Price flat, Rs.10 paid on Rs.100: total return over the window is +10%."""
    df = frame([100.0, 100.0], divs=[0.0, 10.0])
    out = compute_adj_close(df)

    total_return = out["adj_close"].iloc[-1] / out["adj_close"].iloc[0] - 1
    assert total_return == pytest.approx(0.1111, abs=1e-4)

    # Price-only return over the same window is zero, which is the whole point.
    price_return = out["close_split"].iloc[-1] / out["close_split"].iloc[0] - 1
    assert price_return == pytest.approx(0.0)


# -- combined ------------------------------------------------------------


def test_pipeline_produces_all_three_series():
    df = frame([500.0] * 5, splits=[1.0, 1.0, 2.0, 1.0, 1.0], divs=[0.0, 0.0, 0.0, 0.0, 5.0])
    out = apply_corporate_actions(df)

    for col in ("close_raw", "close_split", "adj_close"):
        assert col in out.columns

    # Raw jumps at the split; adjusted series does not.
    assert out["close_raw"].iloc[1] == pytest.approx(1000.0)
    assert out["close_raw"].iloc[2] == pytest.approx(500.0)
    assert out["close_split"].iloc[1] == out["close_split"].iloc[2]


def test_unsorted_input_rejected():
    df = frame([100.0, 101.0, 102.0]).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ProviderError, match="date-sorted"):
        apply_corporate_actions(df)


def test_empty_frame_passes_through():
    out = apply_corporate_actions(frame([]).iloc[0:0])
    assert out.empty
